"""Hypothesis strategies for generating (Template, DataSchema) pairs."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import hypothesis.strategies as st
from hypothesis.strategies import SearchStrategy

from westlean.content_model import ELEMENTS, get_valid_children
from westlean.field_strategies import field_strategy
from westlean.data_schema import (
    BoolField,
    DataSchema,
    ListField,
    ObjectField,
    SchemaField,
    StringField,
    UrlField,
)
from westlean.template_ast import (
    AttributeValue,
    ConditionalBlock,
    Element,
    LoopBlock,
    TemplateNode,
    TemplateVar,
    TextNode,
)

# Module-level flags for constraining loop body generation.
_single_element_loops: ContextVar[bool] = ContextVar(
    "_single_element_loops", default=False
)
_max_loop_depth: ContextVar[int] = ContextVar("_max_loop_depth", default=1)
_min_loop_length: ContextVar[int] = ContextVar("_min_loop_length", default=0)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SAFE_TEXT = st.text(
    st.characters(whitelist_categories=("L", "N", "Zs"), blacklist_characters="<>&\"'"),
    min_size=1,
    max_size=30,
)

_SAFE_ATTR_TEXT = st.text(
    st.characters(whitelist_categories=("L", "N"), blacklist_characters="<>&\"'"),
    min_size=1,
    max_size=15,
)

_CSS_CLASS = st.from_regex(r"[a-z][a-z0-9-]{1,15}", fullmatch=True)


def _fresh_name(prefix: str, used: set[str]) -> str:
    """Return a unique name starting with *prefix*."""
    for i in range(200):
        name = prefix if i == 0 else f"{prefix}_{i}"
        if name not in used:
            used.add(name)
            return name
    raise RuntimeError("name exhaustion")


# ---------------------------------------------------------------------------
# Attribute generation
# ---------------------------------------------------------------------------


@st.composite
def _attributes(
    draw: st.DrawFn,
    elem_def: object,  # ElementDef - avoid circular for type-checkers
    used_names: set[str],
    schema_fields: dict[str, SchemaField],
    loop_context: str | None,
) -> dict[str, AttributeValue]:
    """Generate attributes for an element, mixing static and dynamic values."""
    attrs: dict[str, AttributeValue] = {}
    from westlean.content_model import ElementDef  # noqa: F811

    assert isinstance(elem_def, ElementDef)

    for attr_spec in elem_def.attributes:
        include = attr_spec.required or draw(st.booleans())
        if not include:
            continue

        if attr_spec.values:
            val = draw(st.sampled_from(attr_spec.values))
            attrs[attr_spec.name] = AttributeValue(parts=(val,))
        elif attr_spec.templatable and draw(st.booleans()):
            name = _fresh_name(f"attr_{attr_spec.name}", used_names)
            if attr_spec.name in ("href", "src"):
                field_type: SchemaField = UrlField()
            else:
                field_type = StringField()
            schema_fields[name] = field_type
            if loop_context:
                path = f"{loop_context}.{name}"
            else:
                path = name
            attrs[attr_spec.name] = AttributeValue(parts=(TemplateVar(path=path),))
        else:
            val = draw(_SAFE_ATTR_TEXT)
            attrs[attr_spec.name] = AttributeValue(parts=(val,))

    # Optionally add a class attribute
    if draw(st.integers(min_value=0, max_value=3)) == 0:
        attrs["class"] = AttributeValue(parts=(draw(_CSS_CLASS),))

    return attrs


# ---------------------------------------------------------------------------
# Node strategies
# ---------------------------------------------------------------------------


@st.composite
def _element(
    draw: st.DrawFn,
    tag: str,
    depth: int,
    used_names: set[str],
    schema_fields: dict[str, SchemaField],
    loop_context: str | None = None,
    allow_loops: bool = True,
    allow_conditionals: bool = True,
    loop_depth: int = 0,
) -> Element:
    """Generate an Element with valid children per the content model."""
    elem_def = ELEMENTS[tag]
    attrs = draw(_attributes(elem_def, used_names, schema_fields, loop_context))

    if elem_def.void or depth <= 0:
        return Element(tag=tag, attributes=attrs, children=())

    valid_tags = get_valid_children(tag)

    max_kids = max(elem_def.min_children, min(5, 2 + depth))

    # Decide whether this element will contain a loop child.
    # When it does, all other children must be Elements (not TextNode/TemplateVar)
    # to prevent lxml text/tail redistribution when the loop is empty.
    include_loop = (
        allow_loops
        and loop_depth < _max_loop_depth.get()
        and bool(valid_tags)
        and depth >= 1
        and draw(st.booleans())
    )

    if include_loop:
        # LOOP MODE: Element-only siblings + one LoopBlock.
        # Generate the loop first so we know its body element tags,
        # then exclude those tags from siblings to prevent collision.
        loop = draw(
            _loop(
                valid_tags,
                depth - 1,
                used_names,
                schema_fields,
                single_element=_single_element_loops.get(),
                loop_depth=loop_depth,
                loop_context=loop_context,
            )
        )
        loop_body_tags = {c.tag for c in loop.children if isinstance(c, Element)}
        safe_sibling_tags = [t for t in valid_tags if t not in loop_body_tags]

        if safe_sibling_tags:
            num_siblings = draw(
                st.integers(
                    min_value=max(0, elem_def.min_children - 1),
                    max_value=max(0, max_kids - 1),
                )
            )
        else:
            num_siblings = max(0, elem_def.min_children - 1)

        children: list[TemplateNode] = []
        for _ in range(num_siblings):
            child_tag = draw(st.sampled_from(safe_sibling_tags))
            child = draw(
                _element(
                    child_tag,
                    depth - 1,
                    used_names,
                    schema_fields,
                    loop_context,
                    allow_loops=allow_loops,
                    allow_conditionals=allow_conditionals,
                    loop_depth=loop_depth,
                )
            )
            children.append(child)

        insert_pos = draw(st.integers(min_value=0, max_value=len(children)))
        children.insert(insert_pos, loop)

        return Element(tag=tag, attributes=attrs, children=tuple(children))

    # NORMAL MODE: no loop at this level, regular text/var/element/conditional mix.
    num_children = draw(
        st.integers(min_value=elem_def.min_children, max_value=max_kids)
    )

    children = []
    for _ in range(num_children):
        child = draw(
            _template_node(
                valid_tags,
                elem_def.accepts_text,
                depth - 1,
                used_names,
                schema_fields,
                loop_context,
                allow_loops=allow_loops,
                allow_conditionals=allow_conditionals,
                loop_depth=loop_depth,
            )
        )
        children.append(child)

    return Element(tag=tag, attributes=attrs, children=tuple(children))


@st.composite
def _template_node(
    draw: st.DrawFn,
    valid_child_tags: list[str],
    accepts_text: bool,
    depth: int,
    used_names: set[str],
    schema_fields: dict[str, SchemaField],
    loop_context: str | None = None,
    allow_loops: bool = True,
    allow_conditionals: bool = True,
    loop_depth: int = 0,
) -> TemplateNode:
    """Generate any template node type."""
    # Build available node types based on context
    if depth <= 0 or not valid_child_tags:
        # Leaf-only: text or template var
        if accepts_text:
            kind = draw(st.sampled_from(["text", "var"]))
        else:
            # Parent doesn't accept text (e.g. ul, table).
            # Must produce a child element even at depth 0.
            if valid_child_tags:
                kind = "element"
            else:
                kind = "text"  # fallback
        return draw(
            _leaf_or_element(
                kind,
                valid_child_tags,
                depth,
                used_names,
                schema_fields,
                loop_context,
                allow_loops,
                allow_conditionals,
                loop_depth,
            )
        )

    choices = []
    if accepts_text:
        choices += ["text", "var"]
    # Elements get double weight
    choices += ["element", "element"]
    if accepts_text and allow_conditionals:
        choices += ["conditional"]
    # Loops are NOT offered here — they are placed exclusively by _element()
    # in loop mode, which ensures all loop siblings are Elements (preventing
    # lxml text/tail redistribution when loops are empty).

    kind = draw(st.sampled_from(choices))
    return draw(
        _leaf_or_element(
            kind,
            valid_child_tags,
            depth,
            used_names,
            schema_fields,
            loop_context,
            allow_loops,
            allow_conditionals,
            loop_depth,
        )
    )


@st.composite
def _leaf_or_element(
    draw: st.DrawFn,
    kind: str,
    valid_child_tags: list[str],
    depth: int,
    used_names: set[str],
    schema_fields: dict[str, SchemaField],
    loop_context: str | None,
    allow_loops: bool = True,
    allow_conditionals: bool = True,
    loop_depth: int = 0,
) -> TemplateNode:
    """Dispatch to the right sub-strategy based on *kind*."""
    if kind == "text":
        return TextNode(text=draw(_SAFE_TEXT))

    if kind == "var":
        name = _fresh_name("text_var", used_names)
        schema_fields[name] = StringField()
        if loop_context:
            return TemplateVar(path=f"{loop_context}.{name}")
        return TemplateVar(path=name)

    if kind == "element":
        child_tag = (
            draw(st.sampled_from(valid_child_tags)) if valid_child_tags else "span"
        )
        return draw(
            _element(
                child_tag,
                depth - 1,
                used_names,
                schema_fields,
                loop_context,
                allow_loops=allow_loops,
                allow_conditionals=allow_conditionals,
                loop_depth=loop_depth,
            )
        )

    if kind == "conditional":
        return draw(
            _conditional(
                valid_child_tags, depth - 1, used_names, schema_fields, loop_context
            )
        )

    raise ValueError(f"Unknown node kind: {kind}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Conditional strategy
# ---------------------------------------------------------------------------


@st.composite
def _conditional(
    draw: st.DrawFn,
    valid_child_tags: list[str],
    depth: int,
    used_names: set[str],
    schema_fields: dict[str, SchemaField],
    loop_context: str | None = None,
) -> ConditionalBlock:
    """Generate a conditional block with a BoolField predicate."""
    pred_name = _fresh_name("show", used_names)
    if loop_context:
        predicate_path = f"{loop_context}.{pred_name}"
    else:
        schema_fields[pred_name] = BoolField()
        predicate_path = pred_name

    num_then = draw(st.integers(min_value=1, max_value=2))
    then_children: list[TemplateNode] = []
    for _ in range(num_then):
        child = draw(
            _template_node(
                valid_child_tags, True, depth, used_names, schema_fields, loop_context
            )
        )
        then_children.append(child)

    has_else = draw(st.booleans())
    else_children: list[TemplateNode] = []
    if has_else:
        child = draw(
            _template_node(
                valid_child_tags, True, depth, used_names, schema_fields, loop_context
            )
        )
        else_children.append(child)

    return ConditionalBlock(
        predicate_path=predicate_path,
        children=tuple(then_children),
        else_children=tuple(else_children),
    )


# ---------------------------------------------------------------------------
# Loop strategy
# ---------------------------------------------------------------------------


@st.composite
def _loop(
    draw: st.DrawFn,
    valid_child_tags: list[str],
    depth: int,
    used_names: set[str],
    schema_fields: dict[str, SchemaField],
    single_element: bool = False,
    loop_depth: int = 0,
    loop_context: str | None = None,
) -> LoopBlock:
    """Generate a loop block with its own item schema."""
    list_name = _fresh_name("items", used_names)
    item_var = draw(st.sampled_from(["item", "entry", "row"]))

    item_fields: dict[str, SchemaField] = {}
    item_used: set[str] = set()

    if single_element:
        # v2: single element per iteration for simpler loop detection
        child = draw(
            _loop_body_node(
                valid_child_tags,
                depth,
                item_used,
                item_fields,
                item_var,
                force_element=True,
                loop_depth=loop_depth + 1,
            )
        )
        children: list[TemplateNode] = [child]
    else:
        num_children = draw(st.integers(min_value=1, max_value=3))
        children = []
        for _ in range(num_children):
            child = draw(
                _loop_body_node(
                    valid_child_tags,
                    depth,
                    item_used,
                    item_fields,
                    item_var,
                    loop_depth=loop_depth + 1,
                )
            )
            children.append(child)

    item_schema = ObjectField(fields=dict(item_fields))
    schema_fields[list_name] = ListField(
        item_schema=item_schema,
        min_length=_min_loop_length.get(),
    )

    # When inside a loop body, the list path must be relative to the
    # outer loop's item variable (e.g. "item.subitems" not "subitems").
    if loop_context:
        list_path = f"{loop_context}.{list_name}"
    else:
        list_path = list_name

    return LoopBlock(
        list_path=list_path,
        item_var=item_var,
        children=tuple(children),
    )


@st.composite
def _loop_body_node(
    draw: st.DrawFn,
    valid_child_tags: list[str],
    depth: int,
    item_used: set[str],
    item_fields: dict[str, SchemaField],
    item_var: str,
    force_element: bool = False,
    loop_depth: int = 1,
) -> TemplateNode:
    """Generate nodes inside a loop body.

    Template vars here contribute to *item_fields* rather than the
    top-level schema.
    """
    if force_element:
        kind = "element"
    elif depth <= 0 or not valid_child_tags:
        kind = draw(st.sampled_from(["text", "var"]))
    else:
        kind = draw(st.sampled_from(["text", "var", "element", "element"]))

    if kind == "text":
        return TextNode(text=draw(_SAFE_TEXT))

    if kind == "var":
        name = _fresh_name("field", item_used)
        item_fields[name] = StringField()
        return TemplateVar(path=f"{item_var}.{name}")

    # Element inside loop body — use _element() so it can contain inner
    # loops when loop_depth < max_loop_depth.
    child_tag = draw(st.sampled_from(valid_child_tags)) if valid_child_tags else "span"
    return draw(
        _element(
            child_tag,
            depth - 1,
            item_used,
            item_fields,
            loop_context=item_var,
            allow_loops=True,
            allow_conditionals=False,
            loop_depth=loop_depth,
        )
    )


def _has_loop_generation_hazard(template: TemplateNode) -> bool:
    """Safety check: verify the generator produced a clean loop structure.

    Returns True if any of these patterns are found:
    1. Non-Element sibling (TextNode/TemplateVar/ConditionalBlock) of a LoopBlock
    2. Multiple LoopBlocks in the same parent
    3. Loop body element tag collides with a sibling element tag
    """
    if not isinstance(template, Element):
        return False

    children = template.children
    has_loop = False
    loop_body_tags: set[str] = set()
    sibling_tags: set[str] = set()
    loop_count = 0

    for child in children:
        if isinstance(child, LoopBlock):
            has_loop = True
            loop_count += 1
            for lc in child.children:
                if isinstance(lc, Element):
                    loop_body_tags.add(lc.tag)
        elif isinstance(child, Element):
            sibling_tags.add(child.tag)
        elif isinstance(child, (TextNode, TemplateVar, ConditionalBlock)):
            if has_loop or any(isinstance(c, LoopBlock) for c in children):
                return True

    if has_loop:
        if loop_count > 1:
            return True
        if loop_body_tags & sibling_tags:
            return True

    # Recurse into child elements and loop bodies
    for child in children:
        if isinstance(child, Element) and _has_loop_generation_hazard(child):
            return True
        if isinstance(child, LoopBlock):
            for lc in child.children:
                if isinstance(lc, Element) and _has_loop_generation_hazard(lc):
                    return True
    return False


def has_loop_alignment_hazard(template: TemplateNode) -> bool:
    """Check if template has loop patterns that break alignment or extraction.

    Returns True if any of these patterns are found:
    1. Loop body element shares a tag with a fixed sibling element
    2. Two loops are adjacent with no Element separator between them
    3. TextNode/TemplateVar/ConditionalBlock adjacent to a LoopBlock
       (text merges when loop is empty due to lxml text/tail redistribution)
    """
    if not isinstance(template, Element):
        return False

    loop_tags: set[str] = set()
    sibling_tags: set[str] = set()
    last_was_loop = False
    adjacent_loops = False
    children = template.children

    has_loop = False
    for i, child in enumerate(children):
        if isinstance(child, LoopBlock):
            has_loop = True
            if last_was_loop:
                adjacent_loops = True
            last_was_loop = True
            for lc in child.children:
                if isinstance(lc, Element):
                    loop_tags.add(lc.tag)
        elif isinstance(child, Element):
            sibling_tags.add(child.tag)
            last_was_loop = False
        else:
            pass  # TextNode/TemplateVar/ConditionalBlock don't create DOM elements

    # Non-Element siblings of loops cause text/tail redistribution when
    # the loop is empty: TextNode/TemplateVar merge into different positions,
    # and ConditionalBlock can render text content with the same effect.
    if has_loop:
        for child in children:
            if isinstance(child, (TextNode, TemplateVar, ConditionalBlock)):
                return True

    if loop_tags & sibling_tags:
        return True
    if adjacent_loops:
        return True

    # Recurse into child elements and loop bodies
    for child in children:
        if isinstance(child, Element) and has_loop_alignment_hazard(child):
            return True
        if isinstance(child, LoopBlock):
            for lc in child.children:
                if isinstance(lc, Element) and has_loop_alignment_hazard(lc):
                    return True
    return False


# ---------------------------------------------------------------------------
# Public strategies
# ---------------------------------------------------------------------------

_ROOT_TAGS = ["div", "section", "article", "main", "nav", "aside"]


@st.composite
def template_and_schema(
    draw: st.DrawFn,
    max_depth: int = 4,
    root_tags: list[str] | None = None,
    allow_loops: bool = True,
    allow_conditionals: bool = True,
    single_element_loops: bool = False,
    max_loop_depth: int = 1,
    min_loop_length: int = 0,
) -> tuple[Element, DataSchema]:
    """Generate a ``(template_root, data_schema)`` pair.

    The template root is an :class:`Element` whose tree respects the HTML5
    content model.  The schema describes the data needed to render it.

    Set *allow_loops* / *allow_conditionals* to ``False`` to restrict
    generated templates to fixed-structure trees (text and attribute
    variables only).

    Set *single_element_loops* to ``True`` to restrict loop bodies to a
    single child element (used by v2 algorithms that detect single-element
    repeating patterns).

    Set *max_loop_depth* to control nesting: 1 = flat loops only (default),
    2 = one level of nested loops, etc.

    Set *min_loop_length* to guarantee a minimum number of loop iterations
    in generated data.  When set to 1, no training page will have an empty
    loop, ensuring the DOM tree always contains the loop body element(s).
    This produces templates compatible with Plotkin's LGG algorithm, which
    requires same-arity (compatible) inputs at every node.
    """
    tags = root_tags or _ROOT_TAGS
    used_names: set[str] = set()
    schema_fields: dict[str, SchemaField] = {}

    tok_sel = _single_element_loops.set(single_element_loops)
    tok_mld = _max_loop_depth.set(max_loop_depth)
    tok_mll = _min_loop_length.set(min_loop_length)
    try:
        root_tag = draw(st.sampled_from(tags))
        root = draw(
            _element(
                root_tag,
                max_depth,
                used_names,
                schema_fields,
                allow_loops=allow_loops,
                allow_conditionals=allow_conditionals,
            )
        )
    finally:
        _single_element_loops.reset(tok_sel)
        _max_loop_depth.reset(tok_mld)
        _min_loop_length.reset(tok_mll)

    # Safety invariant: the generator should never produce templates where
    # TextNode/TemplateVar/ConditionalBlock is a sibling of a LoopBlock.
    # (Tag collision is still possible and is a known limitation.)
    if allow_loops:
        assert not _has_loop_generation_hazard(root), (
            "Generator produced a template with a loop alignment hazard"
        )

    schema = ObjectField(fields=dict(schema_fields))
    return root, schema


def template_with_data(
    max_depth: int = 4,
    root_tags: list[str] | None = None,
    allow_loops: bool = True,
    allow_conditionals: bool = True,
) -> SearchStrategy[tuple[Element, DataSchema, dict[str, Any]]]:
    """Generate a ``(template_root, schema, data)`` triple.

    Uses the *flatmap* pattern: first generate ``(template, schema)``, then
    generate matching data from the schema.
    """
    return template_and_schema(
        max_depth=max_depth,
        root_tags=root_tags,
        allow_loops=allow_loops,
        allow_conditionals=allow_conditionals,
    ).flatmap(
        lambda pair: field_strategy(pair[1]).map(lambda data: (pair[0], pair[1], data))
    )
