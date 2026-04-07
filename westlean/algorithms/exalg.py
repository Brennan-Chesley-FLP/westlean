"""EXALG template inference via token-frequency equivalence classes.

Based on Arasu & Garcia-Molina (SIGMOD 2003). Linearizes pages into
token streams, computes occurrence-frequency vectors, groups tokens into
equivalence classes, and builds a template from the class structure.

Unlike the tree-based algorithms (FiVaTech, Anti-Unification, k-Testable),
ExAlg works on **linearized** token streams and discovers template structure
from **frequency patterns** rather than tree alignment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from lxml import etree

from westlean.compat import element_tag
from westlean.protocol import EmptyTemplate


# ---------------------------------------------------------------------------
# Token representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """A token from a linearized HTML page.

    Carries both structural info (for template matching) and positional
    info (for harness output).
    """

    kind: str  # 'open', 'close', 'text', 'tail', 'attr'
    tag: str  # element tag name
    attr_name: str  # attribute name (for 'attr' kind)
    value: str  # text/attribute value content
    position_key: str  # DOM position key (e.g., "0/text")
    context: str  # tag-path context for DiffFormat (e.g., "div/h1")

    @property
    def structural_key(self) -> tuple[str, str, str]:
        """Structural identity: (kind, tag, attr_name)."""
        return (self.kind, self.tag, self.attr_name)


# ---------------------------------------------------------------------------
# Template element types
# ---------------------------------------------------------------------------


@dataclass
class Literal:
    """Fixed template content that must match exactly."""

    token: Token


@dataclass
class Var:
    """Variable data position — extracts a value."""

    name: str
    kind: str
    tag: str
    attr_name: str
    position_key: str  # representative position key
    always_has_value: bool = True


@dataclass
class Set:
    """Repeating pattern (loop). Body matches 0+ times."""

    body: list  # list[TemplateElement]
    var_name: str


@dataclass
class Optional:
    """Content present in some pages but not others."""

    elements: list  # list[TemplateElement]


TemplateElement = Literal | Var | Set | Optional


# ---------------------------------------------------------------------------
# Tokenization (linearize DOM → token stream)
# ---------------------------------------------------------------------------


def _linearize(root: etree._Element, prefix: str = "", ctx: str = "") -> list[Token]:
    """Convert an lxml element tree to a flat token stream.

    Each token carries a ``context`` (tag-path like "div/h1") for
    DiffFormat grouping and a ``position_key`` for harness output.
    """
    tokens: list[Token] = []
    _linearize_elem(root, prefix, ctx, tokens)
    return tokens


def _linearize_elem(
    elem: etree._Element, prefix: str, ctx: str, out: list[Token]
) -> None:
    tag = element_tag(elem)
    tag_ctx = f"{ctx}/{tag}" if ctx else tag

    out.append(Token("open", tag, "", "", prefix, tag_ctx))

    for attr_name in sorted(elem.attrib):
        key = f"{prefix}/@{attr_name}" if prefix else f"@{attr_name}"
        out.append(Token("attr", "", attr_name, elem.attrib[attr_name], key, tag_ctx))

    text_key = f"{prefix}/text" if prefix else "text"
    out.append(Token("text", "", "", elem.text or "", text_key, tag_ctx))

    for i, child in enumerate(elem):
        child_prefix = f"{prefix}/{i}" if prefix else str(i)
        _linearize_elem(child, child_prefix, tag_ctx, out)
        # Include child's tag in tail token so tail-after-h1 differs from tail-after-p
        out.append(
            Token(
                "tail",
                element_tag(child),
                "",
                child.tail or "",
                f"{child_prefix}/tail",
                tag_ctx,
            )
        )

    out.append(Token("close", tag, "", "", prefix, tag_ctx))


# ---------------------------------------------------------------------------
# ECGM module: equivalence classes from occurrence vectors
# ---------------------------------------------------------------------------


def _structural_key_vector(
    streams: list[list[Token]],
) -> dict[tuple[str, str, str, str], tuple[int, ...]]:
    """Compute occurrence vector for each (structural_key, context) across pages.

    Returns mapping from (kind, tag, attr_name, context) → count-per-page tuple.
    """
    all_keys: set[tuple[str, str, str, str]] = set()
    per_page: list[Counter[tuple[str, str, str, str]]] = []
    for stream in streams:
        counter: Counter[tuple[str, str, str, str]] = Counter()
        for tok in stream:
            key = (*tok.structural_key, tok.context)
            counter[key] += 1
            all_keys.add(key)
        per_page.append(counter)

    vectors: dict[tuple[str, str, str, str], tuple[int, ...]] = {}
    for key in all_keys:
        vectors[key] = tuple(c[key] for c in per_page)

    return vectors


def _classify_structural_keys(
    vectors: dict[tuple[str, str, str, str], tuple[int, ...]],
    n_pages: int,
) -> tuple[
    set[tuple[str, str, str, str]],  # template constants: (1,1,...,1) or fixed count
    set[tuple[str, str, str, str]],  # loop markers: varying count, all > 0
    set[tuple[str, str, str, str]],  # optional markers: some 0s
]:
    """Classify structural keys by their occurrence vector pattern."""
    one_vec = (1,) * n_pages
    template_keys: set[tuple[str, str, str, str]] = set()
    loop_keys: set[tuple[str, str, str, str]] = set()
    optional_keys: set[tuple[str, str, str, str]] = set()

    for key, vec in vectors.items():
        if vec == one_vec:
            template_keys.add(key)
        elif all(v > 0 for v in vec) and len(set(vec)) > 1:
            loop_keys.add(key)
        elif any(v == 0 for v in vec):
            optional_keys.add(key)
        elif len(set(vec)) == 1:
            # Fixed count > 1 (e.g., (3,3,3)) — consistent structure
            template_keys.add(key)

    return template_keys, loop_keys, optional_keys


# ---------------------------------------------------------------------------
# DiffFormat: context refinement (paper Section 4.3, Obs. 4.4)
# ---------------------------------------------------------------------------


def _refine_contexts(
    streams: list[list[Token]],
    template_keys: set[tuple[str, str, str, str]],
    vectors: dict[tuple[str, str, str, str], tuple[int, ...]],
) -> list[list[Token]]:
    """Add sibling-index disambiguation to tokens within fixed-count parents.

    When a template-constant element appears N > 1 times per page (e.g.,
    two ``<tbody>`` children of a ``<table>``), all descendants share the
    same tag-path context.  This conflates tokens from different instances.

    This function rewrites contexts so that descendants of the first
    ``<tbody>`` get context ``…/tbody#0/…`` and the second gets
    ``…/tbody#1/…``, allowing frequency analysis to distinguish them.

    Only elements classified as template constants with fixed count > 1
    are indexed — loop-varying parents have unstable sibling positions.
    """
    # Identify open tokens that need sibling indexing: template constant,
    # consistent count > 1 across all pages.
    need_index: set[tuple[str, str, str, str]] = set()
    for sk in template_keys:
        if sk[0] == "open" and sk in vectors:
            vec = vectors[sk]
            if len(set(vec)) == 1 and vec[0] > 1:
                need_index.add(sk)

    if not need_index:
        return streams

    refined: list[list[Token]] = []
    for stream in streams:
        new_stream: list[Token] = []
        # Stack: (tag, refined_context) for each open element in scope
        ctx_stack: list[tuple[str, str]] = []
        # Per-scope same-tag sibling counters (one dict per stack level)
        sib_counters: list[dict[str, int]] = [{}]

        for tok in stream:
            if tok.kind == "open":
                orig_sk = (*tok.structural_key, tok.context)
                parent_ctx = ctx_stack[-1][1] if ctx_stack else ""

                if orig_sk in need_index:
                    # Indexed element — assign same-tag sibling number
                    counters = sib_counters[-1]
                    idx = counters.get(tok.tag, 0)
                    counters[tok.tag] = idx + 1
                    new_ctx = (
                        f"{parent_ctx}/{tok.tag}#{idx}"
                        if parent_ctx
                        else f"{tok.tag}#{idx}"
                    )
                else:
                    new_ctx = f"{parent_ctx}/{tok.tag}" if parent_ctx else tok.tag

                ctx_stack.append((tok.tag, new_ctx))
                sib_counters.append({})
                new_stream.append(
                    Token(
                        tok.kind,
                        tok.tag,
                        tok.attr_name,
                        tok.value,
                        tok.position_key,
                        new_ctx,
                    )
                )

            elif tok.kind == "close":
                ctx = ctx_stack[-1][1] if ctx_stack else tok.context
                new_stream.append(
                    Token(
                        tok.kind,
                        tok.tag,
                        tok.attr_name,
                        tok.value,
                        tok.position_key,
                        ctx,
                    )
                )
                if ctx_stack:
                    ctx_stack.pop()
                    sib_counters.pop()

            elif tok.kind == "tail":
                # Tail belongs to the parent scope (after the closed child)
                parent_ctx = ctx_stack[-1][1] if ctx_stack else ""
                new_stream.append(
                    Token(
                        tok.kind,
                        tok.tag,
                        tok.attr_name,
                        tok.value,
                        tok.position_key,
                        parent_ctx,
                    )
                )

            else:
                # text, attr — belong to the current element
                ctx = ctx_stack[-1][1] if ctx_stack else tok.context
                new_stream.append(
                    Token(
                        tok.kind,
                        tok.tag,
                        tok.attr_name,
                        tok.value,
                        tok.position_key,
                        ctx,
                    )
                )

        refined.append(new_stream)
    return refined


# ---------------------------------------------------------------------------
# ConstTemp: build template from EQ class analysis
# ---------------------------------------------------------------------------


class _Counter:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> str:
        name = f"var_{self._n}"
        self._n += 1
        return name


def _is_value_token(kind: str) -> bool:
    return kind in ("text", "tail", "attr")


def _split_iterations(tokens: list[Token]) -> list[list[Token]]:
    """Split a token sequence into loop iterations using open/close depth.

    Each iteration consists of an 'open' tag at depth 0, its content,
    the matching 'close' tag, and the immediately following 'tail' token
    (if present).
    """
    iterations: list[list[Token]] = []
    current: list[Token] = []
    depth = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        current.append(tok)
        if tok.kind == "open":
            depth += 1
        elif tok.kind == "close":
            depth -= 1
            if depth == 0:
                # Include trailing tail token if present
                if i + 1 < len(tokens) and tokens[i + 1].kind == "tail":
                    current.append(tokens[i + 1])
                    i += 1
                iterations.append(current)
                current = []
        i += 1
    if current:
        if iterations:
            iterations[-1].extend(current)
        else:
            iterations.append(current)
    return iterations


def _build_template(
    streams: list[list[Token]],
    ctr: _Counter,
    _depth: int = 0,
) -> list[TemplateElement]:
    """Build a linearized template from multiple page token streams.

    Uses frequency analysis to identify template-constant tokens (the
    skeleton), then analyzes gaps between skeleton tokens for loops,
    optionals, and variables.
    """
    if _depth > 10:
        # Prevent infinite recursion from degenerate cases
        return []

    n_pages = len(streams)
    if n_pages == 0:
        return []

    # --- ECGM: compute structural key vectors and classify ---
    vectors = _structural_key_vector(streams)
    template_keys, loop_keys, optional_keys = _classify_structural_keys(
        vectors, n_pages
    )

    # --- DiffFormat: refine contexts for fixed-count sibling disambiguation ---
    # When a template-constant element appears N>1 times (e.g., two <tbody>),
    # index each instance so descendants get distinct structural keys.
    streams = _refine_contexts(streams, template_keys, vectors)
    vectors = _structural_key_vector(streams)
    template_keys, loop_keys, optional_keys = _classify_structural_keys(
        vectors, n_pages
    )

    # --- HandInv: demote skeleton tokens nested inside loop-body elements ---
    # A template-key token physically inside a loop element (between a
    # loop-key open and its matching close) is invalid — its (1,1,...,1)
    # vector is coincidental (e.g., an attribute on the first loop item).
    loop_open_keys = {sk for sk in loop_keys if sk[0] == "open"}
    loop_close_keys = {sk for sk in loop_keys if sk[0] == "close"}
    demoted: set[tuple[str, str, str, str]] = set()

    for stream in streams:
        depth = 0
        for tok in stream:
            sk = (*tok.structural_key, tok.context)
            if tok.kind == "open" and sk in loop_open_keys:
                depth += 1
            elif tok.kind == "close" and sk in loop_close_keys:
                depth -= 1
            elif sk in template_keys and depth > 0:
                demoted.add(sk)

    template_keys -= demoted

    # --- DiffEq: promote first instance of loop elements with fixed values ---
    # When a loop-key element group has the same value in its first
    # instance across all pages but different values in later instances,
    # the first instance plays a different role (template constant) from
    # the rest (loop body data).  Promote the first instance of all
    # tokens in that element group to the skeleton.
    promoted_first: set[tuple[str, str, str, str]] = set()

    loop_by_context: dict[str, list[tuple[str, str, str, str]]] = {}
    for sk in loop_keys:
        loop_by_context.setdefault(sk[3], []).append(sk)

    # Collect contexts of loop-key OPEN tokens for nesting checks.
    # Only element-level contexts (open tokens) indicate actual loop
    # bodies — tail tokens at a parent context don't create nesting.
    loop_element_contexts = {sk[3] for sk in loop_keys if sk[0] == "open"}

    for context, group in loop_by_context.items():
        # Skip if this context is nested inside another loop element
        # (e.g., don't promote option tokens nested inside a loop optgroup)
        nested_in_loop = any(
            context != lc and context.startswith(lc + "/")
            for lc in loop_element_contexts
        )
        if nested_in_loop:
            continue

        # Only promote when every page has at least 2 instances.
        # If min count is 1, the "first instance" is just the sole
        # loop iteration — not a distinguishable fixed element.
        open_sk = next((sk for sk in group if sk[0] == "open"), None)
        if open_sk and open_sk in vectors and min(vectors[open_sk]) < 2:
            continue

        value_sks = [sk for sk in group if _is_value_token(sk[0])]
        if not value_sks:
            continue

        for vsk in value_sks:
            first_vals: list[str] = []
            has_diff = False
            for stream in streams:
                instances = [t for t in stream if (*t.structural_key, t.context) == vsk]
                if not instances:
                    break
                first_vals.append(instances[0].value)
                if len(instances) > 1 and any(
                    i.value != instances[0].value for i in instances[1:]
                ):
                    has_diff = True

            if (
                len(first_vals) == n_pages
                and len(set(first_vals)) == 1
                and first_vals[0]
                and has_diff
            ):
                # Promote all structural keys in this element group
                promoted_first.update(group)
                # Also promote the tail token (at parent context, matching tag)
                open_sks = [sk for sk in group if sk[0] == "open"]
                if open_sks:
                    tag = open_sks[0][1]
                    parent_ctx = context.rsplit("/", 1)[0] if "/" in context else ""
                    tail_sk = ("tail", tag, "", parent_ctx)
                    if tail_sk in loop_keys:
                        promoted_first.add(tail_sk)
                break  # one qualifying value key is enough

    # --- Extract skeleton and gaps from each page ---
    # Skeleton = template-constant tokens (appear in same order in every page)
    # Gaps = tokens between consecutive skeleton tokens (may contain loops/optionals)
    # Tokens in promoted_first are skeleton on their FIRST occurrence only.
    skeletons: list[list[Token]] = []
    gap_tokens: list[list[list[Token]]] = []  # [page][gap_idx] = tokens

    for stream in streams:
        skel: list[Token] = []
        gaps: list[list[Token]] = [[]]  # gaps[0] = before first skeleton token
        promoted_count: dict[tuple[str, str, str, str], int] = {}
        for tok in stream:
            sk = (*tok.structural_key, tok.context)
            if sk in template_keys:
                skel.append(tok)
                gaps.append([])
            elif sk in promoted_first:
                n = promoted_count.get(sk, 0)
                promoted_count[sk] = n + 1
                if n == 0:
                    skel.append(tok)
                    gaps.append([])
                else:
                    gaps[-1].append(tok)
            else:
                gaps[-1].append(tok)
        skeletons.append(skel)
        gap_tokens.append(gaps)

    # Verify all pages have the same skeleton length
    skel_len = len(skeletons[0])
    if any(len(s) != skel_len for s in skeletons[1:]):
        return []

    # --- Build template: interleave skeleton tokens and gap analyses ---
    template: list[TemplateElement] = []

    for gap_idx in range(skel_len + 1):
        # Handle gap before skeleton[gap_idx] (or after last skeleton token)
        all_gap = [gt[gap_idx] if gap_idx < len(gt) else [] for gt in gap_tokens]
        if any(len(g) > 0 for g in all_gap):
            template.extend(_analyze_gap(all_gap, ctr, _depth))

        # Handle skeleton token at this position
        if gap_idx < skel_len:
            tok0 = skeletons[0][gap_idx]
            if _is_value_token(tok0.kind):
                values = [skeletons[pi][gap_idx].value for pi in range(n_pages)]
                if len(set(values)) == 1:
                    template.append(Literal(tok0))
                else:
                    template.append(
                        Var(
                            ctr.next(),
                            tok0.kind,
                            tok0.tag,
                            tok0.attr_name,
                            tok0.position_key,
                            all(v != "" for v in values),
                        )
                    )
            else:
                template.append(Literal(tok0))

    return template


def _analyze_gap(
    all_gap: list[list[Token]],
    ctr: _Counter,
    _depth: int = 0,
) -> list[TemplateElement]:
    """Analyze tokens in a gap between skeleton positions.

    Determines if the gap contains loops, optionals, or variables.
    """
    counts = [len(g) for g in all_gap]

    if all(c == 0 for c in counts):
        return []

    has_zero = any(c == 0 for c in counts)
    all_same_count = len(set(counts)) == 1

    if has_zero and max(counts) > 0:
        non_empty = [g for g in all_gap if len(g) > 0]
        non_zero_counts = [c for c in counts if c > 0]
        if len(set(non_zero_counts)) > 1:
            # Variable non-zero count (e.g., 0,2,3) — Set that allows 0 items
            body = _build_set_body(non_empty, ctr, _depth)
            if body:
                return [Set(body, ctr.next())]
        else:
            # Check if any non-empty instance has multiple iterations
            # (e.g., 0,0,0,0,2 items — single page with loop content)
            has_multi_iter = any(len(_split_iterations(g)) > 1 for g in non_empty)
            if has_multi_iter:
                # Multiple iterations detected → Set, not Optional
                body = _build_set_body(non_empty, ctr, _depth)
                if body:
                    return [Set(body, ctr.next())]
            else:
                # 0 or 1 occurrence — Optional
                body = _build_optional_body(non_empty, ctr)
                if body:
                    return [Optional(body)]
        return []

    if not all_same_count:
        # Varying count, all > 0 — per-tag decomposition handles both
        # loops (repeated tags) and independent optionals.
        return _decompose_multi_tag_gap(all_gap, ctr, _depth)

    # All pages have the same non-zero count — check if structure matches
    ref_sks = [tok.structural_key for tok in all_gap[0]]
    if all([tok.structural_key for tok in g] == ref_sks for g in all_gap[1:]):
        return _build_fixed_gap(all_gap, ctr)

    # Same count but different structure. Two cases:
    # (a) Multiple tag types with varying per-tag counts → decompose per tag
    # (b) Same tags but some optional → PosString analysis
    # Detect (a): any page has multiple top-level elements of the same tag
    has_repeated_tags = False
    for g in all_gap:
        tag_counts: Counter[str] = Counter()
        depth_check = 0
        for tok in g:
            if tok.kind == "open" and depth_check == 0:
                tag_counts[tok.tag] += 1
            if tok.kind == "open":
                depth_check += 1
            elif tok.kind == "close":
                depth_check -= 1
        if any(c > 1 for c in tag_counts.values()):
            has_repeated_tags = True
            break

    if has_repeated_tags:
        return _decompose_multi_tag_gap(all_gap, ctr, _depth)

    # No repeated tags — use PosString analysis for optional detection
    return _analyze_gap_posstring(all_gap, ctr, _depth)


def _analyze_gap_posstring(
    all_gap: list[list[Token]],
    ctr: _Counter,
    _depth: int = 0,
) -> list[TemplateElement]:
    """Analyze a gap with mixed structure using PosString abstraction.

    Splits each page's gap into top-level elements (using open/close
    depth), abstracts each element into a symbol (its tag), then finds
    the common backbone via LCS. Elements between backbone positions
    are analyzed as sub-gaps via ``_analyze_gap``.

    - Symbols present in all pages in the same position → fixed (recurse)
    - Symbols with varying count → Set (loop)
    - Symbols present in some pages but not others → Optional
    """
    # Depth guard: prevents _analyze_gap ↔ _analyze_gap_posstring infinite recursion
    if _depth > 8:
        return _build_optional_body(all_gap, ctr)

    # Split each page's gap into top-level element groups
    per_page_elems: list[list[list[Token]]] = []
    for gap in all_gap:
        per_page_elems.append(_split_iterations(gap))

    # Build symbol sequence for each page: the tag of each top-level element
    per_page_syms: list[list[str]] = []
    for elems in per_page_elems:
        syms: list[str] = []
        for group in elems:
            if group and group[0].kind == "open":
                syms.append(group[0].tag)
            elif group:
                # Non-element tokens (text/tail at top level)
                syms.append(f"_text_{group[0].kind}")
        per_page_syms.append(syms)

    # Find the common symbol subsequence (LCS of all symbol sequences)
    # This is the "skeleton" of the gap — fixed elements present everywhere
    if not per_page_syms:
        return []

    from westlean.child_alignment import lcs

    backbone_syms = per_page_syms[0]
    for syms in per_page_syms[1:]:
        backbone_syms = lcs(backbone_syms, syms)

    if not backbone_syms and not per_page_elems:
        return []

    # Map each page's symbols to backbone positions
    from westlean.child_alignment import map_to_backbone

    per_page_bb_idx: list[list[int]] = []
    for syms in per_page_syms:
        if backbone_syms:
            per_page_bb_idx.append(map_to_backbone(syms, backbone_syms))
        else:
            per_page_bb_idx.append([])

    # Build template: interleave backbone elements and gap regions
    template: list[TemplateElement] = []

    n_pages = len(all_gap)
    n_bb = len(backbone_syms)

    for gap_idx in range(n_bb + 1):
        # Collect elements in the gap before backbone[gap_idx]
        gap_elems_per_page: list[list[list[Token]]] = []
        for pi in range(n_pages):
            if gap_idx == 0:
                start = 0
            else:
                start = per_page_bb_idx[pi][gap_idx - 1] + 1
            if gap_idx < n_bb:
                end = per_page_bb_idx[pi][gap_idx]
            else:
                end = len(per_page_elems[pi])
            gap_elems_per_page.append(per_page_elems[pi][start:end])

        # Analyze this sub-gap
        sub_gap_counts = [len(ge) for ge in gap_elems_per_page]
        if any(c > 0 for c in sub_gap_counts):
            # Flatten each page's sub-gap elements back to token lists
            sub_gap_tokens: list[list[Token]] = []
            for ge in gap_elems_per_page:
                flat: list[Token] = []
                for group in ge:
                    flat.extend(group)
                sub_gap_tokens.append(flat)
            template.extend(_analyze_gap(sub_gap_tokens, ctr, _depth + 1))

        # Add backbone element at this position
        if gap_idx < n_bb:
            # Collect this backbone element from all pages
            bb_elems: list[list[Token]] = []
            for pi in range(n_pages):
                idx = per_page_bb_idx[pi][gap_idx]
                bb_elems.append(per_page_elems[pi][idx])
            template.extend(_build_iteration_template(bb_elems, ctr, _depth))

    return template


def _decompose_multi_tag_gap(
    all_gap: list[list[Token]],
    ctr: _Counter,
    _depth: int = 0,
) -> list[TemplateElement]:
    """Decompose a gap with mixed tag types into per-tag Set/fixed elements.

    When a gap has the same total token count across pages but different
    internal structure (e.g., 2 p's + 1 span vs 1 p + 2 span's), split
    into per-tag groups and analyze each independently.
    """
    # Split each page's gap tokens into top-level element groups
    # (using open/close depth tracking)
    per_page_groups: list[list[list[Token]]] = []
    for gap in all_gap:
        per_page_groups.append(_split_iterations(gap))

    # Identify the tag ordering from the page with the most top-level
    # elements (it has the most complete picture of the ordering), then
    # add any remaining tags from other pages.
    sorted_groups = sorted(per_page_groups, key=len, reverse=True)
    tag_order: list[str] = []
    for groups in sorted_groups:
        for group in groups:
            if group and group[0].kind == "open":
                t = group[0].tag
                if t not in tag_order:
                    tag_order.append(t)

    # Collect per-tag groups from each page
    template: list[TemplateElement] = []
    for tag in tag_order:
        tag_groups: list[list[list[Token]]] = []
        for groups in per_page_groups:
            matching = [
                g for g in groups if g and g[0].kind == "open" and g[0].tag == tag
            ]
            tag_groups.append(matching)

        # Flatten: each page contributes a list of iterations for this tag
        counts = [len(tg) for tg in tag_groups]
        all_iterations: list[list[Token]] = []
        for tg in tag_groups:
            all_iterations.extend(tg)

        if not all_iterations:
            continue

        if len(set(counts)) > 1:
            # Varying count → Set
            body = _build_iteration_template(all_iterations, ctr, _depth)
            if body:
                template.append(Set(body, ctr.next()))
        elif any(c == 0 for c in counts):
            # Some pages missing → Optional
            body = _build_optional_body([it for it in all_iterations], ctr)
            if body:
                template.append(Optional(body))
        else:
            # Fixed count — build fixed template from iterations
            body = _build_iteration_template(all_iterations, ctr, _depth)
            template.extend(body)

    return template


def _build_optional_body(
    instances: list[list[Token]],
    ctr: _Counter,
) -> list[TemplateElement]:
    """Build template for an Optional region.

    All value-carrying tokens become Vars since optionals are only seen
    in some pages — we can't confirm any value is truly fixed.
    """
    if not instances:
        return []
    ref = instances[0]
    template: list[TemplateElement] = []
    for pos in range(len(ref)):
        tok = ref[pos]
        if _is_value_token(tok.kind):
            template.append(
                Var(
                    ctr.next(),
                    tok.kind,
                    tok.tag,
                    tok.attr_name,
                    tok.position_key,
                    False,
                )
            )
        else:
            template.append(Literal(tok))
    return template


def _build_set_body(
    all_gap: list[list[Token]],
    ctr: _Counter,
    _depth: int = 0,
) -> list[TemplateElement]:
    """Build the body template for a Set (loop) from gap token lists.

    Splits each gap into iterations, then builds a template from
    comparing all iterations.
    """
    all_iterations: list[list[Token]] = []
    for gap in all_gap:
        iterations = _split_iterations(gap)
        all_iterations.extend(iterations)

    if not all_iterations:
        return []

    return _build_iteration_template(all_iterations, ctr, _depth)


def _build_iteration_template(
    iterations: list[list[Token]],
    ctr: _Counter,
    _depth: int = 0,
) -> list[TemplateElement]:
    """Build a template from multiple token sequences (loop iterations or optional instances).

    Compares structural keys and values across iterations.  Recursively
    handles inner loops via frequency analysis on the iteration tokens.
    """
    if not iterations:
        return []

    # Check if all iterations have the same length
    lengths = [len(it) for it in iterations]
    if len(set(lengths)) > 1:
        # Different lengths — recurse with full frequency analysis
        return _build_template(iterations, ctr, _depth + 1)

    # All iterations have same length — compare token by token
    template: list[TemplateElement] = []
    for pos in range(lengths[0]):
        tok0 = iterations[0][pos]
        if _is_value_token(tok0.kind):
            values = [it[pos].value for it in iterations]
            if len(set(values)) == 1:
                template.append(Literal(tok0))
            else:
                template.append(
                    Var(
                        ctr.next(),
                        tok0.kind,
                        tok0.tag,
                        tok0.attr_name,
                        tok0.position_key,
                        all(v != "" for v in values),
                    )
                )
        else:
            # Structural token — check structural key consistency
            keys = [it[pos].structural_key for it in iterations]
            if len(set(keys)) == 1:
                template.append(Literal(tok0))
            else:
                # Structural mismatch within iterations — fall back
                return _build_template(iterations, ctr, _depth + 1)

    return template


def _build_fixed_gap(
    all_gap: list[list[Token]],
    ctr: _Counter,
) -> list[TemplateElement]:
    """Build template for a gap with consistent token count across pages."""
    ref = all_gap[0]
    if not ref:
        return []

    template: list[TemplateElement] = []
    for pos in range(len(ref)):
        tok = ref[pos]
        if _is_value_token(tok.kind):
            values = [g[pos].value for g in all_gap if pos < len(g)]
            if len(set(values)) == 1:
                template.append(Literal(tok))
            else:
                template.append(
                    Var(
                        ctr.next(),
                        tok.kind,
                        tok.tag,
                        tok.attr_name,
                        tok.position_key,
                        all(v != "" for v in values),
                    )
                )
        else:
            template.append(Literal(tok))

    return template


# ---------------------------------------------------------------------------
# Template matching (forward scan)
# ---------------------------------------------------------------------------


def _match_extract(
    template: list[TemplateElement],
    tokens: list[Token],
    start: int,
    out: dict[str, Any],
) -> int:
    """Match template against tokens starting at position start.

    Returns the number of tokens consumed, or -1 on mismatch.
    """
    ti = start
    for elem in template:
        if isinstance(elem, Literal):
            if ti >= len(tokens):
                return -1
            page_tok = tokens[ti]
            if page_tok.structural_key != elem.token.structural_key:
                return -1
            if _is_value_token(elem.token.kind):
                if page_tok.value != elem.token.value:
                    return -1
            ti += 1
        elif isinstance(elem, Var):
            if ti >= len(tokens):
                return -1
            page_tok = tokens[ti]
            if page_tok.structural_key != (elem.kind, elem.tag, elem.attr_name):
                return -1
            if (
                elem.always_has_value
                and _is_value_token(elem.kind)
                and not page_tok.value
            ):
                return -1
            if _is_value_token(elem.kind) and page_tok.value:
                out[elem.name] = page_tok.value
            ti += 1
        elif isinstance(elem, Set):
            items: list[dict[str, Any]] = []
            while ti < len(tokens):
                item: dict[str, Any] = {}
                consumed = _match_extract(elem.body, tokens, ti, item)
                if consumed <= 0:
                    break
                items.append(item)
                ti += consumed
            if items:
                out[elem.var_name] = items
        elif isinstance(elem, Optional):
            tentative: dict[str, Any] = {}
            consumed = _match_extract(elem.elements, tokens, ti, tentative)
            if consumed > 0:
                out.update(tentative)
                ti += consumed

    return ti - start


def _match_mask(
    template: list[TemplateElement],
    tokens: list[Token],
    start: int,
    mask: dict[str, bool],
) -> int:
    """Match template against tokens, annotating each position as fixed/variable.

    Returns tokens consumed, or -1 on mismatch.
    """
    ti = start
    for elem in template:
        if isinstance(elem, Literal):
            if ti >= len(tokens):
                return -1
            page_tok = tokens[ti]
            if page_tok.structural_key != elem.token.structural_key:
                return -1
            if _is_value_token(elem.token.kind):
                if page_tok.value != elem.token.value:
                    return -1
                if page_tok.value:
                    mask[page_tok.position_key] = True
            ti += 1
        elif isinstance(elem, Var):
            if ti >= len(tokens):
                return -1
            page_tok = tokens[ti]
            if page_tok.structural_key != (elem.kind, elem.tag, elem.attr_name):
                return -1
            if _is_value_token(elem.kind) and page_tok.value:
                mask[page_tok.position_key] = False
            ti += 1
        elif isinstance(elem, Set):
            while ti < len(tokens):
                tentative: dict[str, bool] = {}
                consumed = _match_mask(elem.body, tokens, ti, tentative)
                if consumed <= 0:
                    break
                # All Set content is variable
                for key in tentative:
                    mask[key] = False
                ti += consumed
        elif isinstance(elem, Optional):
            tentative_mask: dict[str, bool] = {}
            consumed = _match_mask(elem.elements, tokens, ti, tentative_mask)
            if consumed > 0:
                # Optional content is variable
                for key in tentative_mask:
                    mask[key] = False
                ti += consumed

    return ti - start


# ---------------------------------------------------------------------------
# RELAX NG generation from linearized template
# ---------------------------------------------------------------------------

_RNG_NS = "http://relaxng.org/ns/structure/1.0"


def _elements_to_relax_ng(elements: list[TemplateElement]) -> str:
    """Convert a linearized ExAlg template to a RELAX NG schema string."""
    from xml.etree.ElementTree import Element as XE, SubElement, tostring

    grammar = XE("grammar", xmlns=_RNG_NS)
    grammar.set("datatypeLibrary", "http://www.w3.org/2001/XMLSchema-datatypes")
    start = SubElement(grammar, "start")

    _rng_from_flat(elements, 0, start)

    return tostring(grammar, encoding="unicode", xml_declaration=True)


def _rng_from_flat(
    elements: list[TemplateElement],
    pos: int,
    parent: Any,
) -> int:
    """Walk the flat element list starting at *pos*, appending RELAX NG
    nodes to *parent*.  Returns the index past the last consumed element.

    When an ``open`` Literal is encountered, a ``<element>`` is created
    and the function recurses to consume everything up to the matching
    ``close``.
    """
    from xml.etree.ElementTree import SubElement

    while pos < len(elements):
        elem = elements[pos]

        if isinstance(elem, Literal):
            tok = elem.token
            if tok.kind == "open":
                from westlean.compat import COMMENT_TAG

                if tok.tag == COMMENT_TAG:
                    # RELAX NG ignores comments — skip to matching close
                    pos += 1
                    depth = 1
                    while pos < len(elements) and depth > 0:
                        e = elements[pos]
                        if isinstance(e, Literal) and e.token.tag == COMMENT_TAG:
                            if e.token.kind == "open":
                                depth += 1
                            elif e.token.kind == "close":
                                depth -= 1
                        pos += 1
                    continue
                # Start a new <element name="tag">
                el = SubElement(parent, "element", name=tok.tag)
                pos += 1
                # Recurse to fill the element's content until close
                pos = _rng_from_flat(elements, pos, el)
                # RELAX NG requires every <element> to have content;
                # add <empty/> if nothing was generated inside.
                if len(el) == 0:
                    SubElement(el, "empty")
                continue
            elif tok.kind == "close":
                # End of the current element — return to caller
                return pos + 1
            elif tok.kind == "text" or tok.kind == "tail":
                # Fixed text — allow it via <text/> (we can't enforce
                # exact text values in RELAX NG compact without <value>,
                # but <text/> is more permissive and matches the
                # algorithm's extraction behavior)
                if tok.value:
                    SubElement(parent, "text")
                # Empty text/tail: nothing to add to schema
                pos += 1
                continue
            elif tok.kind == "attr":
                attr_el = SubElement(parent, "attribute", name=tok.attr_name)
                SubElement(attr_el, "text")
                pos += 1
                continue

        elif isinstance(elem, Var):
            if elem.kind == "text" or elem.kind == "tail":
                SubElement(parent, "text")
            elif elem.kind == "attr":
                attr_el = SubElement(parent, "attribute", name=elem.attr_name)
                SubElement(attr_el, "text")
            pos += 1
            continue

        elif isinstance(elem, Set):
            zm = SubElement(parent, "zeroOrMore")
            _rng_from_flat(elem.body, 0, zm)
            pos += 1
            continue

        elif isinstance(elem, Optional):
            opt = SubElement(parent, "optional")
            _rng_from_flat(elem.elements, 0, opt)
            pos += 1
            continue

        pos += 1

    return pos


# ---------------------------------------------------------------------------
# Inferred template
# ---------------------------------------------------------------------------


class ExalgTemplate:
    """Template inferred by EXALG token-frequency equivalence class analysis."""

    def __init__(self, elements: list[TemplateElement]) -> None:
        self._elements = elements

    def extract(self, page: etree._Element) -> dict[str, Any] | None:
        tokens = _linearize(page)
        result: dict[str, Any] = {}
        consumed = _match_extract(self._elements, tokens, 0, result)
        if consumed < 0 or consumed != len(tokens):
            return None
        return result

    def fixed_mask(self, page: etree._Element) -> dict[str, bool] | None:
        tokens = _linearize(page)
        mask: dict[str, bool] = {}
        consumed = _match_mask(self._elements, tokens, 0, mask)
        if consumed < 0 or consumed != len(tokens):
            return None
        return mask

    def serialize(self) -> dict[str, Any]:
        from westlean.serialization import ExalgTemplateModel, exalg_elements_to_model

        model = ExalgTemplateModel(elements=exalg_elements_to_model(self._elements))
        return model.model_dump()

    def get_relax_ng(self) -> str:
        return _elements_to_relax_ng(self._elements)

    @classmethod
    def restore(cls, data: dict[str, Any]) -> ExalgTemplate:
        from westlean.serialization import ExalgTemplateModel

        model = ExalgTemplateModel.model_validate(data)
        return model.to_internal()


# ---------------------------------------------------------------------------
# Inferer
# ---------------------------------------------------------------------------


class ExalgInferer:
    """EXALG template inferer (Arasu & Garcia-Molina, SIGMOD 2003).

    Uses token-frequency equivalence classes to discover template
    structure from linearized HTML pages.  Tokens with occurrence
    vector (1,1,...,1) are template constants; varying-count tokens
    indicate loops; partially-present tokens indicate optionals.
    """

    def infer(self, pages: Sequence[etree._Element]) -> ExalgTemplate | EmptyTemplate:
        if len(pages) < 2:
            return EmptyTemplate()

        # Tokenize all pages
        streams = [_linearize(page) for page in pages]

        # Check structural compatibility: root tags must agree
        root_tags = {s[0].tag for s in streams if s}
        if len(root_tags) != 1:
            return EmptyTemplate()

        # Build template via ECGM + ConstTemp
        ctr = _Counter()
        elements = _build_template(streams, ctr)

        if not elements:
            return EmptyTemplate()

        return ExalgTemplate(elements)
