"""Local unranked tree automaton (Kosala et al., IJCAI 2003).

Learns a recursive template tree from positive examples by child sequence
alignment at every level.  A new page is accepted iff it matches the
learned template structure, using regex-like child sequence matching
(backbone + repeating/optional regions) instead of fixed-arity pattern
tuples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from lxml import etree

from westlean.child_alignment import align_children
from westlean.compat import element_tag
from westlean.protocol import EmptyTemplate


# ---------------------------------------------------------------------------
# Internal representation
# ---------------------------------------------------------------------------


class _Counter:
    """Sequential variable name generator: var_0, var_1, ..."""

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> str:
        name = f"var_{self._n}"
        self._n += 1
        return name


# A slot is (is_fixed, value).
# Fixed:    (True, "literal text")
# Variable: (False, "var_0")
_Slot = tuple[bool, str]


@dataclass
class _UTANode:
    """Node in the UTA template tree.

    Structurally identical to FiVaTech's ``_PatternNode`` and
    Anti-Unification's ``_TplNode``, so the shared ``TplNodeModel``
    serialization works for all three.
    """

    tag: str
    attr_names: tuple[str, ...]  # sorted attribute names (structural)
    text: _Slot  # element's .text (before first child)
    tail: _Slot  # element's .tail (after it, in parent context)
    attrs: dict[str, _Slot]  # attr_name -> slot
    children: list[_UTANode] | None  # backbone; None => fully variable
    children_var: str | None  # variable name when children is None
    text_always_present: bool = False
    tail_always_present: bool = False
    # Multi-region support: each region is (after_backbone_pos, template, var_name)
    repeating_regions: list[tuple[int, _UTANode, str]] = field(default_factory=list)
    # Each optional region is (after_backbone_pos, children_templates)
    optional_regions: list[tuple[int, list[_UTANode]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Template tree building (core of the UTA inference)
# ---------------------------------------------------------------------------


def _build_uta_tree(
    pages: list[etree._Element],
    ctr: _Counter,
) -> _UTANode | None:
    """Build a UTA template node from N example elements.

    Aligns children across pages using LCS, then recursively builds
    sub-templates for backbone, repeating, and optional regions.
    """
    # Tags must agree
    tags = {element_tag(p) for p in pages}
    if len(tags) != 1:
        return None

    tag = tags.pop()

    # Attribute names must agree (structural skeleton)
    attr_name_sets = {tuple(sorted(p.attrib)) for p in pages}
    if len(attr_name_sets) != 1:
        return None

    attr_names = attr_name_sets.pop()

    # --- text slot: fixed iff all pages identical ---
    texts = [p.text or "" for p in pages]
    text_always_present = all(p.text for p in pages)
    text: _Slot = (True, texts[0]) if len(set(texts)) == 1 else (False, ctr.next())

    # --- tail slot ---
    tails = [p.tail or "" for p in pages]
    tail_always_present = all(p.tail for p in pages)
    tail: _Slot = (True, tails[0]) if len(set(tails)) == 1 else (False, ctr.next())

    # --- attribute slots ---
    attrs: dict[str, _Slot] = {}
    for name in attr_names:
        vals = {p.attrib[name] for p in pages}
        if len(vals) == 1:
            attrs[name] = (True, next(iter(vals)))
        else:
            attrs[name] = (False, ctr.next())

    # --- children: align via LCS then recurse ---
    children_lists = [list(p) for p in pages]

    # All pages have zero children → leaf node
    if all(len(cl) == 0 for cl in children_lists):
        return _UTANode(
            tag,
            attr_names,
            text,
            tail,
            attrs,
            [],
            None,
            text_always_present,
            tail_always_present,
        )

    alignment = align_children(children_lists)

    if not alignment.backbone and not alignment.repeating and not alignment.optional:
        # No common structure at all — entire children region is variant
        return _UTANode(
            tag,
            attr_names,
            text,
            tail,
            attrs,
            None,
            ctr.next(),
            text_always_present,
            tail_always_present,
        )

    # Merge backbone children
    merged_children: list[_UTANode] = []
    for bi in range(len(alignment.backbone)):
        aligned_elems = [
            children_lists[pi][alignment.backbone_indices[pi][bi]]
            for pi in range(len(pages))
        ]
        child_node = _build_uta_tree(aligned_elems, ctr)
        if child_node is None:
            return _UTANode(
                tag,
                attr_names,
                text,
                tail,
                attrs,
                None,
                ctr.next(),
                text_always_present,
                tail_always_present,
            )
        merged_children.append(child_node)

    # Detect ALL repeating regions (loops)
    repeating_regions: list[tuple[int, _UTANode, str]] = []
    for region in alignment.repeating:
        all_repeat_elems: list[etree._Element] = []
        for pi, cl in enumerate(children_lists):
            if region.after_backbone_pos < 0:
                start = 0
            else:
                start = alignment.backbone_indices[pi][region.after_backbone_pos] + 1
            if region.after_backbone_pos + 1 < len(alignment.backbone):
                end = alignment.backbone_indices[pi][region.after_backbone_pos + 1]
            else:
                end = len(cl)
            for idx in range(start, end):
                if element_tag(cl[idx]) == region.tag:
                    all_repeat_elems.append(cl[idx])

        if all_repeat_elems:
            unit = _build_uta_tree(all_repeat_elems, ctr)
            if unit is not None:
                _relax_always_present(unit)
                repeating_regions.append((region.after_backbone_pos, unit, ctr.next()))

    # Detect ALL optional regions (conditionals)
    optional_regions: list[tuple[int, list[_UTANode]]] = []
    for opt_region in alignment.optional:
        present_elems: list[list[etree._Element]] = []
        for pi, cl in enumerate(children_lists):
            if not opt_region.present_in[pi]:
                continue
            if opt_region.after_backbone_pos < 0:
                start = 0
            else:
                start = (
                    alignment.backbone_indices[pi][opt_region.after_backbone_pos] + 1
                )
            if opt_region.after_backbone_pos + 1 < len(alignment.backbone):
                end = alignment.backbone_indices[pi][opt_region.after_backbone_pos + 1]
            else:
                end = len(cl)
            present_elems.append(cl[start:end])

        if present_elems:
            opt_nodes: list[_UTANode] = []
            max_len = max(len(pe) for pe in present_elems)
            for child_idx in range(max_len):
                elems = [pe[child_idx] for pe in present_elems if child_idx < len(pe)]
                if elems:
                    node = _build_uta_tree(elems, ctr)
                    if node is not None:
                        opt_nodes.append(node)
            if opt_nodes:
                optional_regions.append((opt_region.after_backbone_pos, opt_nodes))

    return _UTANode(
        tag,
        attr_names,
        text,
        tail,
        attrs,
        merged_children,
        None,
        text_always_present,
        tail_always_present,
        repeating_regions,
        optional_regions,
    )


def _relax_always_present(node: _UTANode) -> None:
    """Clear always_present flags throughout a repeating region template.

    Within loops, any position that's variable should allow empty values —
    the flag is over-constraining for loop body templates.
    """
    node.text_always_present = False
    node.tail_always_present = False
    if node.children:
        for child in node.children:
            _relax_always_present(child)


# ---------------------------------------------------------------------------
# Structural matching (UTA acceptance)
# ---------------------------------------------------------------------------


def _matches_structure(
    tpl: _UTANode,
    elem: etree._Element,
    *,
    check_values: bool = True,
) -> bool:
    """Check if an element matches the template structure (recursive).

    Validates tag, attribute names, and child sequence against the
    backbone + repeating/optional regions.

    When *check_values* is True (default), also verifies fixed text and
    attribute values.  Set to False for optional children whose templates
    are built from few examples and may be over-specific about values.
    """
    if tpl.tag != element_tag(elem):
        return False
    if tpl.attr_names != tuple(sorted(elem.attrib)):
        return False
    if check_values:
        for name in tpl.attr_names:
            is_fixed, val = tpl.attrs[name]
            if is_fixed and elem.attrib[name] != val:
                return False
        is_fixed, val = tpl.text
        if is_fixed and val != (elem.text or ""):
            return False

    # Validate children against backbone + regions
    if (tpl.repeating_regions or tpl.optional_regions) and tpl.children is not None:
        elem_children = list(elem)
        backbone = tpl.children
        # Build lookups per gap position
        rep_by_gap: dict[int, list[_UTANode]] = {}
        for pos, child_tpl, _var in tpl.repeating_regions:
            rep_by_gap.setdefault(pos + 1, []).append(child_tpl)
        opt_by_gap: dict[int, list[_UTANode]] = {}
        for pos, children in tpl.optional_regions:
            for c in children:
                opt_by_gap.setdefault(pos + 1, []).append(c)

        pi = 0
        ok = True
        for gap_idx in range(len(backbone) + 1):
            rep_tpls = {t.tag: t for t in rep_by_gap.get(gap_idx, [])}
            opt_tpls = {t.tag: t for t in opt_by_gap.get(gap_idx, [])}
            known = set(rep_tpls) | set(opt_tpls)
            while pi < len(elem_children) and element_tag(elem_children[pi]) in known:
                child = elem_children[pi]
                child_tag = element_tag(child)
                if child_tag in rep_tpls:
                    if not _matches_structure(
                        rep_tpls[child_tag], child, check_values=check_values
                    ):
                        ok = False
                        break
                elif child_tag in opt_tpls:
                    if not _matches_structure(
                        opt_tpls[child_tag], child, check_values=check_values
                    ):
                        ok = False
                        break
                pi += 1
            if not ok:
                break
            if gap_idx < len(backbone):
                if (
                    pi >= len(elem_children)
                    or element_tag(elem_children[pi]) != backbone[gap_idx].tag
                ):
                    ok = False
                    break
                if not _matches_structure(
                    backbone[gap_idx], elem_children[pi], check_values=check_values
                ):
                    ok = False
                    break
                pi += 1
        if not ok or pi != len(elem_children):
            return False
    elif tpl.children is not None:
        elem_children = list(elem)
        if len(elem_children) != len(tpl.children):
            return False
        for ctpl, celem in zip(tpl.children, elem_children):
            if not _matches_structure(ctpl, celem, check_values=check_values):
                return False
    elif list(elem) and not tpl.children_var:
        return False

    return True


# ---------------------------------------------------------------------------
# Inferred template (public API)
# ---------------------------------------------------------------------------


class KTestableTemplate:
    """Template inferred by local unranked tree automaton."""

    def __init__(self, root: _UTANode, k: int = 2) -> None:
        self._root = root
        self._k = k

    # -- extract ----------------------------------------------------------

    def extract(self, page: etree._Element) -> dict[str, Any] | None:
        result: dict[str, Any] = {}
        if not self._extract_node(self._root, page, result):
            return None
        return result

    def _extract_node(
        self, tpl: _UTANode, elem: etree._Element, out: dict[str, Any]
    ) -> bool:
        if tpl.tag != element_tag(elem):
            return False
        if tpl.attr_names != tuple(sorted(elem.attrib)):
            return False

        # Text
        is_fixed, val = tpl.text
        page_text = elem.text or ""
        if is_fixed:
            if val != page_text:
                return False
        else:
            if tpl.text_always_present and not page_text:
                return False
            if page_text:
                out[val] = page_text

        # Attributes
        for name in tpl.attr_names:
            is_fixed, pat_val = tpl.attrs[name]
            page_val = elem.attrib[name]
            if is_fixed:
                if pat_val != page_val:
                    return False
            else:
                out[pat_val] = page_val

        # Children
        if tpl.children is None:
            text = _collect_children_text(elem)
            if text and tpl.children_var:
                out[tpl.children_var] = text
            return True

        page_children = list(elem)

        # Region-aware extraction
        if tpl.repeating_regions or tpl.optional_regions:
            return self._extract_children(tpl, page_children, out)

        # Standard fixed-structure extraction (no regions)
        if len(tpl.children) != len(page_children):
            return False

        for tc, pc in zip(tpl.children, page_children):
            if not self._extract_node(tc, pc, out):
                return False
            is_fixed, val = tc.tail
            page_tail = pc.tail or ""
            if is_fixed:
                if val != page_tail:
                    return False
            else:
                if tc.tail_always_present and not page_tail:
                    return False
                if page_tail:
                    out[val] = page_tail

        return True

    def _extract_children(
        self,
        tpl: _UTANode,
        page_children: list[etree._Element],
        out: dict[str, Any],
    ) -> bool:
        """Extract data from children using gap-indexed region dispatch."""
        assert tpl.children is not None
        backbone = tpl.children

        # Build gap→regions lookups (gap_idx = after_backbone_pos + 1)
        rep_by_gap: dict[int, list[tuple[_UTANode, str]]] = {}
        for pos, child_tpl, var in tpl.repeating_regions:
            rep_by_gap.setdefault(pos + 1, []).append((child_tpl, var))
        opt_by_gap: dict[int, list[list[_UTANode]]] = {}
        for pos, children in tpl.optional_regions:
            opt_by_gap.setdefault(pos + 1, []).append(children)

        pi = 0
        for gap_idx in range(len(backbone) + 1):
            # Consume repeating regions in this gap
            for rtpl, rvar in rep_by_gap.get(gap_idx, []):
                items: list[dict[str, Any]] = []
                while pi < len(page_children) and _matches_structure(
                    rtpl, page_children[pi], check_values=False
                ):
                    item: dict[str, Any] = {}
                    if self._extract_node(rtpl, page_children[pi], item):
                        rc_tail_fixed, rc_tail_val = rtpl.tail
                        rc_page_tail = page_children[pi].tail or ""
                        if not rc_tail_fixed and rc_page_tail:
                            item[rc_tail_val] = rc_page_tail
                        items.append(item)
                    pi += 1
                if items:
                    out[rvar] = items

            # Consume optional regions in this gap — use skeleton check
            # (check_values=False) because optional templates are often
            # built from few examples and may be over-specific about values.
            for opt_children in opt_by_gap.get(gap_idx, []):
                for oc in opt_children:
                    if (
                        pi < len(page_children)
                        and element_tag(page_children[pi]) == oc.tag
                        and _matches_structure(
                            oc, page_children[pi], check_values=False
                        )
                    ):
                        self._extract_node(oc, page_children[pi], out)
                        pi += 1

            # Match backbone child (if not past end)
            if gap_idx < len(backbone):
                bp = backbone[gap_idx]
                if pi >= len(page_children):
                    return False
                if not self._extract_node(bp, page_children[pi], out):
                    return False
                is_fixed, val = bp.tail
                page_tail = page_children[pi].tail or ""
                if is_fixed:
                    if val != page_tail:
                        return False
                else:
                    if bp.tail_always_present and not page_tail:
                        return False
                    if page_tail:
                        out[val] = page_tail
                pi += 1

        return pi == len(page_children)

    # -- fixed_mask -------------------------------------------------------

    def fixed_mask(self, page: etree._Element) -> dict[str, bool] | None:
        mask: dict[str, bool] = {}
        if not self._mask_node(self._root, page, "", mask):
            return None
        return mask

    def _mask_node(
        self,
        tpl: _UTANode,
        elem: etree._Element,
        prefix: str,
        mask: dict[str, bool],
    ) -> bool:
        if tpl.tag != element_tag(elem):
            return False
        if tpl.attr_names != tuple(sorted(elem.attrib)):
            return False

        # Text
        text_key = f"{prefix}/text" if prefix else "text"
        is_fixed, val = tpl.text
        page_text = elem.text or ""
        if is_fixed:
            if val != page_text:
                return False
            if page_text:
                mask[text_key] = True
        else:
            if tpl.text_always_present and not page_text:
                return False
            if page_text:
                mask[text_key] = False

        # Attributes
        for attr_name in sorted(elem.attrib):
            attr_key = f"{prefix}/@{attr_name}" if prefix else f"@{attr_name}"
            is_fixed, _ = tpl.attrs[attr_name]
            mask[attr_key] = is_fixed

        # Children
        if tpl.children is None:
            _mark_children_variable(elem, prefix, mask)
            return True

        page_children = list(elem)

        # Region-aware masking
        if tpl.repeating_regions or tpl.optional_regions:
            return self._mask_children(tpl, page_children, prefix, mask)

        # Standard fixed-structure masking
        if len(tpl.children) != len(page_children):
            return False

        for i, (tc, pc) in enumerate(zip(tpl.children, page_children)):
            child_prefix = f"{prefix}/{i}" if prefix else str(i)
            if not self._mask_node(tc, pc, child_prefix, mask):
                return False
            tail_key = f"{child_prefix}/tail"
            is_fixed, val = tc.tail
            page_tail = pc.tail or ""
            if is_fixed:
                if val != page_tail:
                    return False
                if page_tail:
                    mask[tail_key] = True
            else:
                if tc.tail_always_present and not page_tail:
                    return False
                if page_tail:
                    mask[tail_key] = False

        return True

    def _mask_children(
        self,
        tpl: _UTANode,
        page_children: list[etree._Element],
        prefix: str,
        mask: dict[str, bool],
    ) -> bool:
        """Mask children with gap-indexed region dispatch."""
        assert tpl.children is not None
        backbone = tpl.children

        rep_by_gap: dict[int, list[tuple[_UTANode, str]]] = {}
        for pos, child_tpl, var in tpl.repeating_regions:
            rep_by_gap.setdefault(pos + 1, []).append((child_tpl, var))
        opt_by_gap: dict[int, list[list[_UTANode]]] = {}
        for pos, children in tpl.optional_regions:
            opt_by_gap.setdefault(pos + 1, []).append(children)

        pi = 0
        for gap_idx in range(len(backbone) + 1):
            # Repeating → all variable
            for rtpl, _ in rep_by_gap.get(gap_idx, []):
                while pi < len(page_children) and _matches_structure(
                    rtpl, page_children[pi], check_values=False
                ):
                    child_prefix = f"{prefix}/{pi}" if prefix else str(pi)
                    _mark_subtree_variable(page_children[pi], child_prefix, mask)
                    if page_children[pi].tail:
                        mask[f"{child_prefix}/tail"] = False
                    pi += 1

            # Optional → all variable (skeleton check only)
            for opt_children in opt_by_gap.get(gap_idx, []):
                for oc in opt_children:
                    if (
                        pi < len(page_children)
                        and element_tag(page_children[pi]) == oc.tag
                        and _matches_structure(
                            oc, page_children[pi], check_values=False
                        )
                    ):
                        child_prefix = f"{prefix}/{pi}" if prefix else str(pi)
                        _mark_subtree_variable(page_children[pi], child_prefix, mask)
                        if page_children[pi].tail:
                            mask[f"{child_prefix}/tail"] = False
                        pi += 1

            # Backbone child
            if gap_idx < len(backbone):
                bp = backbone[gap_idx]
                if pi >= len(page_children):
                    return False
                child_prefix = f"{prefix}/{pi}" if prefix else str(pi)
                if not self._mask_node(bp, page_children[pi], child_prefix, mask):
                    return False
                tail_key = f"{child_prefix}/tail"
                is_fixed, val = bp.tail
                page_tail = page_children[pi].tail or ""
                if is_fixed:
                    if val != page_tail:
                        return False
                    if page_tail:
                        mask[tail_key] = True
                else:
                    if bp.tail_always_present and not page_tail:
                        return False
                    if page_tail:
                        mask[tail_key] = False
                pi += 1

        return pi == len(page_children)

    # -- serialization ----------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        from westlean.serialization import KTestableTemplateModel, tpl_node_to_model

        model = KTestableTemplateModel(k=self._k, root=tpl_node_to_model(self._root))
        return model.model_dump()

    def get_relax_ng(self) -> str:
        from westlean.algorithms.anti_unification import _tpl_node_to_relax_ng

        return _tpl_node_to_relax_ng(self._root)  # type: ignore[arg-type]

    @classmethod
    def restore(cls, data: dict[str, Any]) -> KTestableTemplate:
        from westlean.serialization import KTestableTemplateModel

        model = KTestableTemplateModel.model_validate(data)
        return model.to_internal()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_children_text(elem: etree._Element) -> str:
    """Collect all text content from an element's children (not elem.text)."""
    parts: list[str] = []
    for child in elem:
        _collect_elem_text(child, parts)
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _collect_elem_text(elem: etree._Element, parts: list[str]) -> None:
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        _collect_elem_text(child, parts)
        if child.tail:
            parts.append(child.tail)


def _mark_children_variable(
    elem: etree._Element, prefix: str, mask: dict[str, bool]
) -> None:
    """Mark all positions under *elem*'s children as variable."""
    for i, child in enumerate(elem):
        child_prefix = f"{prefix}/{i}" if prefix else str(i)
        _mark_subtree_variable(child, child_prefix, mask)
        if child.tail:
            mask[f"{child_prefix}/tail"] = False


def _mark_subtree_variable(
    elem: etree._Element, prefix: str, mask: dict[str, bool]
) -> None:
    """Mark every text/attribute position in a subtree as variable (False)."""
    if elem.text:
        mask[f"{prefix}/text"] = False
    for attr_name in sorted(elem.attrib):
        mask[f"{prefix}/@{attr_name}"] = False
    for i, child in enumerate(elem):
        child_prefix = f"{prefix}/{i}"
        _mark_subtree_variable(child, child_prefix, mask)
        if child.tail:
            mask[f"{child_prefix}/tail"] = False


# ---------------------------------------------------------------------------
# Inferer
# ---------------------------------------------------------------------------


class KTestableInferer:
    """Local unranked tree automaton inferer (Kosala et al., IJCAI 2003).

    Builds a recursive template tree from training pages using child
    sequence alignment at every level.  A candidate page is accepted iff
    it matches the learned template structure with regex-like child
    sequence matching (backbone + repeating/optional regions).

    The ``k`` parameter is preserved for API compatibility.  Internally,
    the recursive template tree provides equivalent (and often better)
    discrimination regardless of ``k``.
    """

    def __init__(self, k: int = 2) -> None:
        if k not in (2, 3):
            raise ValueError(f"k must be 2 or 3, got {k}")
        self._k = k

    def infer(
        self, pages: Sequence[etree._Element]
    ) -> KTestableTemplate | EmptyTemplate:
        if len(pages) < 2:
            return EmptyTemplate()

        ctr = _Counter()
        root = _build_uta_tree(list(pages), ctr)
        if root is None:
            return EmptyTemplate()
        return KTestableTemplate(root, self._k)
