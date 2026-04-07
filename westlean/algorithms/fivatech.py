"""FiVaTech-inspired fixed/variant pattern tree template inference.

Based on Kayed & Chang (IEEE TKDE 2010). Merges ALL input trees
simultaneously into a single pattern tree where each node is labeled
"fixed" or "variant," using LCS-based child sequence alignment rather
than strict positional matching.
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
class _PatternNode:
    """Node in the FiVaTech pattern tree."""

    tag: str
    attr_names: tuple[str, ...]  # sorted attribute names (structural)
    text: _Slot  # element's .text (before first child)
    tail: _Slot  # element's .tail (after it, in parent context)
    attrs: dict[str, _Slot]  # attr_name -> slot
    children: list[_PatternNode] | None  # None => children region is variable
    children_var: str | None  # variable name when children is None
    text_always_present: bool = False  # was .text non-empty in ALL training pages?
    tail_always_present: bool = False  # was .tail non-empty in ALL training pages?
    # Multi-region support: each region is (after_backbone_pos, template, var_name)
    repeating_regions: list[tuple[int, _PatternNode, str]] = field(default_factory=list)
    # Each optional region is (after_backbone_pos, children_templates)
    optional_regions: list[tuple[int, list[_PatternNode]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LCS utilities for child sequence alignment
# ---------------------------------------------------------------------------


def _lcs(a: list[str], b: list[str]) -> list[str]:
    """Longest common subsequence of two tag sequences."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to recover the subsequence
    result: list[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            result.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    result.reverse()
    return result


def _map_to_backbone(seq: list[str], backbone: list[str]) -> list[int]:
    """Map backbone positions to indices in *seq* (greedy left-to-right).

    Since *backbone* is the LCS of all sequences, it is guaranteed to be
    a subsequence of *seq*, so the mapping always succeeds.
    """
    result: list[int] = []
    j = 0
    for tag in backbone:
        while seq[j] != tag:
            j += 1
        result.append(j)
        j += 1
    return result


# ---------------------------------------------------------------------------
# Simultaneous tree merge (core of FiVaTech)
# ---------------------------------------------------------------------------


def _merge_trees(
    pages: list[etree._Element],
    ctr: _Counter,
) -> _PatternNode | None:
    """Merge N page elements simultaneously into a pattern node.

    Unlike pairwise anti-unification, this examines all pages at once,
    so the result is independent of input order.
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
        return _PatternNode(
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
        return _PatternNode(
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
    merged_children: list[_PatternNode] = []
    for bi in range(len(alignment.backbone)):
        aligned_elems = [
            children_lists[pi][alignment.backbone_indices[pi][bi]]
            for pi in range(len(pages))
        ]
        child_node = _merge_trees(aligned_elems, ctr)
        if child_node is None:
            return _PatternNode(
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
    repeating_regions: list[tuple[int, _PatternNode, str]] = []
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
            unit = _merge_trees(all_repeat_elems, ctr)
            if unit is not None:
                _relax_always_present(unit)
                repeating_regions.append((region.after_backbone_pos, unit, ctr.next()))

    # Detect ALL optional regions (conditionals)
    optional_regions: list[tuple[int, list[_PatternNode]]] = []
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
            opt_nodes: list[_PatternNode] = []
            max_len = max(len(pe) for pe in present_elems)
            for child_idx in range(max_len):
                elems = [pe[child_idx] for pe in present_elems if child_idx < len(pe)]
                if elems:
                    node = _merge_trees(elems, ctr)
                    if node is not None:
                        opt_nodes.append(node)
            if opt_nodes:
                optional_regions.append((opt_region.after_backbone_pos, opt_nodes))

    return _PatternNode(
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


# ---------------------------------------------------------------------------
# Inferred template (public API)
# ---------------------------------------------------------------------------


class FiVaTechTemplate:
    """Template inferred by FiVaTech-style simultaneous tree merging."""

    def __init__(self, root: _PatternNode) -> None:
        self._root = root

    def serialize(self) -> dict[str, Any]:
        from westlean.serialization import FiVaTechTemplateModel, tpl_node_to_model

        model = FiVaTechTemplateModel(root=tpl_node_to_model(self._root))
        return model.model_dump()

    def get_relax_ng(self) -> str:
        from westlean.algorithms.anti_unification import _tpl_node_to_relax_ng

        return _tpl_node_to_relax_ng(self._root)  # type: ignore[arg-type]

    @classmethod
    def restore(cls, data: dict[str, Any]) -> FiVaTechTemplate:
        from westlean.serialization import FiVaTechTemplateModel

        model = FiVaTechTemplateModel.model_validate(data)
        return model.to_internal()

    # -- extract ----------------------------------------------------------

    def extract(self, page: etree._Element) -> dict[str, Any] | None:
        result: dict[str, Any] = {}
        if not self._extract_node(self._root, page, result):
            return None
        return result

    def _extract_node(
        self, pat: _PatternNode, elem: etree._Element, out: dict[str, Any]
    ) -> bool:
        if pat.tag != element_tag(elem):
            return False
        if pat.attr_names != tuple(sorted(elem.attrib)):
            return False

        # Text
        is_fixed, val = pat.text
        page_text = elem.text or ""
        if is_fixed:
            if val != page_text:
                return False
        else:
            if pat.text_always_present and not page_text:
                return False
            if page_text:
                out[val] = page_text

        # Attributes
        for name in pat.attr_names:
            is_fixed, pat_val = pat.attrs[name]
            page_val = elem.attrib[name]
            if is_fixed:
                if pat_val != page_val:
                    return False
            else:
                out[pat_val] = page_val

        # Children
        if pat.children is None:
            text = _collect_children_text(elem)
            if text and pat.children_var:
                out[pat.children_var] = text
            return True

        page_children = list(elem)

        # Region-aware extraction
        if pat.repeating_regions or pat.optional_regions:
            return self._extract_children(pat, page_children, out)

        # Standard fixed-structure extraction (no regions)
        if len(pat.children) != len(page_children):
            return False

        for tc, pc in zip(pat.children, page_children):
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
        pat: _PatternNode,
        page_children: list[etree._Element],
        out: dict[str, Any],
    ) -> bool:
        """Extract data from children using gap-indexed region dispatch."""
        assert pat.children is not None
        backbone = pat.children

        # Build gap→regions lookups (gap_idx = after_backbone_pos + 1)
        rep_by_gap: dict[int, list[tuple[_PatternNode, str]]] = {}
        for pos, tpl, var in pat.repeating_regions:
            rep_by_gap.setdefault(pos + 1, []).append((tpl, var))
        opt_by_gap: dict[int, list[list[_PatternNode]]] = {}
        for pos, children in pat.optional_regions:
            opt_by_gap.setdefault(pos + 1, []).append(children)

        pi = 0
        for gap_idx in range(len(backbone) + 1):
            # Consume repeating regions in this gap
            for rtpl, rvar in rep_by_gap.get(gap_idx, []):
                items: list[dict[str, Any]] = []
                while pi < len(page_children) and _matches_repeat_structure(
                    rtpl, page_children[pi]
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

            # Consume optional regions in this gap — skeleton check only
            # (check_values=False) because optional templates are often
            # built from few examples and may be over-specific about values.
            for opt_children in opt_by_gap.get(gap_idx, []):
                for oc in opt_children:
                    if (
                        pi < len(page_children)
                        and element_tag(page_children[pi]) == oc.tag
                        and _matches_repeat_structure(
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
        pat: _PatternNode,
        elem: etree._Element,
        prefix: str,
        mask: dict[str, bool],
    ) -> bool:
        if pat.tag != element_tag(elem):
            return False
        if pat.attr_names != tuple(sorted(elem.attrib)):
            return False

        # Text
        text_key = f"{prefix}/text" if prefix else "text"
        is_fixed, val = pat.text
        page_text = elem.text or ""
        if is_fixed:
            if val != page_text:
                return False
            if page_text:
                mask[text_key] = True
        else:
            if pat.text_always_present and not page_text:
                return False
            if page_text:
                mask[text_key] = False

        # Attributes
        for attr_name in sorted(elem.attrib):
            attr_key = f"{prefix}/@{attr_name}" if prefix else f"@{attr_name}"
            is_fixed, _ = pat.attrs[attr_name]
            mask[attr_key] = is_fixed

        # Children
        if pat.children is None:
            _mark_children_variable(elem, prefix, mask)
            return True

        page_children = list(elem)

        # Region-aware masking
        if pat.repeating_regions or pat.optional_regions:
            return self._mask_children(pat, page_children, prefix, mask)

        # Standard fixed-structure masking
        if len(pat.children) != len(page_children):
            return False

        for i, (tc, pc) in enumerate(zip(pat.children, page_children)):
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
        pat: _PatternNode,
        page_children: list[etree._Element],
        prefix: str,
        mask: dict[str, bool],
    ) -> bool:
        """Mask children with gap-indexed region dispatch."""
        assert pat.children is not None
        backbone = pat.children

        rep_by_gap: dict[int, list[tuple[_PatternNode, str]]] = {}
        for pos, tpl, var in pat.repeating_regions:
            rep_by_gap.setdefault(pos + 1, []).append((tpl, var))
        opt_by_gap: dict[int, list[list[_PatternNode]]] = {}
        for pos, children in pat.optional_regions:
            opt_by_gap.setdefault(pos + 1, []).append(children)

        pi = 0
        for gap_idx in range(len(backbone) + 1):
            # Repeating → all variable
            for rtpl, _ in rep_by_gap.get(gap_idx, []):
                while pi < len(page_children) and _matches_repeat_structure(
                    rtpl, page_children[pi]
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
                        and _matches_repeat_structure(
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches_repeat_structure(
    rtpl: _PatternNode,
    elem: etree._Element,
    *,
    check_values: bool = True,
) -> bool:
    """Check if an element could be an instance of a repeating template.

    Validates tag, attribute names, and children tag sequence so that
    structurally different pages (e.g. ``a()`` vs ``a(abbr())``) are
    rejected.

    When *check_values* is True (default), also verifies fixed text and
    attribute values.  Set to False for optional children whose templates
    are built from few examples and may be over-specific about values.
    """
    if rtpl.tag != element_tag(elem):
        return False
    if rtpl.attr_names != tuple(sorted(elem.attrib)):
        return False
    if check_values:
        for name in rtpl.attr_names:
            is_fixed, val = rtpl.attrs[name]
            if is_fixed and elem.attrib[name] != val:
                return False
        is_fixed, val = rtpl.text
        if is_fixed and val != (elem.text or ""):
            return False
    # Validate children structure (tag sequence must match template).
    # When the template has repeating/optional regions, validate that
    # the element's children are compatible with the backbone + regions
    # by recursively checking region elements against their templates.
    if (rtpl.repeating_regions or rtpl.optional_regions) and rtpl.children is not None:
        elem_children = list(elem)
        backbone = rtpl.children
        # Build lookups per gap position
        rep_by_gap: dict[int, list[_PatternNode]] = {}
        for pos, tpl, _var in rtpl.repeating_regions:
            rep_by_gap.setdefault(pos + 1, []).append(tpl)
        opt_by_gap: dict[int, list[_PatternNode]] = {}
        for pos, children in rtpl.optional_regions:
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
                # Recursively validate against the region template
                if child_tag in rep_tpls:
                    if not _matches_repeat_structure(
                        rep_tpls[child_tag], child, check_values=check_values
                    ):
                        ok = False
                        break
                elif child_tag in opt_tpls:
                    if not _matches_repeat_structure(
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
                # Recursively validate backbone child
                if not _matches_repeat_structure(
                    backbone[gap_idx], elem_children[pi], check_values=check_values
                ):
                    ok = False
                    break
                pi += 1
        if not ok or pi != len(elem_children):
            return False
    elif rtpl.children is not None:
        elem_children = list(elem)
        if len(elem_children) != len(rtpl.children):
            return False
        for ctpl, celem in zip(rtpl.children, elem_children):
            if ctpl.tag != element_tag(celem):
                return False
    elif list(elem):
        # Template has no children (leaf or variable) but element does
        return False
    return True


def _relax_always_present(node: _PatternNode) -> None:
    """Clear always_present flags throughout a repeating region template.

    Within loops, any position that's variable should allow empty values —
    the flag is over-constraining for loop body templates.
    """
    node.text_always_present = False
    node.tail_always_present = False
    if node.children:
        for child in node.children:
            _relax_always_present(child)


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


class FiVaTechInferer:
    """FiVaTech-inspired template inferer (Kayed & Chang, IEEE TKDE 2010).

    Merges all input trees simultaneously into a single pattern tree using
    LCS-based child sequence alignment, classifying each node position as
    fixed or variant.
    """

    def infer(
        self, pages: Sequence[etree._Element]
    ) -> FiVaTechTemplate | EmptyTemplate:
        ctr = _Counter()
        result = _merge_trees(list(pages), ctr)
        if result is None:
            return EmptyTemplate()
        return FiVaTechTemplate(result)
