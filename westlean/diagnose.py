"""Diagnose why a tree-based template fails to match a page.

Works with AntiUnifiedTemplate, FiVaTechTemplate, and KTestableTemplate
since they all share the same internal node structure (_TplNode/_PatternNode/
_UTANode with identical field names).
"""

from __future__ import annotations

from typing import Any

from lxml import etree

from westlean.compat import element_tag


def diagnose_mismatch(template: Any, page: etree._Element) -> str | None:
    """Return a human-readable description of why *template* rejects *page*.

    Returns ``None`` if the page actually matches (no mismatch).
    """
    root = getattr(template, "_root", None)
    if root is None:
        return "template has no _root (EmptyTemplate?)"
    return _diag_node(root, page, "")


def _struct_matches(tpl: Any, elem: etree._Element) -> bool:
    """Lightweight structural match (tags + attr names, recursive, no values).

    Mirrors kTestable's _matches_structure(check_values=False).
    """
    if tpl.tag != element_tag(elem):
        return False
    if tpl.attr_names != tuple(sorted(elem.attrib)):
        return False
    if tpl.children is None:
        # Variable children — accept anything
        return True
    elem_children = list(elem)
    if (tpl.repeating_regions or tpl.optional_regions) and tpl.children is not None:
        # Region matching — simplified: check backbone reachability
        pi = 0
        for gap_idx in range(len(tpl.children) + 1):
            rep_tags = {r[1].tag for r in tpl.repeating_regions if r[0] + 1 == gap_idx}
            opt_tags: set[str] = set()
            for pos, children in tpl.optional_regions:
                if pos + 1 == gap_idx:
                    opt_tags.update(c.tag for c in children)
            known = rep_tags | opt_tags
            while pi < len(elem_children) and element_tag(elem_children[pi]) in known:
                pi += 1
            if gap_idx < len(tpl.children):
                if pi >= len(elem_children):
                    return False
                if not _struct_matches(tpl.children[gap_idx], elem_children[pi]):
                    return False
                pi += 1
        return pi == len(elem_children)
    if len(tpl.children) != len(elem_children):
        return False
    for tc, pc in zip(tpl.children, elem_children):
        if not _struct_matches(tc, pc):
            return False
    return True


def _diag_node(tpl: Any, elem: etree._Element, path: str) -> str | None:
    tag = element_tag(elem)
    node_path = f"{path}/{tpl.tag}" if path else tpl.tag

    if tpl.tag != tag:
        return f"{node_path}: tag mismatch: template={tpl.tag!r} page={tag!r}"

    tpl_attrs = tpl.attr_names
    page_attrs = tuple(sorted(elem.attrib))
    if tpl_attrs != page_attrs:
        return (
            f"{node_path}: attribute names mismatch: "
            f"template={list(tpl_attrs)} page={list(page_attrs)}"
        )

    # Text
    is_fixed, val = tpl.text
    page_text = elem.text or ""
    if is_fixed and val != page_text:
        return f"{node_path}: fixed text mismatch: template={val!r} page={page_text!r}"
    if not is_fixed and tpl.text_always_present and not page_text:
        return f"{node_path}: text required but page has none"

    # Attributes
    for name in tpl_attrs:
        is_fixed, pat_val = tpl.attrs[name]
        page_val = elem.attrib[name]
        if is_fixed and pat_val != page_val:
            return (
                f"{node_path}/@{name}: fixed attr mismatch: "
                f"template={pat_val!r} page={page_val!r}"
            )

    # Variable children — always matches
    if tpl.children is None:
        return None

    page_children = list(elem)

    # Region-aware matching
    if tpl.repeating_regions or tpl.optional_regions:
        return _diag_children(tpl, page_children, node_path)

    # Fixed-structure matching
    if len(tpl.children) != len(page_children):
        return (
            f"{node_path}: children count mismatch: "
            f"template={len(tpl.children)} page={len(page_children)}"
        )

    for i, (tc, pc) in enumerate(zip(tpl.children, page_children)):
        result = _diag_node(tc, pc, node_path)
        if result is not None:
            return result
        is_fixed, val = tc.tail
        page_tail = pc.tail or ""
        if is_fixed and val != page_tail:
            return (
                f"{node_path}/{tc.tag}[{i}]/tail: fixed tail mismatch: "
                f"template={val!r} page={page_tail!r}"
            )
        if not is_fixed and tc.tail_always_present and not page_tail:
            return f"{node_path}/{tc.tag}[{i}]/tail: tail required but page has none"

    return None


def _diag_children(
    tpl: Any, page_children: list[etree._Element], path: str
) -> str | None:
    """Diagnose region-aware child matching."""
    backbone = tpl.children
    assert backbone is not None

    rep_by_gap: dict[int, list[Any]] = {}
    for pos, rtpl, _var in tpl.repeating_regions:
        rep_by_gap.setdefault(pos + 1, []).append(rtpl)
    opt_by_gap: dict[int, list[Any]] = {}
    for pos, children in tpl.optional_regions:
        for c in children:
            opt_by_gap.setdefault(pos + 1, []).append(c)

    pi = 0
    for gap_idx in range(len(backbone) + 1):
        # Consume repeating region elements (using structural match)
        for rtpl in rep_by_gap.get(gap_idx, []):
            while pi < len(page_children) and _struct_matches(rtpl, page_children[pi]):
                pi += 1

        # Consume optional region elements
        for oc in opt_by_gap.get(gap_idx, []):
            if pi < len(page_children) and _struct_matches(oc, page_children[pi]):
                pi += 1

        # Match backbone child
        if gap_idx < len(backbone):
            bp = backbone[gap_idx]
            if pi >= len(page_children):
                return (
                    f"{path}: ran out of page children at position {pi}, "
                    f"expected backbone {bp.tag!r} (backbone index {gap_idx})"
                )
            pc = page_children[pi]
            pc_tag = element_tag(pc)
            if bp.tag != pc_tag:
                return (
                    f"{path}: at position {pi}, expected backbone {bp.tag!r} "
                    f"but found {pc_tag!r}"
                )
            if not _struct_matches(bp, pc):
                return (
                    f"{path}: backbone {bp.tag!r} at position {pi} "
                    f"structurally incompatible (attr names or children differ)"
                )
            result = _diag_node(bp, pc, path)
            if result is not None:
                return result
            # Check tail
            is_fixed, val = bp.tail
            page_tail = pc.tail or ""
            if is_fixed and val != page_tail:
                return (
                    f"{path}/{bp.tag}[{pi}]/tail: fixed tail mismatch: "
                    f"template={val!r} page={page_tail!r}"
                )
            pi += 1

    if pi != len(page_children):
        # Figure out WHY leftover children weren't consumed
        leftover_tags = [
            element_tag(page_children[j])
            for j in range(pi, min(pi + 5, len(page_children)))
        ]
        # Check if any leftover matches a repeat template structurally
        for rtpl_list in rep_by_gap.values():
            for rtpl in rtpl_list:
                for j in range(pi, len(page_children)):
                    pc = page_children[j]
                    if element_tag(pc) == rtpl.tag and not _struct_matches(rtpl, pc):
                        return (
                            f"{path}: repeat element {rtpl.tag!r} at position {j} "
                            f"has matching tag but incompatible structure "
                            f"(attr names or children differ)"
                        )
        return (
            f"{path}: {len(page_children) - pi} unconsumed children "
            f"after position {pi}: {leftover_tags}"
        )

    return None
