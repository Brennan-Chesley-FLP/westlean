"""Anti-unification (Plotkin 1970 / Reynolds 1970) template inference.

Computes the Least General Generalization (LGG) of DOM trees by walking
them in parallel, keeping common structure and replacing differing
text/attribute positions with named variables.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from lxml import etree

from westlean.child_alignment import lcs
from westlean.protocol import EmptyTemplate


# ---------------------------------------------------------------------------
# Internal template representation
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
class _TplNode:
    """Node in the anti-unified template tree."""

    tag: str
    attr_names: tuple[str, ...]  # sorted attribute names (structural)
    text: _Slot  # element's .text (before first child)
    tail: _Slot  # element's .tail (after it, in parent context)
    attrs: dict[str, _Slot]  # attr_name -> slot
    children: list[_TplNode] | None  # None => children region is variable
    children_var: str | None  # variable name when children is None
    text_always_present: bool = False  # was .text non-empty in ALL training pages?
    tail_always_present: bool = False  # was .tail non-empty in ALL training pages?
    # Multi-region support: each region is (after_backbone_pos, child_template, var_name)
    repeating_regions: list[tuple[int, _TplNode, str]] = field(default_factory=list)
    # Each optional region is (after_backbone_pos, children_templates)
    optional_regions: list[tuple[int, list[_TplNode]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Convert a single page into a fully-fixed template
# ---------------------------------------------------------------------------


def _page_to_template(elem: etree._Element) -> _TplNode:
    attr_names = tuple(sorted(elem.attrib))
    return _TplNode(
        tag=str(elem.tag),
        attr_names=attr_names,
        text=(True, elem.text or ""),
        tail=(True, elem.tail or ""),
        attrs={name: (True, elem.attrib[name]) for name in attr_names},
        children=[_page_to_template(child) for child in elem],
        children_var=None,
        text_always_present=bool(elem.text),
        tail_always_present=bool(elem.tail),
    )


# ---------------------------------------------------------------------------
# Anti-unify two _TplNode trees (for merging repeating children)
# ---------------------------------------------------------------------------


def _anti_unify_nodes(a: _TplNode, b: _TplNode, ctr: _Counter) -> _TplNode | None:
    """Anti-unify two _TplNode trees into their LGG."""
    if a.tag != b.tag or a.attr_names != b.attr_names:
        return None

    # Text
    if a.text[0] and b.text[0]:
        text: _Slot = a.text if a.text[1] == b.text[1] else (False, ctr.next())
    else:
        text = a.text if not a.text[0] else b.text
    text_ap = a.text_always_present and b.text_always_present

    # Tail
    if a.tail[0] and b.tail[0]:
        tail: _Slot = a.tail if a.tail[1] == b.tail[1] else (False, ctr.next())
    else:
        tail = a.tail if not a.tail[0] else b.tail
    tail_ap = a.tail_always_present and b.tail_always_present

    # Attrs
    attrs: dict[str, _Slot] = {}
    for name in a.attr_names:
        af, av = a.attrs[name]
        bf, bv = b.attrs[name]
        if af and bf:
            attrs[name] = (True, av) if av == bv else (False, ctr.next())
        else:
            attrs[name] = a.attrs[name] if not af else b.attrs[name]

    # Children
    if a.children is None or b.children is None:
        var = a.children_var or b.children_var or ctr.next()
        return _TplNode(
            a.tag, a.attr_names, text, tail, attrs, None, var, text_ap, tail_ap
        )

    # If one side already has regions from a previous merge, fold the
    # other side's children against those regions.
    if a.repeating_regions or a.optional_regions:
        result = _fold_with_regions_nodes(a, b.children, ctr)
        if result is not None:
            bb, rr, orr = result
            return _TplNode(
                a.tag,
                a.attr_names,
                text,
                tail,
                attrs,
                bb,
                None,
                text_ap,
                tail_ap,
                rr,
                orr,
            )
    if b.repeating_regions or b.optional_regions:
        result = _fold_with_regions_nodes(b, a.children, ctr)
        if result is not None:
            bb, rr, orr = result
            return _TplNode(
                a.tag,
                a.attr_names,
                text,
                tail,
                attrs,
                bb,
                None,
                text_ap,
                tail_ap,
                rr,
                orr,
            )

    if len(a.children) != len(b.children):
        # Try region detection (nested loops) before collapsing
        result = _try_detect_regions_nodes(a.children, b.children, ctr)
        if result is not None:
            bb, rr, orr = result
            return _TplNode(
                a.tag,
                a.attr_names,
                text,
                tail,
                attrs,
                bb,
                None,
                text_ap,
                tail_ap,
                rr,
                orr,
            )
        return _TplNode(
            a.tag, a.attr_names, text, tail, attrs, None, ctr.next(), text_ap, tail_ap
        )

    new_children: list[_TplNode] = []
    pairwise_ok = True
    for ac, bc in zip(a.children, b.children):
        merged = _anti_unify_nodes(ac, bc, ctr)
        if merged is None:
            pairwise_ok = False
            break
        new_children.append(merged)

    if pairwise_ok:
        return _TplNode(
            a.tag, a.attr_names, text, tail, attrs, new_children, None, text_ap, tail_ap
        )

    # Pairwise failed — try region detection even with same count
    result = _try_detect_regions_nodes(a.children, b.children, ctr)
    if result is not None:
        bb, rr, orr = result
        return _TplNode(
            a.tag, a.attr_names, text, tail, attrs, bb, None, text_ap, tail_ap, rr, orr
        )
    return _TplNode(
        a.tag, a.attr_names, text, tail, attrs, None, ctr.next(), text_ap, tail_ap
    )


# ---------------------------------------------------------------------------
# Region-aware folding: TplNode-vs-TplNode (nested loops)
# ---------------------------------------------------------------------------


def _fold_with_regions_nodes(
    tpl: _TplNode,
    other_children: list[_TplNode],
    ctr: _Counter,
) -> (
    tuple[
        list[_TplNode],
        list[tuple[int, _TplNode, str]],
        list[tuple[int, list[_TplNode]]],
    ]
    | None
):
    """Fold TplNode children against a template with established regions.

    Like ``_fold_with_regions`` but the incoming children are TplNodes
    (from a previous merge) rather than page elements.
    """
    backbone = tpl.children
    assert backbone is not None

    rep_by_gap: dict[int, list[tuple[int, _TplNode, str]]] = {}
    for pos, rtpl, rvar in tpl.repeating_regions:
        rep_by_gap.setdefault(pos + 1, []).append((pos, rtpl, rvar))
    opt_by_gap: dict[int, list[tuple[int, list[_TplNode]]]] = {}
    for pos, children in tpl.optional_regions:
        opt_by_gap.setdefault(pos + 1, []).append((pos, children))

    pi = 0
    new_backbone: list[_TplNode] = []
    new_repeating: list[tuple[int, _TplNode, str]] = []
    new_optional: list[tuple[int, list[_TplNode]]] = []

    for gap_idx in range(len(backbone) + 1):
        # Consume repeating regions
        for pos, rtpl, rvar in rep_by_gap.get(gap_idx, []):
            new_rtpl = rtpl
            while pi < len(other_children) and other_children[pi].tag == rtpl.tag:
                merged = _anti_unify_nodes(new_rtpl, other_children[pi], ctr)
                if merged is None:
                    break
                new_rtpl = merged
                pi += 1
            new_repeating.append((pos, new_rtpl, rvar))

        # Consume optional regions
        for pos, opt_children in opt_by_gap.get(gap_idx, []):
            remaining: list[_TplNode] = []
            for opt in opt_children:
                if pi < len(other_children) and other_children[pi].tag == opt.tag:
                    merged = _anti_unify_nodes(opt, other_children[pi], ctr)
                    remaining.append(merged if merged is not None else opt)
                    pi += 1
                else:
                    remaining.append(opt)
            if remaining:
                new_optional.append((pos, remaining))

        # Match backbone child
        if gap_idx < len(backbone):
            bp = backbone[gap_idx]
            if pi >= len(other_children):
                return None
            merged = _anti_unify_nodes(bp, other_children[pi], ctr)
            if merged is None:
                return None
            new_backbone.append(merged)
            pi += 1

    if pi != len(other_children):
        return None

    return (new_backbone, new_repeating, new_optional)


# ---------------------------------------------------------------------------
# Region detection: TplNode-vs-TplNode (nested loops)
# ---------------------------------------------------------------------------


def _try_detect_regions_nodes(
    a_children: list[_TplNode],
    b_children: list[_TplNode],
    ctr: _Counter,
) -> (
    tuple[
        list[_TplNode],
        list[tuple[int, _TplNode, str]],
        list[tuple[int, list[_TplNode]]],
    ]
    | None
):
    """Detect regions from two mismatched TplNode child sequences.

    Same logic as ``_try_detect_regions`` but operates on two TplNode lists
    (used when anti-unifying repeat body templates that have different child
    counts, i.e. nested loops).
    """
    a_tags = [c.tag for c in a_children]
    b_tags = [c.tag for c in b_children]

    a_counts = Counter(a_tags)
    b_counts = Counter(b_tags)
    variable = {
        t for t in set(a_tags) | set(b_tags) if a_counts.get(t, 0) != b_counts.get(t, 0)
    }

    stable_a = [t for t in a_tags if t not in variable]
    stable_b = [t for t in b_tags if t not in variable]
    backbone_tags = lcs(stable_a, stable_b)

    # Value-aware fallback (TplNode variant)
    if not backbone_tags:
        candidates = {
            t for t in variable if a_counts.get(t, 0) >= 1 and b_counts.get(t, 0) >= 1
        }
        if len(candidates) == 1:
            ctag = next(iter(candidates))
            first_a = next(c for c in a_children if c.tag == ctag)
            first_b = next(c for c in b_children if c.tag == ctag)
            a_fixed, a_val = first_a.text
            b_fixed, b_val = first_b.text
            if a_fixed and b_fixed and a_val == b_val:
                has_different = any(
                    c.tag == ctag
                    and c is not first_a
                    and (not c.text[0] or c.text[1] != a_val)
                    for c in a_children
                ) or any(
                    c.tag == ctag
                    and c is not first_b
                    and (not c.text[0] or c.text[1] != a_val)
                    for c in b_children
                )
                if has_different:
                    backbone_tags = [ctag]
                    variable = variable - candidates

    def _map(tags: list[str], bb: list[str], skip: set[str]) -> list[int] | None:
        result: list[int] = []
        j = 0
        for bt in bb:
            while j < len(tags) and (tags[j] != bt or tags[j] in skip):
                j += 1
            if j >= len(tags):
                return None
            result.append(j)
            j += 1
        return result

    a_bi = _map(a_tags, backbone_tags, variable)
    b_bi = _map(b_tags, backbone_tags, variable)
    if a_bi is None or b_bi is None:
        return None

    # Anti-unify backbone children
    new_backbone: list[_TplNode] = []
    for i in range(len(backbone_tags)):
        merged = _anti_unify_nodes(a_children[a_bi[i]], b_children[b_bi[i]], ctr)
        if merged is None:
            return None
        new_backbone.append(merged)

    repeating_regions: list[tuple[int, _TplNode, str]] = []
    optional_regions: list[tuple[int, list[_TplNode]]] = []

    for gap in range(len(backbone_tags) + 1):
        after_backbone_pos = gap - 1
        a_start = 0 if gap == 0 else a_bi[gap - 1] + 1
        a_end = len(a_children) if gap == len(backbone_tags) else a_bi[gap]
        b_start = 0 if gap == 0 else b_bi[gap - 1] + 1
        b_end = len(b_children) if gap == len(backbone_tags) else b_bi[gap]

        ac = a_end - a_start
        bc = b_end - b_start
        if ac == 0 and bc == 0:
            continue

        gap_tags = set()
        for i in range(a_start, a_end):
            gap_tags.add(a_children[i].tag)
        for i in range(b_start, b_end):
            gap_tags.add(b_children[i].tag)

        if len(gap_tags) == 1 and max(ac, bc) > 1:
            # Repeating region
            all_nodes = list(a_children[a_start:a_end]) + list(
                b_children[b_start:b_end]
            )
            seed = all_nodes[0]
            for node in all_nodes[1:]:
                merged = _anti_unify_nodes(seed, node, ctr)
                if merged is None:
                    break
                seed = merged
            else:
                repeating_regions.append((after_backbone_pos, seed, ctr.next()))

        elif len(gap_tags) == 1 and max(ac, bc) == 1 and min(ac, bc) == 0:
            # Optional
            if bc > 0:
                optional_regions.append(
                    (after_backbone_pos, list(b_children[b_start:b_end]))
                )
            elif ac > 0:
                optional_regions.append(
                    (after_backbone_pos, list(a_children[a_start:a_end]))
                )

        elif len(gap_tags) > 1 and ac > 0 and bc > 0:
            # Multi-tag gap — decompose per tag
            tag_order: list[str] = []
            for i in range(a_start, a_end):
                if a_children[i].tag not in tag_order:
                    tag_order.append(a_children[i].tag)
            for i in range(b_start, b_end):
                if b_children[i].tag not in tag_order:
                    tag_order.append(b_children[i].tag)

            a_tc = Counter(a_children[i].tag for i in range(a_start, a_end))
            b_tc = Counter(b_children[i].tag for i in range(b_start, b_end))

            for t in tag_order:
                tc_t = a_tc.get(t, 0)
                pc_t = b_tc.get(t, 0)
                if tc_t == 0 and pc_t == 0:
                    continue
                if max(tc_t, pc_t) > 1:
                    all_nodes = [c for c in a_children[a_start:a_end] if c.tag == t]
                    all_nodes += [c for c in b_children[b_start:b_end] if c.tag == t]
                    seed = all_nodes[0]
                    ok = True
                    for node in all_nodes[1:]:
                        merged = _anti_unify_nodes(seed, node, ctr)
                        if merged is None:
                            ok = False
                            break
                        seed = merged
                    if ok:
                        repeating_regions.append((after_backbone_pos, seed, ctr.next()))
                elif max(tc_t, pc_t) == 1 and min(tc_t, pc_t) == 0:
                    if pc_t > 0:
                        optional_regions.append(
                            (
                                after_backbone_pos,
                                [c for c in b_children[b_start:b_end] if c.tag == t],
                            )
                        )
                    else:
                        optional_regions.append(
                            (
                                after_backbone_pos,
                                [c for c in a_children[a_start:a_end] if c.tag == t],
                            )
                        )

        elif len(gap_tags) > 1 and (ac == 0 or bc == 0):
            # Multi-tag optional
            if bc > 0:
                optional_regions.append(
                    (after_backbone_pos, list(b_children[b_start:b_end]))
                )
            else:
                optional_regions.append(
                    (after_backbone_pos, list(a_children[a_start:a_end]))
                )

    if not new_backbone and not repeating_regions and not optional_regions:
        return None

    return (new_backbone, repeating_regions, optional_regions)


# ---------------------------------------------------------------------------
# Region detection: first-time child count mismatch
# ---------------------------------------------------------------------------


def _try_detect_regions(
    tpl_children: list[_TplNode],
    page_children: list[etree._Element],
    ctr: _Counter,
) -> (
    tuple[
        list[_TplNode],
        list[tuple[int, _TplNode, str]],
        list[tuple[int, list[_TplNode]]],
    ]
    | None
):
    """Detect backbone + repeating + optional from two mismatched child sequences.

    Returns (backbone, repeating_regions, optional_regions) or None.
    """
    tpl_tags = [c.tag for c in tpl_children]
    page_tags = [str(c.tag) for c in page_children]

    tpl_counts = Counter(tpl_tags)
    page_counts = Counter(page_tags)
    variable = {
        t
        for t in set(tpl_tags) | set(page_tags)
        if tpl_counts.get(t, 0) != page_counts.get(t, 0)
    }

    stable_tpl = [t for t in tpl_tags if t not in variable]
    stable_page = [t for t in page_tags if t not in variable]
    backbone_tags = lcs(stable_tpl, stable_page)

    # Value-aware fallback: when backbone is empty and a single
    # variable-count tag is always present, check if the first
    # instance has identical values (fixed element + loop pattern).
    if not backbone_tags:
        candidates = {
            t
            for t in variable
            if tpl_counts.get(t, 0) >= 1 and page_counts.get(t, 0) >= 1
        }
        if len(candidates) == 1:
            ctag = next(iter(candidates))
            # Compare first TplNode's fixed text+attrs vs first page element
            first_tpl = next(c for c in tpl_children if c.tag == ctag)
            first_page = next(c for c in page_children if c.tag == ctag)
            tpl_text_fixed, tpl_text_val = first_tpl.text
            page_text = first_page.text or ""
            if tpl_text_fixed and tpl_text_val == page_text:
                # First instance has matching fixed text.  Also verify
                # other instances differ — distinguishes "fixed + loop"
                # from "coincidentally same first items."
                has_different = any(
                    c.tag == ctag
                    and c is not first_tpl
                    and (not c.text[0] or c.text[1] != tpl_text_val)
                    for c in tpl_children
                ) or any(
                    c.tag == ctag
                    and c is not first_page
                    and (c.text or "") != tpl_text_val
                    for c in page_children
                )
                if has_different:
                    backbone_tags = [ctag]
                    variable = variable - candidates

    # Map sequences to backbone indices (skipping variable-count tags)
    def _map(tags: list[str], bb: list[str], skip: set[str]) -> list[int] | None:
        result: list[int] = []
        j = 0
        for bt in bb:
            while j < len(tags) and (tags[j] != bt or tags[j] in skip):
                j += 1
            if j >= len(tags):
                return None
            result.append(j)
            j += 1
        return result

    tpl_bi = _map(tpl_tags, backbone_tags, variable)
    page_bi = _map(page_tags, backbone_tags, variable)
    if tpl_bi is None or page_bi is None:
        return None

    # Anti-unify backbone children
    new_backbone: list[_TplNode] = []
    for i in range(len(backbone_tags)):
        merged = _anti_unify(tpl_children[tpl_bi[i]], page_children[page_bi[i]], ctr)
        if merged is None:
            return None
        new_backbone.append(merged)

    # Analyze gaps for repeating/optional — accumulate into lists
    repeating_regions: list[tuple[int, _TplNode, str]] = []
    optional_regions: list[tuple[int, list[_TplNode]]] = []

    for gap in range(len(backbone_tags) + 1):
        # after_backbone_pos is gap - 1 (i.e. -1 for before first backbone child)
        after_backbone_pos = gap - 1

        t_start = 0 if gap == 0 else tpl_bi[gap - 1] + 1
        t_end = len(tpl_children) if gap == len(backbone_tags) else tpl_bi[gap]
        p_start = 0 if gap == 0 else page_bi[gap - 1] + 1
        p_end = len(page_children) if gap == len(backbone_tags) else page_bi[gap]

        tc = t_end - t_start
        pc = p_end - p_start

        if tc == 0 and pc == 0:
            continue

        gap_tags: set[str] = set()
        for i in range(t_start, t_end):
            gap_tags.add(tpl_children[i].tag)
        for i in range(p_start, p_end):
            gap_tags.add(str(page_children[i].tag))

        if len(gap_tags) == 1 and max(tc, pc) > 1:
            # Repeating region — merge all gap children into one template
            all_nodes: list[_TplNode] = []
            for i in range(t_start, t_end):
                all_nodes.append(tpl_children[i])
            for i in range(p_start, p_end):
                all_nodes.append(_page_to_template(page_children[i]))

            seed = all_nodes[0]
            for node in all_nodes[1:]:
                result = _anti_unify_nodes(seed, node, ctr)
                if result is None:
                    break
                seed = result
            else:
                repeating_regions.append((after_backbone_pos, seed, ctr.next()))

        elif len(gap_tags) == 1 and max(tc, pc) == 1 and min(tc, pc) == 0:
            # Optional region
            if pc > 0:
                optional_regions.append(
                    (
                        after_backbone_pos,
                        [
                            _page_to_template(page_children[i])
                            for i in range(p_start, p_end)
                        ],
                    )
                )
            elif tc > 0:
                optional_regions.append(
                    (after_backbone_pos, list(tpl_children[t_start:t_end]))
                )

        elif len(gap_tags) > 1:
            if (tc > 0 and pc == 0) or (tc == 0 and pc > 0):
                # Multi-tag optional (present in one side only)
                if pc > 0:
                    optional_regions.append(
                        (
                            after_backbone_pos,
                            [
                                _page_to_template(page_children[i])
                                for i in range(p_start, p_end)
                            ],
                        )
                    )
                else:
                    optional_regions.append(
                        (after_backbone_pos, list(tpl_children[t_start:t_end]))
                    )
            elif tc > 0 and pc > 0:
                # Multi-tag gap with children on both sides — decompose per tag
                tag_order: list[str] = []
                for i in range(t_start, t_end):
                    if tpl_children[i].tag not in tag_order:
                        tag_order.append(tpl_children[i].tag)
                for i in range(p_start, p_end):
                    ptag = str(page_children[i].tag)
                    if ptag not in tag_order:
                        tag_order.append(ptag)

                tpl_tc = Counter(tpl_children[i].tag for i in range(t_start, t_end))
                page_tc = Counter(
                    str(page_children[i].tag) for i in range(p_start, p_end)
                )

                # Validate: tags must appear in consistent order (contiguous runs)
                valid = True
                for seq_tags in [
                    [tpl_children[i].tag for i in range(t_start, t_end)],
                    [str(page_children[i].tag) for i in range(p_start, p_end)],
                ]:
                    seen: list[str] = []
                    for t in seq_tags:
                        if not seen or seen[-1] != t:
                            seen.append(t)
                    ti2 = 0
                    for st in seen:
                        while ti2 < len(tag_order) and tag_order[ti2] != st:
                            ti2 += 1
                        if ti2 >= len(tag_order):
                            valid = False
                            break
                        ti2 += 1
                    if not valid:
                        break

                if valid:
                    for t in tag_order:
                        tc_t = tpl_tc.get(t, 0)
                        pc_t = page_tc.get(t, 0)
                        if tc_t == 0 and pc_t == 0:
                            continue
                        if max(tc_t, pc_t) > 1:
                            # Repeating — merge all instances of this tag
                            tag_nodes: list[_TplNode] = []
                            for i in range(t_start, t_end):
                                if tpl_children[i].tag == t:
                                    tag_nodes.append(tpl_children[i])
                            for i in range(p_start, p_end):
                                if str(page_children[i].tag) == t:
                                    tag_nodes.append(
                                        _page_to_template(page_children[i])
                                    )
                            seed = tag_nodes[0]
                            ok = True
                            for node in tag_nodes[1:]:
                                merged = _anti_unify_nodes(seed, node, ctr)
                                if merged is None:
                                    ok = False
                                    break
                                seed = merged
                            if ok:
                                repeating_regions.append(
                                    (after_backbone_pos, seed, ctr.next())
                                )
                        elif max(tc_t, pc_t) == 1 and min(tc_t, pc_t) == 0:
                            # Optional — present in one side only
                            if pc_t > 0:
                                optional_regions.append(
                                    (
                                        after_backbone_pos,
                                        [
                                            _page_to_template(page_children[i])
                                            for i in range(p_start, p_end)
                                            if page_children[i].tag == t
                                        ],
                                    )
                                )
                            else:
                                optional_regions.append(
                                    (
                                        after_backbone_pos,
                                        [
                                            tpl_children[i]
                                            for i in range(t_start, t_end)
                                            if tpl_children[i].tag == t
                                        ],
                                    )
                                )

    if not new_backbone and not repeating_regions and not optional_regions:
        return None

    return (new_backbone, repeating_regions, optional_regions)


# ---------------------------------------------------------------------------
# Region-aware folding: template already has regions from previous fold
# ---------------------------------------------------------------------------


def _fold_with_regions(
    tpl: _TplNode,
    page_children: list[etree._Element],
    ctr: _Counter,
) -> (
    tuple[
        list[_TplNode],
        list[tuple[int, _TplNode, str]],
        list[tuple[int, list[_TplNode]]],
    ]
    | None
):
    """Fold page children against a template with established regions.

    Returns (backbone, repeating_regions, optional_regions) or None.
    """
    backbone = tpl.children
    assert backbone is not None

    # Build gap→regions lookups (gap_idx = after_backbone_pos + 1)
    rep_by_gap: dict[int, list[tuple[int, _TplNode, str]]] = {}
    for pos, rtpl, rvar in tpl.repeating_regions:
        rep_by_gap.setdefault(pos + 1, []).append((pos, rtpl, rvar))
    opt_by_gap: dict[int, list[tuple[int, list[_TplNode]]]] = {}
    for pos, children in tpl.optional_regions:
        opt_by_gap.setdefault(pos + 1, []).append((pos, children))

    pi = 0
    new_backbone: list[_TplNode] = []
    new_repeating: list[tuple[int, _TplNode, str]] = []
    new_optional: list[tuple[int, list[_TplNode]]] = []

    for gap_idx in range(len(backbone) + 1):
        # Consume repeating regions in this gap
        for pos, rtpl, rvar in rep_by_gap.get(gap_idx, []):
            new_rtpl = rtpl
            while pi < len(page_children) and page_children[pi].tag == rtpl.tag:
                if rtpl.attr_names != tuple(sorted(page_children[pi].attrib)):
                    break
                merged = _anti_unify(new_rtpl, page_children[pi], ctr)
                if merged is None:
                    return None
                new_rtpl = merged
                pi += 1
            new_repeating.append((pos, new_rtpl, rvar))

        # Consume optional regions in this gap (upgrade to repeating if needed)
        for pos, opt_children in opt_by_gap.get(gap_idx, []):
            remaining_optional: list[_TplNode] = []
            for opt in opt_children:
                if pi < len(page_children) and page_children[pi].tag == opt.tag:
                    # Count consecutive same-tag children to detect upgrade
                    lookahead = pi + 1
                    while (
                        lookahead < len(page_children)
                        and page_children[lookahead].tag == opt.tag
                    ):
                        lookahead += 1
                    if lookahead - pi > 1:
                        # Upgrade to repeating region
                        new_rtpl = opt
                        while (
                            pi < len(page_children) and page_children[pi].tag == opt.tag
                        ):
                            merged = _anti_unify(new_rtpl, page_children[pi], ctr)
                            if merged is None:
                                break
                            new_rtpl = merged
                            pi += 1
                        new_repeating.append((pos, new_rtpl, ctr.next()))
                    else:
                        # Stay optional
                        merged = _anti_unify(opt, page_children[pi], ctr)
                        if merged is not None:
                            remaining_optional.append(merged)
                        else:
                            remaining_optional.append(opt)
                        pi += 1
                else:
                    remaining_optional.append(opt)
            if remaining_optional:
                new_optional.append((pos, remaining_optional))

        # Match backbone child (if not past end)
        if gap_idx < len(backbone):
            bp = backbone[gap_idx]
            if pi >= len(page_children):
                return None
            merged = _anti_unify(bp, page_children[pi], ctr)
            if merged is None:
                return None
            new_backbone.append(merged)
            pi += 1

    if pi != len(page_children):
        return None

    return (new_backbone, new_repeating, new_optional)


# ---------------------------------------------------------------------------
# Pairwise anti-unification: template x page -> generalized template
# ---------------------------------------------------------------------------


def _anti_unify(tpl: _TplNode, page: etree._Element, ctr: _Counter) -> _TplNode | None:
    """Return the LGG of *tpl* and *page*, or ``None`` if roots are incompatible."""

    if tpl.tag != page.tag:
        return None
    page_attrs = tuple(sorted(page.attrib))
    if tpl.attr_names != page_attrs:
        return None

    # --- text slot ---
    page_text = page.text or ""
    text_present = bool(page.text)
    if tpl.text[0]:  # currently fixed
        text: _Slot = tpl.text if tpl.text[1] == page_text else (False, ctr.next())
    else:
        text = tpl.text
    text_always_present = tpl.text_always_present and text_present

    # --- tail slot ---
    page_tail = page.tail or ""
    tail_present = bool(page.tail)
    if tpl.tail[0]:
        tail: _Slot = tpl.tail if tpl.tail[1] == page_tail else (False, ctr.next())
    else:
        tail = tpl.tail
    tail_always_present = tpl.tail_always_present and tail_present

    # --- attribute slots ---
    attrs: dict[str, _Slot] = {}
    for name in tpl.attr_names:
        is_fixed, tpl_val = tpl.attrs[name]
        if is_fixed:
            attrs[name] = (
                (True, tpl_val) if tpl_val == page.attrib[name] else (False, ctr.next())
            )
        else:
            attrs[name] = tpl.attrs[name]

    # --- children ---
    if tpl.children is None:
        # Already variable from a previous fold.
        return _TplNode(
            tpl.tag,
            tpl.attr_names,
            text,
            tail,
            attrs,
            None,
            tpl.children_var,
            text_always_present,
            tail_always_present,
        )

    page_children = list(page)

    # Template has regions from a previous fold
    if tpl.repeating_regions or tpl.optional_regions:
        result = _fold_with_regions(tpl, page_children, ctr)
        if result is not None:
            bb, rr, orr = result
            return _TplNode(
                tpl.tag,
                tpl.attr_names,
                text,
                tail,
                attrs,
                bb,
                None,
                text_always_present,
                tail_always_present,
                rr,
                orr,
            )
        return _TplNode(
            tpl.tag,
            tpl.attr_names,
            text,
            tail,
            attrs,
            None,
            ctr.next(),
            text_always_present,
            tail_always_present,
        )

    # Child count mismatch — try region detection
    if len(tpl.children) != len(page_children):
        result = _try_detect_regions(tpl.children, page_children, ctr)
        if result is not None:
            bb, rr, orr = result
            return _TplNode(
                tpl.tag,
                tpl.attr_names,
                text,
                tail,
                attrs,
                bb,
                None,
                text_always_present,
                tail_always_present,
                rr,
                orr,
            )
        return _TplNode(
            tpl.tag,
            tpl.attr_names,
            text,
            tail,
            attrs,
            None,
            ctr.next(),
            text_always_present,
            tail_always_present,
        )

    # Same child count, no regions — try standard pairwise anti-unification
    new_children: list[_TplNode] = []
    pairwise_ok = True
    for tc, pc in zip(tpl.children, page_children):
        child = _anti_unify(tc, pc, ctr)
        if child is None:
            pairwise_ok = False
            break
        new_children.append(child)

    if pairwise_ok:
        return _TplNode(
            tpl.tag,
            tpl.attr_names,
            text,
            tail,
            attrs,
            new_children,
            None,
            text_always_present,
            tail_always_present,
        )

    # Pairwise failed (tag mismatch) — try region detection even with same count
    result = _try_detect_regions(tpl.children, page_children, ctr)
    if result is not None:
        bb, rr, orr = result
        return _TplNode(
            tpl.tag,
            tpl.attr_names,
            text,
            tail,
            attrs,
            bb,
            None,
            text_always_present,
            tail_always_present,
            rr,
            orr,
        )
    return _TplNode(
        tpl.tag,
        tpl.attr_names,
        text,
        tail,
        attrs,
        None,
        ctr.next(),
        text_always_present,
        tail_always_present,
    )


# ---------------------------------------------------------------------------
# RELAX NG generation from _TplNode tree
# ---------------------------------------------------------------------------

_RNG_NS = "http://relaxng.org/ns/structure/1.0"


def _tpl_node_to_relax_ng(root: _TplNode) -> str:
    """Convert a _TplNode template tree to a RELAX NG schema string."""
    from xml.etree.ElementTree import Element as XE, SubElement, tostring

    grammar = XE("grammar", xmlns=_RNG_NS)
    grammar.set("datatypeLibrary", "http://www.w3.org/2001/XMLSchema-datatypes")
    start = SubElement(grammar, "start")

    # Check if we need the "anyContent" wildcard definition
    needs_any = _needs_any_wildcard(root)
    if needs_any:
        # Define a recursive pattern that accepts any element tree.
        # Uses <zeroOrMore> of <choice> between text and any-element,
        # avoiding the <interleave>+<text> conflict.
        define = SubElement(grammar, "define", name="anyContent")
        zm = SubElement(define, "zeroOrMore")
        choice = SubElement(zm, "choice")
        SubElement(choice, "text")
        el = SubElement(choice, "element")
        SubElement(el, "anyName")
        zm_attr = SubElement(el, "zeroOrMore")
        attr = SubElement(zm_attr, "attribute")
        SubElement(attr, "anyName")
        SubElement(el, "ref", name="anyContent")

    _rng_node(root, start)
    return tostring(grammar, encoding="unicode", xml_declaration=True)


def _needs_any_wildcard(node: _TplNode) -> bool:
    """Check if any node in the tree has children_var (needs wildcard)."""
    if node.children_var:
        return True
    if node.children:
        for child in node.children:
            if _needs_any_wildcard(child):
                return True
    for _, rtpl, _ in node.repeating_regions:
        if _needs_any_wildcard(rtpl):
            return True
    for _, opt_children in node.optional_regions:
        for oc in opt_children:
            if _needs_any_wildcard(oc):
                return True
    return False


def _rng_node(node: _TplNode, parent: Any) -> None:
    """Append RELAX NG patterns for *node* to *parent*.

    Uses ``<interleave>`` with ``<text/>`` so that text content (both
    ``.text`` and ``.tail`` in lxml terms) is permitted anywhere within
    the element — matching how real HTML elements carry text.
    """
    from xml.etree.ElementTree import SubElement

    el = SubElement(parent, "element", name=node.tag)

    # --- Attributes ---
    for attr_name in node.attr_names:
        attr_el = SubElement(el, "attribute", name=attr_name)
        SubElement(attr_el, "text")

    # --- Child content wrapped in <interleave> with <text/> ---
    # This allows text nodes to appear anywhere among child elements,
    # which matches HTML's mixed-content model.
    #
    # Build an inner <group> of the child element patterns, then
    # wrap in <interleave> with <text/> so free text is permitted.

    # Collect child patterns into a temporary list, then decide
    # whether to use <interleave> or just <empty/>.
    child_patterns: list[tuple[str, Any]] = []  # (kind, data)

    if node.children is not None:
        rep_by_gap: dict[int, list[tuple[int, _TplNode, str]]] = {}
        for pos, rtpl, rvar in node.repeating_regions:
            rep_by_gap.setdefault(pos + 1, []).append((pos, rtpl, rvar))

        opt_by_gap: dict[int, list[tuple[int, list[_TplNode]]]] = {}
        for pos, children in node.optional_regions:
            opt_by_gap.setdefault(pos + 1, []).append((pos, children))

        for gap_idx in range(len(node.children) + 1):
            for _, rtpl, _ in rep_by_gap.get(gap_idx, []):
                child_patterns.append(("repeat", rtpl))
            for _, opt_children in opt_by_gap.get(gap_idx, []):
                child_patterns.append(("optional", opt_children))
            if gap_idx < len(node.children):
                child_patterns.append(("child", node.children[gap_idx]))

    elif node.children_var:
        child_patterns.append(("any", None))

    has_any = any(k == "any" for k, _ in child_patterns)

    if has_any:
        # children_var → accept any content. Use ref directly
        # (not inside <mixed> to avoid interleave+text conflict).
        SubElement(el, "ref", name="anyContent")
    elif child_patterns:
        # Use <mixed><group>...</group></mixed> for mixed content.
        mixed = SubElement(el, "mixed")
        group = SubElement(mixed, "group")
        for kind, data in child_patterns:
            if kind == "child":
                _rng_node(data, group)
            elif kind == "repeat":
                zm = SubElement(group, "zeroOrMore")
                _rng_node(data, zm)
            elif kind == "optional":
                opt = SubElement(group, "optional")
                for oc in data:
                    _rng_node(oc, opt)
    else:
        # Leaf element — just allow text
        SubElement(el, "text")


# ---------------------------------------------------------------------------
# Inferred template (public API)
# ---------------------------------------------------------------------------


class AntiUnifiedTemplate:
    """Template inferred by anti-unification."""

    def __init__(self, root: _TplNode) -> None:
        self._root = root

    def serialize(self) -> dict[str, Any]:
        from westlean.serialization import AntiUnifiedTemplateModel, tpl_node_to_model

        model = AntiUnifiedTemplateModel(root=tpl_node_to_model(self._root))
        return model.model_dump()

    def get_relax_ng(self) -> str:
        return _tpl_node_to_relax_ng(self._root)

    @classmethod
    def restore(cls, data: dict[str, Any]) -> AntiUnifiedTemplate:
        from westlean.serialization import AntiUnifiedTemplateModel

        model = AntiUnifiedTemplateModel.model_validate(data)
        return model.to_internal()

    # -- extract ----------------------------------------------------------

    def extract(self, page: etree._Element) -> dict[str, Any] | None:
        result: dict[str, Any] = {}
        if not self._extract_node(self._root, page, result):
            return None
        return result

    def _extract_node(
        self, tpl: _TplNode, elem: etree._Element, out: dict[str, Any]
    ) -> bool:
        if tpl.tag != elem.tag:
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
            is_fixed, tpl_val = tpl.attrs[name]
            page_val = elem.attrib[name]
            if is_fixed:
                if tpl_val != page_val:
                    return False
            else:
                out[tpl_val] = page_val

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

        # Standard fixed-structure extraction
        if len(tpl.children) != len(page_children):
            return False

        for tc, pc in zip(tpl.children, page_children):
            if not self._extract_node(tc, pc, out):
                return False
            # Tail of this child
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
        tpl: _TplNode,
        page_children: list[etree._Element],
        out: dict[str, Any],
    ) -> bool:
        """Extract data from children using gap-indexed region dispatch."""
        assert tpl.children is not None
        backbone = tpl.children

        # Build gap→regions lookups (gap_idx = after_backbone_pos + 1)
        rep_by_gap: dict[int, list[tuple[_TplNode, str]]] = {}
        for pos, rtpl, rvar in tpl.repeating_regions:
            rep_by_gap.setdefault(pos + 1, []).append((rtpl, rvar))
        opt_by_gap: dict[int, list[list[_TplNode]]] = {}
        for pos, children in tpl.optional_regions:
            opt_by_gap.setdefault(pos + 1, []).append(children)

        pi = 0
        for gap_idx in range(len(backbone) + 1):
            # Consume repeating regions in this gap
            for rtpl, rvar in rep_by_gap.get(gap_idx, []):
                items: list[dict[str, Any]] = []
                while pi < len(page_children) and page_children[pi].tag == rtpl.tag:
                    if not _structurally_compatible(rtpl, page_children[pi]):
                        break
                    item: dict[str, Any] = {}
                    if self._extract_node(rtpl, page_children[pi], item):
                        rc_tail_fixed, rc_tail_val = rtpl.tail
                        rc_page_tail = page_children[pi].tail or ""
                        if not rc_tail_fixed and rc_page_tail:
                            item[rc_tail_val] = rc_page_tail
                        items.append(item)
                    # Skip items that fail extraction instead of breaking
                    pi += 1
                if items:
                    out[rvar] = items

            # Consume optional regions in this gap
            for opt_children in opt_by_gap.get(gap_idx, []):
                for oc in opt_children:
                    if pi < len(page_children) and page_children[pi].tag == oc.tag:
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
        tpl: _TplNode,
        elem: etree._Element,
        prefix: str,
        mask: dict[str, bool],
    ) -> bool:
        if tpl.tag != elem.tag:
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
            # Tail
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
        tpl: _TplNode,
        page_children: list[etree._Element],
        prefix: str,
        mask: dict[str, bool],
    ) -> bool:
        """Mask children with gap-indexed region dispatch."""
        assert tpl.children is not None
        backbone = tpl.children

        rep_by_gap: dict[int, list[tuple[_TplNode, str]]] = {}
        for pos, rtpl, rvar in tpl.repeating_regions:
            rep_by_gap.setdefault(pos + 1, []).append((rtpl, rvar))
        opt_by_gap: dict[int, list[list[_TplNode]]] = {}
        for pos, children in tpl.optional_regions:
            opt_by_gap.setdefault(pos + 1, []).append(children)

        pi = 0
        for gap_idx in range(len(backbone) + 1):
            # Repeating regions -> all variable
            for rtpl, _ in rep_by_gap.get(gap_idx, []):
                while pi < len(page_children) and page_children[pi].tag == rtpl.tag:
                    if not _structurally_compatible(rtpl, page_children[pi]):
                        break
                    child_prefix = f"{prefix}/{pi}" if prefix else str(pi)
                    _mark_subtree_variable(page_children[pi], child_prefix, mask)
                    if page_children[pi].tail:
                        mask[f"{child_prefix}/tail"] = False
                    pi += 1

            # Optional regions -> all variable
            for opt_children in opt_by_gap.get(gap_idx, []):
                for oc in opt_children:
                    if pi < len(page_children) and page_children[pi].tag == oc.tag:
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


def _structurally_compatible(tpl: _TplNode, elem: etree._Element) -> bool:
    """Check if *elem* is structurally compatible with *tpl*.

    Validates tag, attribute names, fixed attribute/text values, and
    children tag sequence — enough to reject pages from a different
    template without full extraction.
    """
    if tpl.tag != elem.tag:
        return False
    if tpl.attr_names != tuple(sorted(elem.attrib)):
        return False
    for name in tpl.attr_names:
        is_fixed, val = tpl.attrs[name]
        if is_fixed and val != elem.attrib[name]:
            return False
    is_fixed, val = tpl.text
    if is_fixed and val != (elem.text or ""):
        return False
    # Validate children structure (tag sequence must match template).
    # When the template has repeating/optional regions, validate that
    # the element's children are compatible with the backbone + regions
    # by recursively checking region elements against their templates.
    if (tpl.repeating_regions or tpl.optional_regions) and tpl.children is not None:
        elem_children = list(elem)
        backbone = tpl.children
        rep_by_gap: dict[int, list[_TplNode]] = {}
        for pos, rtpl, _var in tpl.repeating_regions:
            rep_by_gap.setdefault(pos + 1, []).append(rtpl)
        opt_by_gap: dict[int, list[_TplNode]] = {}
        for pos, children in tpl.optional_regions:
            for c in children:
                opt_by_gap.setdefault(pos + 1, []).append(c)
        pi = 0
        ok = True
        for gap_idx in range(len(backbone) + 1):
            rep_tpls = {t.tag: t for t in rep_by_gap.get(gap_idx, [])}
            opt_tpls = {t.tag: t for t in opt_by_gap.get(gap_idx, [])}
            known = set(rep_tpls) | set(opt_tpls)
            while pi < len(elem_children) and str(elem_children[pi].tag) in known:
                child = elem_children[pi]
                child_tag = str(child.tag)
                if child_tag in rep_tpls:
                    if not _structurally_compatible(rep_tpls[child_tag], child):
                        ok = False
                        break
                elif child_tag in opt_tpls:
                    if not _structurally_compatible(opt_tpls[child_tag], child):
                        ok = False
                        break
                pi += 1
            if not ok:
                break
            if gap_idx < len(backbone):
                if (
                    pi >= len(elem_children)
                    or elem_children[pi].tag != backbone[gap_idx].tag
                ):
                    ok = False
                    break
                if not _structurally_compatible(backbone[gap_idx], elem_children[pi]):
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
            if ctpl.tag != celem.tag:
                return False
    elif list(elem):
        return False
    return True


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


class AntiUnificationInferer:
    """First-order anti-unification template inferer (Plotkin / Reynolds 1970).

    Folds pairwise LGG across all input pages to produce a template where
    matching structure is kept and differing content is replaced with variables.
    """

    def infer(
        self, pages: Sequence[etree._Element]
    ) -> AntiUnifiedTemplate | EmptyTemplate:
        ctr = _Counter()
        tpl = _page_to_template(pages[0])
        for page in pages[1:]:
            result = _anti_unify(tpl, page, ctr)
            if result is None:
                return EmptyTemplate()
            tpl = result
        return AntiUnifiedTemplate(tpl)
