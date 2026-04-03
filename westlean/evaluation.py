"""Evaluation utilities for comparing inferred templates against ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lxml import etree
from lxml.html import fragment_fromstring

from westlean.protocol import TemplateInferer
from westlean.renderer import render
from westlean.template_ast import Element
from westlean.data_schema import DataSchema


# ---------------------------------------------------------------------------
# Parse helper
# ---------------------------------------------------------------------------


def parse_html(html_str: str) -> etree._Element:
    """Parse an HTML string into an lxml element tree."""
    return fragment_fromstring(html_str)


# ---------------------------------------------------------------------------
# Canonical position map
# ---------------------------------------------------------------------------


def build_position_map(root: etree._Element) -> dict[str, str]:
    """Build a map from canonical position keys to their string values."""
    positions: dict[str, str] = {}
    _walk_positions(root, "", positions)
    return positions


def _walk_positions(elem: etree._Element, prefix: str, out: dict[str, str]) -> None:
    text_key = f"{prefix}/text" if prefix else "text"
    if elem.text:
        out[text_key] = elem.text
    for attr_name in sorted(elem.attrib):
        out[f"{prefix}/@{attr_name}" if prefix else f"@{attr_name}"] = elem.attrib[
            attr_name
        ]
    for i, child in enumerate(elem):
        child_prefix = f"{prefix}/{i}" if prefix else str(i)
        _walk_positions(child, child_prefix, out)
        if child.tail:
            out[f"{child_prefix}/tail"] = child.tail


# ---------------------------------------------------------------------------
# Ground truth mask — two-render diff approach
# ---------------------------------------------------------------------------


def ground_truth_mask(template: Element, data: dict) -> dict[str, bool]:
    """Compute the ground-truth fixed/variable mask for a rendered page.

    Renders the template with *data* and with a perturbed copy where every
    string/int field is replaced with a unique sentinel.  Both renders go
    through the same ``render → parse_html`` pipeline the algorithms see,
    so position keys are guaranteed to align.  Positions whose values
    differ between the two renders are variable; identical positions are
    fixed.
    """
    # Build a "perturbed" data dict where every leaf value is unique
    perturbed = _perturb_data(data)

    html_original = render(template, data)
    html_perturbed = render(template, perturbed)

    page_original = parse_html(html_original)
    page_perturbed = parse_html(html_perturbed)

    map_original = build_position_map(page_original)
    map_perturbed = build_position_map(page_perturbed)

    mask: dict[str, bool] = {}
    all_keys = map_original.keys() | map_perturbed.keys()
    for key in all_keys:
        orig_val = map_original.get(key)
        pert_val = map_perturbed.get(key)
        if orig_val is None or pert_val is None:
            # Position present in one render but not the other → variable
            mask[key] = False
        else:
            mask[key] = orig_val == pert_val
    return mask


def _perturb_data(data: dict, _counter: list[int] | None = None) -> dict:
    """Replace every leaf value with a unique sentinel string."""
    if _counter is None:
        _counter = [0]
    result: dict = {}
    for k, v in data.items():
        if isinstance(v, bool):
            result[k] = v  # booleans control conditionals, keep them
        elif isinstance(v, str):
            _counter[0] += 1
            result[k] = f"SENTINEL{_counter[0]}END"
        elif isinstance(v, int):
            _counter[0] += 1
            result[k] = _counter[0] * 97  # unique int
        elif isinstance(v, dict):
            result[k] = _perturb_data(v, _counter)
        elif isinstance(v, list):
            result[k] = [
                _perturb_data(item, _counter) if isinstance(item, dict) else v
                for item in v
            ]
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Value flattening
# ---------------------------------------------------------------------------


def flatten_values(d: dict[str, Any]) -> set[str]:
    """Recursively extract all leaf string/int values from a nested dict.

    Booleans are excluded (they control conditionals, not extracted as text).
    """
    values: set[str] = set()
    for v in d.values():
        if isinstance(v, bool):
            continue
        if isinstance(v, str):
            if v:
                values.add(v)
        elif isinstance(v, int):
            values.add(str(v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    values.update(flatten_values(item))
                elif isinstance(item, str) and item:
                    values.add(item)
        elif isinstance(v, dict):
            values.update(flatten_values(v))
    return values


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Metrics from evaluating an inferred template against ground truth."""

    recognition_rate: float
    discrimination_rate: float
    value_recall: float
    value_precision: float
    mask_recall: float
    mask_precision: float
    n_train: int
    n_test: int
    n_negative: int


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 1.0


# ---------------------------------------------------------------------------
# Alignment-based position mapping (shared by ExAlg + k-Testable v2)
# ---------------------------------------------------------------------------


def build_aligned_position_maps(
    pages: list[etree._Element],
) -> tuple[
    list[dict[str, str]],
    list[str],
    list[tuple[str, int]],
    list[tuple[list[str], int]],
    list[str],
]:
    """Build position maps with stable keys across pages with variable children.

    Uses child alignment to assign backbone children stable indices.
    Positions in repeating/optional regions are excluded (handled separately).

    Returns:
        stable_maps: per-page position maps using backbone-based indices
        backbone_tags: list of backbone child tags
        repeating_regions: list of (tag, after_backbone_pos) for each repeating region
        optional_regions: list of (tags, after_backbone_pos) for each optional region
        loop_vars: list of variable names, one per repeating region
    """
    from westlean.child_alignment import align_children

    children_lists = [list(p) for p in pages]

    # Check if all pages have identical child tag sequences — no alignment needed
    tag_seqs = [tuple(c.tag for c in cl) for cl in children_lists]
    if len(set(tag_seqs)) == 1:
        return [build_position_map(p) for p in pages], [], [], [], []

    alignment = align_children(children_lists)

    if not alignment.backbone and not alignment.repeating and not alignment.optional:
        return [build_position_map(p) for p in pages], [], [], [], []

    repeating_regions: list[tuple[str, int]] = [
        (r.tag, r.after_backbone_pos) for r in alignment.repeating
    ]
    optional_regions: list[tuple[list[str], int]] = [
        (o.tags, o.after_backbone_pos) for o in alignment.optional
    ]
    loop_vars = [f"loop_{i}" for i in range(len(repeating_regions))]

    stable_maps: list[dict[str, str]] = []
    for pi, page in enumerate(pages):
        pos_map: dict[str, str] = {}
        if page.text:
            pos_map["text"] = page.text
        for attr_name in sorted(page.attrib):
            pos_map[f"@{attr_name}"] = page.attrib[attr_name]

        for bi in range(len(alignment.backbone)):
            child_idx = alignment.backbone_indices[pi][bi]
            child = children_lists[pi][child_idx]
            child_prefix = str(bi)
            _walk_positions(child, child_prefix, pos_map)
            if child.tail:
                pos_map[f"{child_prefix}/tail"] = child.tail

        stable_maps.append(pos_map)

    return (
        stable_maps,
        alignment.backbone,
        repeating_regions,
        optional_regions,
        loop_vars,
    )


def match_page_to_backbone(
    page: etree._Element,
    backbone_tags: list[str],
    repeating_regions: list[tuple[str, int]],
    optional_regions: list[tuple[list[str], int]],
) -> tuple[list[int], dict[int, list[int]], dict[int, list[int]]] | None:
    """Match a page's children against a learned backbone pattern.

    Args:
        page: The DOM element whose children to match.
        backbone_tags: Ordered list of backbone child tags.
        repeating_regions: List of (tag, after_backbone_pos) per repeating region.
        optional_regions: List of (tags, after_backbone_pos) per optional region.

    Returns:
        (backbone_indices, repeating_by_region_idx, optional_by_region_idx)
        or None if the page doesn't match.

        repeating_by_region_idx maps region index -> list of child indices.
        optional_by_region_idx maps region index -> list of child indices.
    """
    children = list(page)
    pi = 0
    backbone_indices: list[int] = []
    repeating_by_region: dict[int, list[int]] = {}
    optional_by_region: dict[int, list[int]] = {}

    # Index regions by after_backbone_pos for quick lookup
    rep_by_pos: dict[int, list[tuple[int, str]]] = {}
    for ri, (tag, after_pos) in enumerate(repeating_regions):
        rep_by_pos.setdefault(after_pos, []).append((ri, tag))

    opt_by_pos: dict[int, list[tuple[int, list[str]]]] = {}
    for oi, (tags, after_pos) in enumerate(optional_regions):
        opt_by_pos.setdefault(after_pos, []).append((oi, tags))

    def _consume_gap(after_pos: int) -> bool:
        """Consume repeating and optional regions for a given gap position.

        Builds a tag-to-region lookup to match children flexibly regardless
        of the order regions were detected during alignment.  Repeating
        regions consume runs of same-tag children.  Optional regions consume
        single-occurrence children.
        """
        nonlocal pi

        # Build tag -> (type, index) lookup for this gap
        rep_tags: dict[str, int] = {}
        for ri, rtag in rep_by_pos.get(after_pos, []):
            rep_tags[rtag] = ri
            repeating_by_region[ri] = []

        opt_first_tags: dict[str, tuple[int, list[str]]] = {}
        for oi, otags in opt_by_pos.get(after_pos, []):
            optional_by_region[oi] = []
            if otags[0] not in rep_tags:
                opt_first_tags[otags[0]] = (oi, otags)

        # All known tags in this gap (repeating + optional first tags)
        known_tags = set(rep_tags) | set(opt_first_tags)

        # Consume children that match known gap tags
        while pi < len(children) and str(children[pi].tag) in known_tags:
            tag = str(children[pi].tag)
            if tag in rep_tags:
                ri = rep_tags[tag]
                # Greedily consume run of this tag
                while pi < len(children) and str(children[pi].tag) == tag:
                    repeating_by_region[ri].append(pi)
                    pi += 1
            elif tag in opt_first_tags:
                oi, otags = opt_first_tags[tag]
                indices: list[int] = []
                for ot in otags:
                    if pi < len(children) and str(children[pi].tag) == ot:
                        indices.append(pi)
                        pi += 1
                optional_by_region[oi] = indices
                # Don't match this optional again
                del opt_first_tags[tag]
                known_tags = set(rep_tags) | set(opt_first_tags)

        return True

    for bi, bt in enumerate(backbone_tags):
        # Consume gap before this backbone child (after_pos = bi - 1)
        _consume_gap(bi - 1)
        # Match backbone
        if pi >= len(children) or children[pi].tag != bt:
            return None
        backbone_indices.append(pi)
        pi += 1

    # Consume trailing gap (after last backbone child)
    _consume_gap(len(backbone_tags) - 1)

    if pi != len(children):
        return None
    return backbone_indices, repeating_by_region, optional_by_region


def build_backbone_position_map(
    page: etree._Element,
    backbone_indices: list[int],
) -> dict[str, str]:
    """Build a position map using backbone indices for a single page."""
    children = list(page)
    pos_map: dict[str, str] = {}
    if page.text:
        pos_map["text"] = page.text
    for attr_name in sorted(page.attrib):
        pos_map[f"@{attr_name}"] = page.attrib[attr_name]
    for bi, ci in enumerate(backbone_indices):
        child_prefix = str(bi)
        _walk_positions(children[ci], child_prefix, pos_map)
        if children[ci].tail:
            pos_map[f"{child_prefix}/tail"] = children[ci].tail or ""
    return pos_map


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


def evaluate(
    inferer: TemplateInferer,
    template: Element,
    schema: DataSchema,
    train_data: list[dict],
    test_data: list[dict],
    negative_pages: list[etree._Element],
) -> EvaluationResult:
    """Run a full evaluation of *inferer* against known ground truth.

    1. Renders training pages from *template* + *train_data*.
    2. Calls ``inferer.infer()`` on the training pages.
    3. Evaluates ``extract()`` and ``fixed_mask()`` on test pages and negatives.
    """
    train_pages = [parse_html(render(template, d)) for d in train_data]
    inferred = inferer.infer(train_pages)

    recognized = 0
    vr_num = vr_den = vp_num = vp_den = 0
    mr_num = mr_den = mp_num = mp_den = 0

    for d in test_data:
        page = parse_html(render(template, d))
        extraction = inferred.extract(page)

        if extraction is not None:
            recognized += 1
            truth_vals = flatten_values(d)
            extracted_vals = flatten_values(extraction)
            overlap = truth_vals & extracted_vals
            vr_num += len(overlap)
            vr_den += len(truth_vals)
            vp_num += len(overlap)
            vp_den += len(extracted_vals)

        inf_mask = inferred.fixed_mask(page)
        truth_mask = ground_truth_mask(template, d)

        if inf_mask is not None and truth_mask:
            for key in truth_mask:
                if key in inf_mask:
                    truly_var = not truth_mask[key]
                    labeled_var = not inf_mask[key]
                    if truly_var:
                        mr_den += 1
                        if labeled_var:
                            mr_num += 1
                    if labeled_var:
                        mp_den += 1
                        if truly_var:
                            mp_num += 1

    rejected = sum(1 for p in negative_pages if inferred.extract(p) is None)

    return EvaluationResult(
        recognition_rate=_safe_ratio(recognized, len(test_data)),
        discrimination_rate=_safe_ratio(rejected, len(negative_pages)),
        value_recall=_safe_ratio(vr_num, vr_den),
        value_precision=_safe_ratio(vp_num, vp_den),
        mask_recall=_safe_ratio(mr_num, mr_den),
        mask_precision=_safe_ratio(mp_num, mp_den),
        n_train=len(train_data),
        n_test=len(test_data),
        n_negative=len(negative_pages),
    )
