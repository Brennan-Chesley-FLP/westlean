"""Tracing variant of the FiVaTech inferer.

Emits :class:`~westlean.tracer.TraceStep` events at each algorithm phase
so intermediate state can be visualized in the documentation site.

Phases emitted:

1. ``structure_check`` -- root tag and attribute name agreement
2. ``child_sequences`` -- per-page child tag sequences
3. ``backbone_computation`` -- progressive LCS backbone
4. ``alignment`` -- per-page child-to-backbone mapping
5. ``gap_analysis`` -- repeating and optional region detection from gaps
6. ``recursive_merge`` -- the merged pattern tree (fixed/variant at every position)
7. ``classification`` -- summary of fixed vs variable positions
8. ``result`` -- final serialized template
"""

from __future__ import annotations

from typing import Any, Sequence

from lxml import etree

from westlean.tracer import get_tracer
from westlean.protocol import EmptyTemplate
from westlean.serialization import tpl_node_to_model
from westlean.child_alignment import align_children, _structural_key
from westlean.algorithms.fivatech import (
    FiVaTechTemplate,
    _Counter,
    _PatternNode,
    _merge_trees,
    _lcs,
    _map_to_backbone,
)


def _count_slots(node: _PatternNode) -> dict[str, int]:
    """Count fixed and variable slots in a pattern tree."""
    fixed = 0
    variable = 0

    def _walk(n: _PatternNode) -> None:
        nonlocal fixed, variable
        # text
        if n.text[0]:
            if n.text[1]:  # non-empty fixed text
                fixed += 1
        else:
            variable += 1
        # tail
        if n.tail[0]:
            if n.tail[1]:
                fixed += 1
        else:
            variable += 1
        # attrs
        for _, slot in n.attrs.items():
            if slot[0]:
                fixed += 1
            else:
                variable += 1
        # children variable
        if n.children is None and n.children_var:
            variable += 1
        # recurse
        if n.children:
            for c in n.children:
                _walk(c)
        for _, tpl, _ in n.repeating_regions:
            _walk(tpl)
        for _, children in n.optional_regions:
            for c in children:
                _walk(c)

    _walk(node)
    return {"fixed": fixed, "variable": variable}


def _collect_classification(
    node: _PatternNode, prefix: str = ""
) -> list[dict[str, Any]]:
    """Collect per-position fixed/variable classification for tracing."""
    rows: list[dict[str, Any]] = []
    path = prefix or "(root)"

    # text
    is_fixed, val = node.text
    if val or not is_fixed:
        rows.append(
            {
                "path": f"{path}/text",
                "type": "text",
                "classification": "fixed" if is_fixed else "variable",
                "value": val if is_fixed else None,
                "var_name": val if not is_fixed else None,
            }
        )

    # tail
    is_fixed, val = node.tail
    if val or not is_fixed:
        rows.append(
            {
                "path": f"{path}/tail",
                "type": "tail",
                "classification": "fixed" if is_fixed else "variable",
                "value": val if is_fixed else None,
                "var_name": val if not is_fixed else None,
            }
        )

    # attrs
    for attr_name in node.attr_names:
        is_fixed, val = node.attrs[attr_name]
        rows.append(
            {
                "path": f"{path}/@{attr_name}",
                "type": "attribute",
                "classification": "fixed" if is_fixed else "variable",
                "value": val if is_fixed else None,
                "var_name": val if not is_fixed else None,
            }
        )

    # children
    if node.children is None and node.children_var:
        rows.append(
            {
                "path": f"{path}/children",
                "type": "children",
                "classification": "variable",
                "value": None,
                "var_name": node.children_var,
            }
        )
    elif node.children:
        for i, child in enumerate(node.children):
            child_prefix = f"{prefix}/{i}" if prefix else str(i)
            rows.extend(_collect_classification(child, child_prefix))

    # repeating regions
    for pos, tpl, var in node.repeating_regions:
        rows.append(
            {
                "path": f"{path}/repeat[after={pos}]",
                "type": "repeating_region",
                "classification": "variable",
                "value": None,
                "var_name": var,
                "repeat_tag": tpl.tag,
            }
        )

    # optional regions
    for pos, children in node.optional_regions:
        tags = [c.tag for c in children]
        rows.append(
            {
                "path": f"{path}/optional[after={pos}]",
                "type": "optional_region",
                "classification": "variable",
                "value": None,
                "var_name": None,
                "optional_tags": tags,
            }
        )

    return rows


class TracingFiVaTechInferer:
    """FiVaTech inferer that emits trace steps when a :class:`Tracer` is active."""

    def infer(
        self, pages: Sequence[etree._Element]
    ) -> FiVaTechTemplate | EmptyTemplate:
        tracer = get_tracer()
        page_list = list(pages)

        # Phase 1: Structure check -- tags and attr names must agree
        tags = {p.tag for p in page_list}
        attr_name_sets = {tuple(sorted(p.attrib)) for p in page_list}
        tag = next(iter(tags)) if len(tags) == 1 else None
        attr_names = (
            list(next(iter(attr_name_sets))) if len(attr_name_sets) == 1 else None
        )
        all_match = len(tags) == 1 and len(attr_name_sets) == 1

        if tracer:
            tracer.emit(
                "fivatech",
                "structure_check",
                "Check that root tags and attribute names agree across all pages"
                " (paper Section 3: peer node recognition at the root level)",
                {
                    "tag": tag,
                    "attr_names": attr_names,
                    "all_match": all_match,
                    "page_count": len(page_list),
                },
            )

        if not all_match:
            return EmptyTemplate()

        # Phase 2: Child sequences with structural keys
        children_lists = [list(p) for p in page_list]
        tag_sequences = [[str(c.tag) for c in cl] for cl in children_lists]
        key_sequences = [[_structural_key(c) for c in cl] for cl in children_lists]

        if tracer:
            tracer.emit(
                "fivatech",
                "child_sequences",
                "Extract child tag sequences from each page."
                " Structural keys (tag + attribute names) distinguish elements"
                ' like <p> from <p class="x">',
                {
                    "sequences": tag_sequences,
                    "structural_keys": key_sequences,
                },
            )

        # Phase 3: Backbone computation (progressive LCS)
        backbone = tag_sequences[0] if tag_sequences else []
        lcs_steps: list[dict[str, Any]] = []
        for i, seq in enumerate(tag_sequences[1:], 1):
            prev_backbone = list(backbone)
            backbone = _lcs(backbone, seq)
            lcs_steps.append(
                {
                    "step": i,
                    "with_page": i + 1,
                    "input_backbone": prev_backbone,
                    "page_sequence": seq,
                    "result_backbone": list(backbone),
                }
            )

        if tracer:
            tracer.emit(
                "fivatech",
                "backbone_computation",
                "Compute progressive LCS backbone across all child sequences."
                " This is our replacement for the paper's peer matrix alignment"
                " (Section 3.2)",
                {
                    "backbone": backbone,
                    "sequence_count": len(tag_sequences),
                    "lcs_steps": lcs_steps,
                },
            )

        # Phase 4: Alignment -- map each page's children to backbone
        if backbone:
            alignments = [_map_to_backbone(ts, backbone) for ts in tag_sequences]
        else:
            alignments = [[] for _ in tag_sequences]

        if tracer:
            tracer.emit(
                "fivatech",
                "alignment",
                "Align each page's children to the backbone via greedy"
                " left-to-right matching. Non-aligned children fall into gaps",
                {
                    "backbone": backbone,
                    "mappings": alignments,
                },
            )

        # Phase 5: Gap analysis -- run align_children to get repeating/optional
        alignment_result = align_children(children_lists)
        if tracer:
            repeating_info = [
                {
                    "tag": r.tag,
                    "after_backbone_pos": r.after_backbone_pos,
                    "counts_per_page": r.counts,
                }
                for r in alignment_result.repeating
            ]
            optional_info = [
                {
                    "tags": o.tags,
                    "after_backbone_pos": o.after_backbone_pos,
                    "present_in": o.present_in,
                }
                for o in alignment_result.optional
            ]
            tracer.emit(
                "fivatech",
                "gap_analysis",
                "Classify gaps between backbone positions: repeating regions"
                " (paper Section 3.3: repetitive pattern mining) and optional"
                " regions (paper Section 3.4: optional node merging)",
                {
                    "backbone": alignment_result.backbone,
                    "repeating_regions": repeating_info,
                    "optional_regions": optional_info,
                    "has_repeating": len(repeating_info) > 0,
                    "has_optional": len(optional_info) > 0,
                },
            )

        # Now run the actual merge (which does all of the above internally
        # plus the recursive descent).
        ctr = _Counter()
        result = _merge_trees(page_list, ctr)

        # Phase 6: Recursive merge result
        if tracer:
            if result is not None:
                model = tpl_node_to_model(result)
                tracer.emit(
                    "fivatech",
                    "recursive_merge",
                    "Result of recursive simultaneous tree merge with"
                    " fixed/variant labeling at every position",
                    {
                        "pattern_tree": model.model_dump(),
                    },
                )
            else:
                tracer.emit(
                    "fivatech",
                    "recursive_merge",
                    "Merge failed -- structurally incompatible pages",
                    {},
                )

        # Phase 7: Classification summary
        if tracer and result is not None:
            slot_counts = _count_slots(result)
            positions = _collect_classification(result)
            tracer.emit(
                "fivatech",
                "classification",
                "Fixed/variant classification of every position"
                " (paper Section 3: the core output of the pattern tree)",
                {
                    "fixed_count": slot_counts["fixed"],
                    "variable_count": slot_counts["variable"],
                    "positions": positions,
                },
            )

        # Phase 8: Final result
        if result is None:
            template: FiVaTechTemplate | EmptyTemplate = EmptyTemplate()
        else:
            template = FiVaTechTemplate(result)

        if tracer:
            tracer.emit(
                "fivatech",
                "result",
                "Final inferred template",
                template.serialize(),
            )

        return template
