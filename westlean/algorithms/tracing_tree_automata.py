"""Tracing variant of the k-testable tree automata inferer.

Emits :class:`~westlean.tracer.TraceStep` events at each algorithm phase
so intermediate state can be visualized in the documentation site.

Phases emitted:

1. ``structure_analysis`` — Root tag check, child tag sequences per page
2. ``child_sequences`` — Child tag sequences (consumed by SequenceAlignment viz)
3. ``backbone_computation`` — LCS backbone result (consumed by SequenceAlignment viz)
4. ``alignment`` — Per-page backbone index mappings (consumed by SequenceAlignment viz)
5. ``template_tree`` — Recursive UTA template tree summary
6. ``generalization`` — Repeating/optional region detection and relaxation
7. ``result`` — Final serialized template
"""

from __future__ import annotations

from typing import Any, Sequence

from lxml import etree

from westlean.tracer import get_tracer
from westlean.protocol import EmptyTemplate
from westlean.child_alignment import align_children
from westlean.algorithms.tree_automata import (
    KTestableTemplate,
    _Counter,
    _build_uta_tree,
)


class TracingKTestableInferer:
    """k-Testable inferer that emits trace steps when a :class:`Tracer` is active."""

    def __init__(self, k: int = 2) -> None:
        if k not in (2, 3):
            raise ValueError(f"k must be 2 or 3, got {k}")
        self._k = k

    def infer(
        self, pages: Sequence[etree._Element]
    ) -> KTestableTemplate | EmptyTemplate:
        tracer = get_tracer()

        if len(pages) < 2:
            return EmptyTemplate()

        page_list = list(pages)

        # Phase 1: Structure analysis
        tags = {p.tag for p in page_list}
        attr_name_sets = {tuple(sorted(p.attrib)) for p in page_list}
        children_per_page = [[child.tag for child in page] for page in page_list]

        if tracer:
            tracer.emit(
                "tree_automata",
                "structure_analysis",
                "Verify root tag/attribute agreement and collect child sequences",
                {
                    "k": self._k,
                    "page_count": len(page_list),
                    "root_tags": list(tags),
                    "root_tag_match": len(tags) == 1,
                    "attr_names_match": len(attr_name_sets) == 1,
                    "children_per_page": children_per_page,
                },
            )

        if len(tags) != 1 or len(attr_name_sets) != 1:
            if tracer:
                tracer.emit(
                    "tree_automata",
                    "result",
                    "Pages structurally incompatible -- empty template",
                    {"empty": True},
                )
            return EmptyTemplate()

        # Phase 2: Child sequences (for SequenceAlignment viz)
        if tracer:
            tracer.emit(
                "tree_automata",
                "child_sequences",
                "Child tag sequences from each page (input to LCS alignment)",
                {
                    "sequences": children_per_page,
                },
            )

        # Phase 3: Backbone computation via LCS alignment
        # Run the shared child_alignment module on the root's children to
        # surface the backbone for visualization, before the full recursive build.
        children_lists = [list(p) for p in page_list]
        alignment = align_children(children_lists)

        if tracer:
            tracer.emit(
                "tree_automata",
                "backbone_computation",
                (
                    f"LCS backbone: {len(alignment.backbone)} common children; "
                    f"{len(alignment.repeating)} repeating region(s), "
                    f"{len(alignment.optional)} optional region(s)"
                ),
                {
                    "backbone": alignment.backbone,
                    "sequence_count": len(children_per_page),
                },
            )

        # Phase 4: Alignment mappings (for SequenceAlignment viz)
        if tracer:
            tracer.emit(
                "tree_automata",
                "alignment",
                "Map each page's children to backbone positions",
                {
                    "backbone": alignment.backbone,
                    "mappings": alignment.backbone_indices,
                },
            )

        # Phase 5: Build UTA template tree (the full recursive build)
        ctr = _Counter()
        root = _build_uta_tree(page_list, ctr)

        if root is None:
            if tracer:
                tracer.emit(
                    "tree_automata",
                    "result",
                    "Pages structurally incompatible -- empty template",
                    {"empty": True},
                )
            return EmptyTemplate()

        if tracer:

            def _summarize_node(node: Any, depth: int = 0) -> dict:
                """Summarize a _UTANode for tracing."""
                summary: dict[str, Any] = {
                    "tag": node.tag,
                    "text_fixed": node.text[0],
                    "tail_fixed": node.tail[0],
                    "attrs_fixed": {k: v[0] for k, v in node.attrs.items()},
                }
                if node.text[0]:
                    summary["text_value"] = node.text[1]
                if not node.text[0]:
                    summary["text_var"] = node.text[1]
                if node.children is not None:
                    summary["backbone_count"] = len(node.children)
                    if depth < 3:
                        summary["backbone"] = [
                            _summarize_node(c, depth + 1) for c in node.children
                        ]
                else:
                    summary["children_variable"] = True
                    summary["children_var_name"] = node.children_var
                if node.repeating_regions:
                    summary["repeating_regions"] = [
                        {
                            "tag": tpl.tag,
                            "after_backbone_pos": pos,
                            "var_name": var,
                        }
                        for pos, tpl, var in node.repeating_regions
                    ]
                if node.optional_regions:
                    summary["optional_regions"] = [
                        {
                            "tags": [c.tag for c in children],
                            "after_backbone_pos": pos,
                        }
                        for pos, children in node.optional_regions
                    ]
                return summary

            tracer.emit(
                "tree_automata",
                "template_tree",
                "Built recursive UTA template tree from aligned training pages",
                _summarize_node(root),
            )

        # Phase 6: Generalization summary
        if tracer:
            n_repeating = 0
            n_optional = 0
            n_variable_text = 0
            n_fixed_text = 0
            n_variable_attrs = 0
            n_fixed_attrs = 0

            def _count_slots(node: Any) -> None:
                nonlocal n_repeating, n_optional
                nonlocal n_variable_text, n_fixed_text
                nonlocal n_variable_attrs, n_fixed_attrs
                is_fixed, _ = node.text
                if is_fixed:
                    n_fixed_text += 1
                else:
                    n_variable_text += 1
                for _, slot in node.attrs.items():
                    if slot[0]:
                        n_fixed_attrs += 1
                    else:
                        n_variable_attrs += 1
                n_repeating += len(node.repeating_regions)
                n_optional += len(node.optional_regions)
                if node.children:
                    for child in node.children:
                        _count_slots(child)
                for _, tpl, _unused_var in node.repeating_regions:
                    _count_slots(tpl)
                for _, children in node.optional_regions:
                    for c in children:
                        _count_slots(c)

            _count_slots(root)

            tracer.emit(
                "tree_automata",
                "generalization",
                (
                    "Generalization: repeating regions relax always_present flags; "
                    "optional regions use skeleton-only matching"
                ),
                {
                    "repeating_regions_total": n_repeating,
                    "optional_regions_total": n_optional,
                    "fixed_text_slots": n_fixed_text,
                    "variable_text_slots": n_variable_text,
                    "fixed_attr_slots": n_fixed_attrs,
                    "variable_attr_slots": n_variable_attrs,
                    "generalization_notes": [
                        note
                        for note in [
                            (
                                f"{n_repeating} repeating region(s): "
                                "always_present flags cleared in loop body templates "
                                "(analogous to g-testable wildcard relaxation)"
                            )
                            if n_repeating > 0
                            else None,
                            (
                                f"{n_optional} optional region(s): "
                                "extraction uses skeleton-only matching (check_values=False), "
                                "allowing unseen text values in optional children"
                            )
                            if n_optional > 0
                            else None,
                            "No repeating or optional regions detected"
                            if n_repeating == 0 and n_optional == 0
                            else None,
                        ]
                        if note is not None
                    ],
                },
            )

        # Phase 7: Result
        template = KTestableTemplate(root, self._k)

        if tracer:
            tracer.emit(
                "tree_automata",
                "result",
                "Final inferred template",
                template.serialize(),
            )

        return template
