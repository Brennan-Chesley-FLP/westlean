"""Tracing variant of the EXALG inferer.

Emits :class:`~westlean.tracer.TraceStep` events at each algorithm phase
so intermediate state can be visualized in the documentation site.

Phases correspond to the paper's ECGM sub-modules and ConstTemp:

1. ``tokenization`` -- linearize pages into token streams
2. ``equivalence_classes`` -- FindEq: compute occurrence vectors, classify keys
3. ``diffformat`` -- DiffFormat: context refinement for sibling disambiguation
4. ``handinv`` -- HandInv: demote skeleton tokens nested inside loop bodies
5. ``diffeq`` -- DiffEq: promote first instances of loop elements with fixed values
6. ``skeleton`` -- skeleton/gap extraction summary
7. ``result`` -- final inferred template
"""

from __future__ import annotations

from typing import Any, Sequence

from lxml import etree

from westlean.tracer import get_tracer
from westlean.protocol import EmptyTemplate
from westlean.algorithms.exalg import (
    ExalgTemplate,
    _Counter,
    _build_template,
    _classify_structural_keys,
    _is_value_token,
    _linearize,
    _refine_contexts,
    _structural_key_vector,
)


def _sk_display(sk: tuple[str, str, str, str]) -> str:
    """Format a structural key tuple for display."""
    kind, tag, attr_name, context = sk
    parts = [kind]
    if tag:
        parts.append(tag)
    if attr_name:
        parts.append(f"@{attr_name}")
    if context:
        parts.append(f"[{context}]")
    return " ".join(parts)


class TracingExalgInferer:
    """EXALG inferer that emits trace steps when a :class:`Tracer` is active.

    Each ECGM sub-module (DiffFormat, FindEq, HandInv, DiffEq) gets its
    own trace step, plus tokenization, skeleton extraction, and result.
    """

    def infer(self, pages: Sequence[etree._Element]) -> ExalgTemplate | EmptyTemplate:
        tracer = get_tracer()

        if len(pages) < 2:
            return EmptyTemplate()

        # Phase 1: Tokenization (linearization)
        streams = [_linearize(page) for page in pages]

        root_tags = {s[0].tag for s in streams if s}
        if len(root_tags) != 1:
            return EmptyTemplate()

        if tracer:
            tracer.emit(
                "exalg",
                "tokenization",
                "Linearize each page into a token stream with structural keys and tag-path contexts",
                {
                    "page_count": len(pages),
                    "tokens_per_page": [len(s) for s in streams],
                    "sample_tokens": [
                        {
                            "kind": tok.kind,
                            "tag": tok.tag,
                            "attr_name": tok.attr_name,
                            "value": tok.value[:50] if tok.value else "",
                            "position_key": tok.position_key,
                            "context": tok.context,
                        }
                        for tok in streams[0][:30]
                    ]
                    if streams
                    else [],
                },
            )

        # Phase 2: FindEq -- occurrence vectors and equivalence class classification
        n_pages = len(pages)
        vectors = _structural_key_vector(streams)
        template_keys, loop_keys, optional_keys = _classify_structural_keys(
            vectors, n_pages
        )

        if tracer:
            # Group keys by their vector for display
            vector_groups: dict[tuple[int, ...], list[str]] = {}
            for sk, vec in vectors.items():
                vector_groups.setdefault(vec, []).append(_sk_display(sk))

            tracer.emit(
                "exalg",
                "equivalence_classes",
                "Compute occurrence vectors and classify structural keys into template/loop/optional",
                {
                    "total_structural_keys": len(vectors),
                    "template_constants": len(template_keys),
                    "loop_markers": len(loop_keys),
                    "optional_markers": len(optional_keys),
                    "sample_vectors": {
                        _sk_display(k): list(v) for k, v in list(vectors.items())[:20]
                    },
                    "equivalence_class_count": len(vector_groups),
                    "largest_classes": [
                        {
                            "vector": list(vec),
                            "size": len(members),
                            "sample_members": members[:5],
                        }
                        for vec, members in sorted(
                            vector_groups.items(), key=lambda x: -len(x[1])
                        )[:5]
                    ],
                },
            )

        # Phase 3: DiffFormat -- context refinement
        streams_before = streams
        streams = _refine_contexts(streams, template_keys, vectors)

        refined = streams is not streams_before
        if refined:
            vectors_after = _structural_key_vector(streams)
            tk_after, lk_after, ok_after = _classify_structural_keys(
                vectors_after, n_pages
            )
        else:
            vectors_after = vectors
            tk_after, lk_after, ok_after = template_keys, loop_keys, optional_keys

        if tracer:
            # Find which contexts changed
            changed_contexts: list[dict[str, str]] = []
            if refined:
                old_contexts = {tok.context for s in streams_before for tok in s}
                new_contexts = {tok.context for s in streams for tok in s}
                added = new_contexts - old_contexts
                for ctx in sorted(added)[:10]:
                    changed_contexts.append(
                        {"context": ctx, "status": "added (sibling-indexed)"}
                    )

            tracer.emit(
                "exalg",
                "diffformat",
                "Refine contexts for fixed-count sibling disambiguation (Observation 4.4)",
                {
                    "contexts_refined": len(changed_contexts) > 0,
                    "changed_contexts": changed_contexts,
                    "template_constants_before": len(template_keys),
                    "template_constants_after": len(tk_after),
                    "loop_markers_before": len(loop_keys),
                    "loop_markers_after": len(lk_after),
                },
            )

        # Phase 4: HandInv -- demote skeleton tokens nested inside loop bodies
        # Re-run the classification on the refined streams
        template_keys = tk_after
        loop_keys = lk_after
        optional_keys = ok_after
        vectors = vectors_after

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

        if tracer:
            tracer.emit(
                "exalg",
                "handinv",
                "Validate skeleton tokens -- demote those nested inside loop bodies",
                {
                    "demoted_count": len(demoted),
                    "demoted_keys": [
                        _sk_display(sk) for sk in sorted(demoted, key=str)[:15]
                    ],
                    "template_constants_before": len(template_keys),
                    "template_constants_after": len(template_keys) - len(demoted),
                },
            )

        # Phase 5: DiffEq -- promote first instances of loop elements
        template_keys_after_handinv = template_keys - demoted

        # Replicate DiffEq logic to report promotions
        promoted_first: set[tuple[str, str, str, str]] = set()
        loop_by_context: dict[str, list[tuple[str, str, str, str]]] = {}
        for sk in loop_keys:
            loop_by_context.setdefault(sk[3], []).append(sk)

        loop_element_contexts = {sk[3] for sk in loop_keys if sk[0] == "open"}

        promotion_decisions: list[dict[str, Any]] = []

        for context, group in loop_by_context.items():
            nested_in_loop = any(
                context != lc and context.startswith(lc + "/")
                for lc in loop_element_contexts
            )
            if nested_in_loop:
                promotion_decisions.append(
                    {
                        "context": context,
                        "decision": "skip_nested",
                        "reason": "Nested inside another loop element",
                    }
                )
                continue

            open_sk = next((sk for sk in group if sk[0] == "open"), None)
            if open_sk and open_sk in vectors and min(vectors[open_sk]) < 2:
                promotion_decisions.append(
                    {
                        "context": context,
                        "decision": "skip_min_count",
                        "reason": f"Min instance count {min(vectors[open_sk])} < 2",
                        "vector": list(vectors[open_sk]),
                    }
                )
                continue

            value_sks = [sk for sk in group if _is_value_token(sk[0])]
            if not value_sks:
                continue

            for vsk in value_sks:
                first_vals: list[str] = []
                has_diff = False
                for stream in streams:
                    instances = [
                        t for t in stream if (*t.structural_key, t.context) == vsk
                    ]
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
                    promoted_first.update(group)
                    promotion_decisions.append(
                        {
                            "context": context,
                            "decision": "promoted",
                            "reason": f"First instance fixed value '{first_vals[0][:30]}', later instances vary",
                            "group_size": len(group),
                        }
                    )
                    break

        if tracer:
            tracer.emit(
                "exalg",
                "diffeq",
                "Differentiate roles within EQ class spans -- promote fixed first instances (Observation 4.5)",
                {
                    "promoted_count": len(promoted_first),
                    "promoted_keys": [
                        _sk_display(sk) for sk in sorted(promoted_first, key=str)[:15]
                    ],
                    "promotion_decisions": promotion_decisions[:10],
                    "skeleton_size_final": len(template_keys_after_handinv)
                    + len(promoted_first),
                },
            )

        # Phase 6: Skeleton extraction summary
        # We compute the skeleton to report on it, then delegate actual
        # template construction to _build_template (which re-derives everything).
        if tracer:
            skeleton_keys = template_keys_after_handinv
            skel_count = 0
            gap_count = 0
            for stream in streams[:1]:  # just analyze first page for summary
                in_gap = True
                for tok in stream:
                    sk = (*tok.structural_key, tok.context)
                    if sk in skeleton_keys:
                        skel_count += 1
                        if in_gap:
                            gap_count += 1
                        in_gap = True
                    elif sk in promoted_first:
                        # First occurrence is skeleton
                        skel_count += 1
                        if in_gap:
                            gap_count += 1
                        in_gap = True
                    else:
                        in_gap = False
                if not in_gap:
                    gap_count += 1  # trailing gap

            tracer.emit(
                "exalg",
                "skeleton",
                "Extract skeleton tokens and gap regions for ConstTemp analysis",
                {
                    "skeleton_tokens": skel_count,
                    "gap_regions": gap_count,
                    "template_constants": len(skeleton_keys),
                    "promoted_first_instances": len(promoted_first),
                },
            )

        # Delegate to _build_template for actual template construction
        # (it re-runs the ECGM sub-modules internally, which is redundant
        # but keeps the tracing variant purely observational)
        ctr = _Counter()
        elements = _build_template(streams, ctr)

        if not elements:
            if tracer:
                tracer.emit(
                    "exalg",
                    "result",
                    "No template could be constructed",
                    {"empty": True},
                )
            return EmptyTemplate()

        template = ExalgTemplate(elements)

        if tracer:
            tracer.emit(
                "exalg",
                "result",
                "Final inferred template with Literal, Var, Set, and Optional elements",
                template.serialize(),
            )

        return template
