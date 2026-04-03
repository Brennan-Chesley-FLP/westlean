"""Tracing variant of the RoadRunner inferer.

Emits :class:`~westlean.tracer.TraceStep` events at each algorithm phase
so intermediate state can be visualized in the documentation site.

Phases emitted:
  - ``linearization`` -- token stream from first page
  - ``ufre_init`` -- initial all-Literal UFRE
  - ``acme_fold`` -- UFRE after each pairwise ACME generalization, with
    mismatch summary (counts of Vars, Optionals, Repeats introduced)
  - ``repeat_detection`` -- changes from the post-fold _detect_repeats pass
    that converts Optionals with repeating content into Repeat elements
  - ``result`` -- final inferred template
"""

from __future__ import annotations

from typing import Sequence

from lxml import etree

from westlean.tracer import get_tracer
from westlean.protocol import EmptyTemplate
from westlean.serialization import ufre_to_model
from westlean.algorithms.roadrunner import (
    RoadRunnerTemplate,
    _Counter,
    Literal,
    Var,
    Optional,
    Repeat,
    UFREElement,
    linearize,
    _tokens_to_ufre,
    _acme,
    _detect_repeats,
)


def _ufre_summary(ufre: list[UFREElement]) -> dict[str, int]:
    """Count element types in a UFRE (flat, non-recursive)."""
    counts: dict[str, int] = {"literal": 0, "var": 0, "optional": 0, "repeat": 0}
    for elem in ufre:
        if isinstance(elem, Literal):
            counts["literal"] += 1
        elif isinstance(elem, Var):
            counts["var"] += 1
        elif isinstance(elem, Optional):
            counts["optional"] += 1
        elif isinstance(elem, Repeat):
            counts["repeat"] += 1
    return counts


def _diff_summary(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int]:
    """Compute difference in element counts (after - before)."""
    return {k: after.get(k, 0) - before.get(k, 0) for k in before}


class TracingRoadRunnerInferer:
    """RoadRunner inferer that emits trace steps when a :class:`Tracer` is active."""

    def infer(
        self, pages: Sequence[etree._Element]
    ) -> RoadRunnerTemplate | EmptyTemplate:
        tracer = get_tracer()

        ctr = _Counter()

        # Phase 1: Linearization
        tokens = linearize(pages[0])

        if tracer:
            tracer.emit(
                "roadrunner",
                "linearization",
                "Linearize first page into token stream",
                {
                    "tokens": [
                        {
                            "kind": t.kind,
                            "tag": t.tag,
                            "attr_name": t.attr_name,
                            "value": t.value,
                            "position_key": t.position_key,
                        }
                        for t in tokens
                    ],
                },
            )

        # Phase 2: UFRE initialization
        wrapper = _tokens_to_ufre(tokens)

        if tracer:
            element_count = len(wrapper)
            literal_count = sum(1 for e in wrapper if isinstance(e, Literal))
            tracer.emit(
                "roadrunner",
                "ufre_init",
                "Initialize UFRE from first page (all Literals)",
                {
                    "element_count": element_count,
                    "literal_count": literal_count,
                },
            )

        # Phase 3: ACME fold over remaining pages
        for page_index, page in enumerate(pages[1:], start=1):
            page_tokens = linearize(page)
            before_summary = _ufre_summary(wrapper)

            result = _acme(wrapper, page_tokens, ctr)

            if result is None:
                if tracer:
                    tracer.emit(
                        "roadrunner",
                        "acme_fold",
                        f"ACME failed on page {page_index} (incompatible structure)",
                        {
                            "page_index": page_index,
                            "success": False,
                            "ufre_after": None,
                        },
                    )
                return EmptyTemplate()

            after_summary = _ufre_summary(result)
            diff = _diff_summary(before_summary, after_summary)

            # Build a human-readable description of what changed
            changes: list[str] = []
            if diff["var"] > 0:
                changes.append(f"{diff['var']} string mismatch(es) \u2192 Var")
            if diff["optional"] > 0:
                changes.append(f"{diff['optional']} tag mismatch(es) \u2192 Optional")
            if diff["repeat"] > 0:
                changes.append(f"{diff['repeat']} tag mismatch(es) \u2192 Repeat")
            change_text = "; ".join(changes) if changes else "no new generalizations"

            if tracer:
                tracer.emit(
                    "roadrunner",
                    "acme_fold",
                    f"ACME fold with page {page_index}: {change_text}",
                    {
                        "page_index": page_index,
                        "success": True,
                        "before_counts": before_summary,
                        "after_counts": after_summary,
                        "diff": diff,
                        "ufre_after": [
                            elem.model_dump() for elem in ufre_to_model(result)
                        ],
                    },
                )

            # Post-fold: _detect_repeats converts Optionals to Repeats
            pre_detect = _ufre_summary(result)
            wrapper = _detect_repeats(result, ctr)
            post_detect = _ufre_summary(wrapper)
            detect_diff = _diff_summary(pre_detect, post_detect)

            promoted = -detect_diff.get("optional", 0)  # optionals removed
            repeats_added = detect_diff.get("repeat", 0)

            if tracer and (promoted > 0 or repeats_added > 0):
                tracer.emit(
                    "roadrunner",
                    "repeat_detection",
                    f"Post-fold repeat detection: {promoted} Optional(s) \u2192 {repeats_added} Repeat(s)",
                    {
                        "page_index": page_index,
                        "optionals_converted": promoted,
                        "repeats_created": repeats_added,
                        "before_counts": pre_detect,
                        "after_counts": post_detect,
                        "ufre_after": [
                            elem.model_dump() for elem in ufre_to_model(wrapper)
                        ],
                    },
                )

        # Phase 4: Result
        template = RoadRunnerTemplate(wrapper)

        if tracer:
            final_summary = _ufre_summary(wrapper)
            tracer.emit(
                "roadrunner",
                "result",
                "Final inferred template",
                {
                    **template.serialize(),
                    "summary": final_summary,
                },
            )

        return template
