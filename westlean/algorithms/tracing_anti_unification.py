"""Tracing variant of the anti-unification inferer.

Emits :class:`~westlean.tracer.TraceStep` events at each algorithm phase
so intermediate state can be visualized in the documentation site.

Each pairwise fold is split into two or three steps:
- ``compare`` — shows the current template and incoming page side-by-side,
  with mismatching positions highlighted.
- ``region_detection`` (conditional) — emitted when child-count differences
  trigger LCS backbone computation.  Shows the backbone tags, repeating
  regions, and optional regions discovered by the rigid UFOG alignment.
- ``fold`` — shows the merged template with newly created variables.
"""

from __future__ import annotations

from typing import Any, Sequence

from lxml import etree

from westlean.protocol import EmptyTemplate
from westlean.serialization import tpl_node_to_model
from westlean.tracer import get_tracer

from westlean.algorithms.anti_unification import (
    AntiUnifiedTemplate,
    _Counter,
    _TplNode,
    _page_to_template,
    _anti_unify,
)


def _compute_mismatches(
    tpl: _TplNode, page: _TplNode, path: str = ""
) -> list[dict[str, Any]]:
    """Walk template and page trees in parallel, collecting slot mismatches."""
    mismatches: list[dict[str, Any]] = []
    display_path = path or "(root)"

    # Text
    if tpl.text[0]:  # template slot is fixed
        if tpl.text[1] != page.text[1]:
            mismatches.append(
                {
                    "path": display_path,
                    "field": "text",
                    "template_value": tpl.text[1],
                    "page_value": page.text[1],
                }
            )

    # Tail
    if tpl.tail[0]:
        if tpl.tail[1] != page.tail[1]:
            mismatches.append(
                {
                    "path": display_path,
                    "field": "tail",
                    "template_value": tpl.tail[1],
                    "page_value": page.tail[1],
                }
            )

    # Attributes
    for name in tpl.attr_names:
        if name in tpl.attrs and tpl.attrs[name][0]:
            if tpl.attrs[name][1] != page.attrs.get(name, (True, ""))[1]:
                mismatches.append(
                    {
                        "path": display_path,
                        "field": f"@{name}",
                        "template_value": tpl.attrs[name][1],
                        "page_value": page.attrs.get(name, (True, ""))[1],
                    }
                )

    # Children
    if tpl.children is not None and page.children is not None:
        if len(tpl.children) != len(page.children):
            mismatches.append(
                {
                    "path": display_path,
                    "field": "children",
                    "template_value": f"{len(tpl.children)} children",
                    "page_value": f"{len(page.children)} children",
                }
            )
        else:
            for i, (tc, pc) in enumerate(zip(tpl.children, page.children)):
                child_path = f"{path}/{i}" if path else str(i)
                if tc.tag != pc.tag:
                    mismatches.append(
                        {
                            "path": child_path,
                            "field": "tag",
                            "template_value": tc.tag,
                            "page_value": pc.tag,
                        }
                    )
                else:
                    mismatches.extend(_compute_mismatches(tc, pc, child_path))

    return mismatches


def _compute_new_vars(
    before: _TplNode, after: _TplNode, path: str = ""
) -> list[dict[str, Any]]:
    """Find slots that were fixed in *before* but became variable in *after*."""
    new_vars: list[dict[str, Any]] = []
    display_path = path or "(root)"

    if before.text[0] and not after.text[0]:
        new_vars.append(
            {
                "path": display_path,
                "field": "text",
                "var_name": after.text[1],
            }
        )

    if before.tail[0] and not after.tail[0]:
        new_vars.append(
            {
                "path": display_path,
                "field": "tail",
                "var_name": after.tail[1],
            }
        )

    for name in before.attr_names:
        if name in before.attrs and before.attrs[name][0]:
            if name in after.attrs and not after.attrs[name][0]:
                new_vars.append(
                    {
                        "path": display_path,
                        "field": f"@{name}",
                        "var_name": after.attrs[name][1],
                    }
                )

    if before.children is not None and after.children is None:
        new_vars.append(
            {
                "path": display_path,
                "field": "children",
                "var_name": after.children_var or "?",
            }
        )
    elif before.children is not None and after.children is not None:
        for i, (bc, ac) in enumerate(zip(before.children, after.children)):
            child_path = f"{path}/{i}" if path else str(i)
            new_vars.extend(_compute_new_vars(bc, ac, child_path))

    return new_vars


def _detect_region_changes(
    before: _TplNode, after: _TplNode, path: str = ""
) -> list[dict[str, Any]]:
    """Detect where LCS-based region detection was triggered during a fold.

    Returns a list of region-detection events, each describing the backbone
    tags, repeating regions, and optional regions discovered at that node.
    """
    events: list[dict[str, Any]] = []
    display_path = path or "(root)"

    # Region detection happened if the 'after' node gained regions that
    # the 'before' node did not have.
    before_has_regions = bool(before.repeating_regions or before.optional_regions)
    after_has_regions = bool(after.repeating_regions or after.optional_regions)

    if after_has_regions and not before_has_regions:
        # Regions were newly introduced at this node
        backbone_tags = [c.tag for c in (after.children or [])]
        repeating_info = []
        for pos, rtpl, rvar in after.repeating_regions:
            repeating_info.append(
                {
                    "after_backbone_pos": pos,
                    "tag": rtpl.tag,
                    "var_name": rvar,
                }
            )
        optional_info = []
        for pos, opt_children in after.optional_regions:
            optional_info.append(
                {
                    "after_backbone_pos": pos,
                    "tags": [c.tag for c in opt_children],
                }
            )

        # Compute the child counts that triggered detection
        before_child_count = len(before.children) if before.children is not None else 0
        events.append(
            {
                "path": display_path,
                "backbone_tags": backbone_tags,
                "repeating_regions": repeating_info,
                "optional_regions": optional_info,
                "before_child_count": before_child_count,
            }
        )

    # Also check if children region entirely collapsed to a variable
    # (before had children, after does not, and no regions)
    if before.children is not None and after.children is None and not after_has_regions:
        before_child_count = len(before.children) if before.children else 0
        events.append(
            {
                "path": display_path,
                "backbone_tags": [],
                "repeating_regions": [],
                "optional_regions": [],
                "before_child_count": before_child_count,
                "collapsed_to_variable": after.children_var or "?",
            }
        )

    # Recurse into children that exist in both
    if before.children is not None and after.children is not None:
        for i, (bc, ac) in enumerate(zip(before.children, after.children)):
            child_path = f"{path}/{i}" if path else str(i)
            events.extend(_detect_region_changes(bc, ac, child_path))

    return events


def _count_slots_fixed(node: _TplNode) -> int:
    count = 0
    if node.text[0] and node.text[1]:
        count += 1
    if node.tail[0] and node.tail[1]:
        count += 1
    for name in node.attr_names:
        if node.attrs[name][0]:
            count += 1
    if node.children is not None:
        for c in node.children:
            count += _count_slots_fixed(c)
    return count


def _count_slots_var(node: _TplNode) -> int:
    count = 0
    if not node.text[0]:
        count += 1
    if not node.tail[0]:
        count += 1
    for name in node.attr_names:
        if not node.attrs[name][0]:
            count += 1
    if node.children is None and node.children_var:
        count += 1
    elif node.children is not None:
        for c in node.children:
            count += _count_slots_var(c)
    for _, rtpl, _ in node.repeating_regions:
        count += _count_slots_var(rtpl)
    return count


class TracingAntiUnificationInferer:
    """Anti-unification inferer that emits trace steps when a :class:`Tracer` is active."""

    def infer(
        self, pages: Sequence[etree._Element]
    ) -> AntiUnifiedTemplate | EmptyTemplate:
        tracer = get_tracer()

        ctr = _Counter()
        tpl = _page_to_template(pages[0])

        if tracer:
            tracer.emit(
                "anti_unification",
                "initial_template",
                "Convert first page to fully-fixed template"
                f" ({_count_slots_fixed(tpl)} slots, all fixed)",
                {"template": tpl_node_to_model(tpl).model_dump()},
            )

        for i, page in enumerate(pages[1:], start=1):
            page_tpl = _page_to_template(page)

            # --- Compare step ---
            if tracer:
                mismatches = _compute_mismatches(tpl, page_tpl)
                tracer.emit(
                    "anti_unification",
                    "compare",
                    f"Compare template with page {i + 1}"
                    f" ({len(mismatches)} replaceable position"
                    f"{'s' if len(mismatches) != 1 else ''} found)",
                    {
                        "page_index": i,
                        "template_before": tpl_node_to_model(tpl).model_dump(),
                        "incoming_page": tpl_node_to_model(page_tpl).model_dump(),
                        "mismatches": mismatches,
                    },
                )

            tpl_before = tpl
            result = _anti_unify(tpl, page, ctr)

            if result is None:
                if tracer:
                    tracer.emit(
                        "anti_unification",
                        "fold",
                        f"Fold with page {i + 1} failed"
                        " (incompatible root structure — no LGG exists)",
                        {"page_index": i, "template_after": None, "new_vars": []},
                    )
                return EmptyTemplate()

            tpl = result

            # --- Region detection step (emitted only when regions are new) ---
            if tracer:
                region_events = _detect_region_changes(tpl_before, tpl)
                if region_events:
                    # Build a human-readable description
                    parts = []
                    for evt in region_events:
                        bb = evt["backbone_tags"]
                        rr = evt["repeating_regions"]
                        orr = evt["optional_regions"]
                        collapsed = evt.get("collapsed_to_variable")
                        if collapsed:
                            parts.append(
                                f"at {evt['path']}: children collapsed"
                                f" to hedge variable {collapsed}"
                            )
                        else:
                            bb_str = (
                                ", ".join(f"<{t}>" for t in bb) if bb else "(empty)"
                            )
                            parts.append(
                                f"at {evt['path']}: LCS backbone [{bb_str}]"
                                f", {len(rr)} repeating"
                                f", {len(orr)} optional"
                            )
                    tracer.emit(
                        "anti_unification",
                        "region_detection",
                        f"LCS-based child alignment for page {i + 1}: "
                        + "; ".join(parts),
                        {
                            "page_index": i,
                            "regions": region_events,
                        },
                    )

            # --- Fold step ---
            if tracer:
                new_vars = _compute_new_vars(tpl_before, tpl)
                slot_summary = {
                    "fixed": _count_slots_fixed(tpl),
                    "variable": _count_slots_var(tpl),
                }
                tracer.emit(
                    "anti_unification",
                    "fold",
                    f"Fold page {i + 1} into template"
                    + (
                        f" ({len(new_vars)} new variable"
                        f"{'s' if len(new_vars) != 1 else ''}"
                        f" introduced by Plotkin step 4)"
                        if new_vars
                        else " (no new variables — all positions agree)"
                    )
                    + f" [{slot_summary['fixed']} fixed,"
                    f" {slot_summary['variable']} variable]",
                    {
                        "page_index": i,
                        "template_after": tpl_node_to_model(tpl).model_dump(),
                        "new_vars": new_vars,
                        "slot_summary": slot_summary,
                    },
                )

        template = AntiUnifiedTemplate(tpl)

        if tracer:
            final_summary = {
                "fixed": _count_slots_fixed(tpl),
                "variable": _count_slots_var(tpl),
            }
            tracer.emit(
                "anti_unification",
                "result",
                f"Final inferred template: {final_summary['fixed']} fixed"
                f" and {final_summary['variable']} variable positions",
                template.serialize(),
            )

        return template
