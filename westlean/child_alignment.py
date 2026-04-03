"""Shared child-sequence alignment with repeat and optional detection.

Used by tree-based algorithms (FiVaTech, Anti-Unification, ExAlg) to
handle varying child counts caused by loops and conditionals.
"""

from __future__ import annotations

from dataclasses import dataclass

from lxml import etree


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RepeatingRegion:
    """A region where children of the same tag repeat with variable count."""

    tag: str
    after_backbone_pos: int  # backbone index this region follows (-1 = before first)
    counts: list[int]  # count per page


@dataclass
class OptionalRegion:
    """A region present in some pages but not others."""

    tags: list[str]
    after_backbone_pos: int
    present_in: list[bool]  # per-page


@dataclass
class AlignmentResult:
    """Result of aligning children across N pages."""

    backbone: list[str]  # common child tag subsequence
    backbone_indices: list[list[int]]  # per-page: index of each backbone child
    repeating: list[RepeatingRegion]
    optional: list[OptionalRegion]


# ---------------------------------------------------------------------------
# LCS
# ---------------------------------------------------------------------------


def lcs(a: list[str], b: list[str]) -> list[str]:
    """Longest common subsequence of two tag sequences."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
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


def map_to_backbone(seq: list[str], backbone: list[str]) -> list[int]:
    """Map backbone positions to indices in *seq* (greedy left-to-right)."""
    result: list[int] = []
    j = 0
    for tag in backbone:
        while seq[j] != tag:
            j += 1
        result.append(j)
        j += 1
    return result


# ---------------------------------------------------------------------------
# Main alignment
# ---------------------------------------------------------------------------


def _structural_key(elem: etree._Element) -> str:
    """Tag + sorted attribute names.

    Distinguishes e.g. ``<a>`` from ``<a class='x'>``.
    Child tags are deliberately excluded: when a child element contains
    a loop, its child count varies across pages, giving different keys
    for what is structurally the same element.
    """
    attrs = tuple(sorted(elem.attrib))
    tag = str(elem.tag)
    parts = [tag]
    if attrs:
        parts.append(f"[{','.join(attrs)}]")
    return "".join(parts)


def align_children(
    pages_children: list[list[etree._Element]],
) -> AlignmentResult:
    """Align children across N pages, detecting repeats and optionals.

    Uses structural keys (tag + attribute names) for alignment so that
    ``<p>`` and ``<p class="x">`` are treated as different elements.

    1. Compute progressive LCS backbone of structural key sequences.
    2. Map each page's children to backbone positions.
    3. Classify gaps between backbone positions:
       - Same-tag children with variable count (max > 1) → RepeatingRegion
       - Same-tag children present in some pages (max == 1) → OptionalRegion
    """
    n_pages = len(pages_children)
    key_sequences = [[_structural_key(c) for c in cl] for cl in pages_children]

    # Pre-filter: exclude structural keys with variable counts across pages
    from collections import Counter

    key_counts: dict[str, list[int]] = {}
    all_keys_seen = set(k for seq in key_sequences for k in seq)
    for seq in key_sequences:
        seq_counts = Counter(seq)
        for key in all_keys_seen:
            key_counts.setdefault(key, []).append(seq_counts.get(key, 0))
    variable_keys = {key for key, cnts in key_counts.items() if len(set(cnts)) > 1}

    # Compute LCS backbone using only stable structural keys
    stable_sequences = [
        [k for k in seq if k not in variable_keys] for seq in key_sequences
    ]
    backbone_keys = stable_sequences[0]
    for seq in stable_sequences[1:]:
        backbone_keys = lcs(backbone_keys, seq)

    # Value-aware fallback: when the backbone is empty and a single
    # variable-count key is present in every page (min >= 1), check
    # whether the first instance has identical values across all pages.
    # If so, that instance is a fixed element cohabiting with loop items
    # of the same tag — promote it to backbone.
    if not backbone_keys:
        candidates = {
            key
            for key, cnts in key_counts.items()
            if len(set(cnts)) > 1 and min(cnts) >= 1
        }
        if len(candidates) == 1:
            ckey = next(iter(candidates))
            # Collect the first instance from each page
            first_elems = []
            for pi, (seq, cl) in enumerate(zip(key_sequences, pages_children)):
                for j, k in enumerate(seq):
                    if k == ckey:
                        first_elems.append(cl[j])
                        break
            if len(first_elems) == n_pages:
                # Compare values: text + attribute values
                def _elem_values(el: etree._Element) -> tuple:
                    return (
                        el.text or "",
                        tuple((k, el.attrib[k]) for k in sorted(el.attrib)),
                    )

                first_vals = [_elem_values(e) for e in first_elems]
                if len(set(first_vals)) == 1:
                    # First instance has same value across pages.
                    # Also verify other instances have DIFFERENT values —
                    # this distinguishes "fixed + loop" from "coincidentally
                    # same first items in a pure loop."
                    first_val = first_vals[0]
                    has_different = False
                    for pi, (seq, cl) in enumerate(zip(key_sequences, pages_children)):
                        for j, k in enumerate(seq):
                            if k == ckey and cl[j] is not first_elems[pi]:
                                if _elem_values(cl[j]) != first_val:
                                    has_different = True
                                    break
                        if has_different:
                            break
                    if has_different:
                        backbone_keys = [ckey]
                        variable_keys = variable_keys - candidates

    # Extract plain tags for the result (strip attribute and child info)
    def _plain_tag(key: str) -> str:
        for sep in ("[", "("):
            idx = key.find(sep)
            if idx != -1:
                key = key[:idx]
        return key

    backbone_tags = [_plain_tag(k) for k in backbone_keys]

    # Map each page's children to backbone (skipping variable-key children)
    backbone_indices: list[list[int]] = []
    for pi, seq in enumerate(key_sequences):
        indices: list[int] = []
        j = 0
        for bk in backbone_keys:
            while j < len(seq) and (seq[j] != bk or seq[j] in variable_keys):
                j += 1
            if j < len(seq):
                indices.append(j)
                j += 1
        backbone_indices.append(indices)

    # Analyze gaps between backbone positions
    repeating: list[RepeatingRegion] = []
    optional: list[OptionalRegion] = []

    for gap_idx in range(len(backbone_keys) + 1):
        gap_children: list[list[etree._Element]] = []
        for pi in range(n_pages):
            if gap_idx == 0:
                start = 0
            else:
                start = backbone_indices[pi][gap_idx - 1] + 1

            if gap_idx == len(backbone_keys):
                end = len(pages_children[pi])
            else:
                end = backbone_indices[pi][gap_idx]

            gap_children.append(pages_children[pi][start:end])

        counts = [len(gc) for gc in gap_children]
        if all(c == 0 for c in counts):
            continue

        after_pos = gap_idx - 1
        all_tags: set[str] = set()
        for gc in gap_children:
            for c in gc:
                all_tags.add(str(c.tag))

        if len(all_tags) == 1:
            tag = all_tags.pop()
            if max(counts) > 1:
                repeating.append(
                    RepeatingRegion(
                        tag=tag,
                        after_backbone_pos=after_pos,
                        counts=counts,
                    )
                )
            elif any(c == 0 for c in counts) and max(counts) == 1:
                optional.append(
                    OptionalRegion(
                        tags=[tag],
                        after_backbone_pos=after_pos,
                        present_in=[c > 0 for c in counts],
                    )
                )
        elif len(all_tags) > 1:
            non_empty_seqs = [
                tuple(str(c.tag) for c in gc) for gc in gap_children if len(gc) > 0
            ]
            if len(set(non_empty_seqs)) == 1 and any(c == 0 for c in counts):
                optional.append(
                    OptionalRegion(
                        tags=list(non_empty_seqs[0]),
                        after_backbone_pos=after_pos,
                        present_in=[c > 0 for c in counts],
                    )
                )
            else:
                # Try to decompose the gap into per-tag groups.
                # Each gap child sequence should be a concatenation of
                # same-tag runs whose tags appear in a consistent order.
                # Collect per-tag counts for each page.
                from collections import Counter as _Counter

                tag_order: list[str] = []
                for gc in gap_children:
                    for c in gc:
                        ctag = str(c.tag)
                        if ctag not in tag_order:
                            tag_order.append(ctag)
                per_tag_counts: dict[str, list[int]] = {t: [] for t in tag_order}
                valid = True
                for gc in gap_children:
                    tc = _Counter(str(c.tag) for c in gc)
                    for t in tag_order:
                        per_tag_counts[t].append(tc.get(t, 0))
                    # Verify ordering: tags must appear in consistent order
                    seen_order: list[str] = []
                    for c in gc:
                        ctag = str(c.tag)
                        if not seen_order or seen_order[-1] != ctag:
                            seen_order.append(ctag)
                    # Check that seen_order is a subsequence of tag_order
                    ti = 0
                    for st in seen_order:
                        while ti < len(tag_order) and tag_order[ti] != st:
                            ti += 1
                        if ti >= len(tag_order):
                            valid = False
                            break
                        ti += 1
                    if not valid:
                        break

                if valid:
                    for t in tag_order:
                        cnts = per_tag_counts[t]
                        if max(cnts) > 1:
                            repeating.append(
                                RepeatingRegion(
                                    tag=t,
                                    after_backbone_pos=after_pos,
                                    counts=cnts,
                                )
                            )
                        elif any(c == 0 for c in cnts) and max(cnts) == 1:
                            optional.append(
                                OptionalRegion(
                                    tags=[t],
                                    after_backbone_pos=after_pos,
                                    present_in=[c > 0 for c in cnts],
                                )
                            )

    return AlignmentResult(
        backbone=backbone_tags,
        backbone_indices=backbone_indices,
        repeating=repeating,
        optional=optional,
    )
