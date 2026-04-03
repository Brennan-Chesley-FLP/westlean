"""RoadRunner grammar inference (Crescenzi, Mecca & Merialdo, VLDB 2001).

Implements the ACME algorithm with backtracking for pairwise generalization
of linearized DOM token streams into a Union-Free Regular Expression (UFRE).

At each tag mismatch the algorithm first hypothesises an iterator (repeating
pattern), using terminal-tag search and backward square verification per the
original paper. Only when no iterator candidate succeeds does it fall back
to an optional hypothesis via cross-search resynchronisation.

Iterator discoveries at the external level are treated as fixpoints (never
backtracked), matching the paper's pruning heuristic.  Backtracking is
localised to trying multiple candidate squares within a single mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from lxml import etree

from westlean.protocol import EmptyTemplate


# ---------------------------------------------------------------------------
# Token representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """A token in the linearized DOM stream."""

    kind: str  # "open", "close", "text", "tail", "attr"
    tag: str  # tag name (open/close); "" otherwise
    attr_name: str  # attribute name (attr); "" otherwise
    value: str  # text/attribute value content
    position_key: str  # canonical position key for DOM mapping

    @property
    def structural_key(self) -> tuple[str, str, str]:
        return (self.kind, self.tag, self.attr_name)


# ---------------------------------------------------------------------------
# UFRE elements
# ---------------------------------------------------------------------------


@dataclass
class Literal:
    """A fixed token that must match exactly."""

    token: Token


@dataclass
class Var:
    """A variable capturing varying content at a position."""

    name: str
    kind: str
    tag: str
    attr_name: str
    position_key: str
    always_has_value: bool = True  # was value non-empty in ALL training pages?

    @property
    def structural_key(self) -> tuple[str, str, str]:
        return (self.kind, self.tag, self.attr_name)


@dataclass
class Optional:
    """Elements that may or may not be present."""

    elements: list  # list[UFREElement]


@dataclass
class Repeat:
    """Elements that repeat variable times (detected loop)."""

    elements: list  # list[UFREElement] — the repeating unit
    var_name: str  # variable name for extracted list


UFREElement = Literal | Var | Optional | Repeat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Counter:
    """Sequential variable name generator: v_0, v_1, ..."""

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> str:
        name = f"v_{self._n}"
        self._n += 1
        return name


def _is_value_token(kind: str) -> bool:
    """Whether this token kind carries extractable data."""
    return kind in ("text", "tail", "attr")


def _structural_key_of(elem: UFREElement) -> tuple[str, str, str] | None:
    """Get structural key from a UFRE element, or None for Optional/Repeat."""
    if isinstance(elem, Literal):
        return elem.token.structural_key
    if isinstance(elem, Var):
        return elem.structural_key
    return None


# ---------------------------------------------------------------------------
# Linearize DOM → token stream
# ---------------------------------------------------------------------------


def linearize(root: etree._Element, prefix: str = "") -> list[Token]:
    """Convert an lxml element tree to a flat token stream."""
    tokens: list[Token] = []
    _linearize_elem(root, prefix, tokens)
    return tokens


def _linearize_elem(elem: etree._Element, prefix: str, out: list[Token]) -> None:
    tag = str(elem.tag)
    out.append(Token("open", tag, "", "", prefix))

    for attr_name in sorted(elem.attrib):
        key = f"{prefix}/@{attr_name}" if prefix else f"@{attr_name}"
        out.append(Token("attr", "", attr_name, elem.attrib[attr_name], key))

    text_key = f"{prefix}/text" if prefix else "text"
    out.append(Token("text", "", "", elem.text or "", text_key))

    for i, child in enumerate(elem):
        child_prefix = f"{prefix}/{i}" if prefix else str(i)
        _linearize_elem(child, child_prefix, out)
        out.append(Token("tail", "", "", child.tail or "", f"{child_prefix}/tail"))

    out.append(Token("close", tag, "", "", prefix))


# ---------------------------------------------------------------------------
# ACME: pairwise generalization with backtracking
# ---------------------------------------------------------------------------

_MAX_LOOK = 50  # max search distance for iterator / optional candidates
_MAX_CANDIDATES = 6  # max candidate squares to try per hypothesis


def _tokens_to_ufre(tokens: list[Token]) -> list[UFREElement]:
    return [Literal(t) for t in tokens]


def _structural_match(a: list[UFREElement], b: list[UFREElement]) -> bool:
    """Check if two UFRE element lists have the same structural pattern."""
    if len(a) != len(b):
        return False
    for ae, be in zip(a, b):
        ak = _structural_key_of(ae)
        bk = _structural_key_of(be)
        if ak is None or bk is None:
            return False
        if ak != bk:
            return False
    return True


def _deep_structural_match(a: list[UFREElement], b: list[UFREElement]) -> bool:
    """Structural match that handles Repeat and Optional elements recursively."""
    if len(a) != len(b):
        return False
    for ae, be in zip(a, b):
        if type(ae) is not type(be):
            return False
        if isinstance(ae, Repeat):
            assert isinstance(be, Repeat)
            if not _deep_structural_match(ae.elements, be.elements):
                return False
        elif isinstance(ae, Optional):
            assert isinstance(be, Optional)
            if not _deep_structural_match(ae.elements, be.elements):
                return False
        else:
            ak = _structural_key_of(ae)
            bk = _structural_key_of(be)
            if ak is None or bk is None or ak != bk:
                return False
    return True


def _matches_template_block(
    template: list[UFREElement],
    flat: list[UFREElement],
) -> bool:
    """Check if *flat* UFRE elements match *template* that may contain Repeats.

    Repeats in the template consume one or more structurally-matching runs
    from the flat list.  Used by ``_detect_repeats`` to identify outer loops
    whose body contains inner Repeats.
    """
    ti = 0
    fi = 0
    while ti < len(template) and fi < len(flat):
        t_elem = template[ti]
        if isinstance(t_elem, Repeat):
            consumed_any = False
            while fi < len(flat):
                n = _flat_match_len(t_elem.elements, flat, fi)
                if n == 0:
                    break
                fi += n
                consumed_any = True
            if not consumed_any:
                return False
            ti += 1
        elif isinstance(t_elem, Optional):
            n = _flat_match_len(t_elem.elements, flat, fi)
            if n > 0:
                fi += n
            ti += 1
        else:
            if fi >= len(flat):
                return False
            tk = _structural_key_of(t_elem)
            fk = _structural_key_of(flat[fi])
            if tk is None or fk is None or tk != fk:
                return False
            ti += 1
            fi += 1
    return ti == len(template) and fi == len(flat)


def _flat_match_len(
    body: list[UFREElement],
    flat: list[UFREElement],
    start: int,
) -> int:
    """Match flat UFRE elements against a Repeat body. Return consumed or 0."""
    pos = start
    for elem in body:
        if pos >= len(flat):
            return 0
        ek = _structural_key_of(elem)
        fk = _structural_key_of(flat[pos])
        if ek is None or fk is None or ek != fk:
            return 0
        pos += 1
    return pos - start


def _acme(
    wrapper: list[UFREElement],
    page_tokens: list[Token],
    ctr: _Counter,
) -> list[UFREElement] | None:
    """Generalize *wrapper* against a new page's token stream.

    Uses the ACME algorithm from the RoadRunner paper:

    1. Walk both streams in lockstep.
    2. On string mismatch (same structure, different value) → create Var.
    3. On tag mismatch → try iterator hypothesis **first** (terminal-tag
       search + backward square verification), then fall back to optional
       (cross-search resynchronisation).

    Returns the generalized UFRE, or ``None`` if structures are incompatible.
    """
    result: list[UFREElement] = []
    wi = 0
    ti = 0

    while wi < len(wrapper) and ti < len(page_tokens):
        w_elem = wrapper[wi]
        t_tok = page_tokens[ti]

        # --- Existing Repeat in wrapper: consume matching iterations ---
        if isinstance(w_elem, Repeat):
            new_repeat, ti = _consume_repeat(w_elem, page_tokens, ti, ctr)
            result.append(new_repeat)
            wi += 1
            continue

        # --- Existing Optional in wrapper: try match, else skip ---
        if isinstance(w_elem, Optional):
            consumed = _try_consume_optional(w_elem.elements, page_tokens, ti)
            result.append(w_elem)
            if consumed > 0:
                ti += consumed
            wi += 1
            continue

        # --- Literal or Var: compare structural keys ---
        w_key = _structural_key_of(w_elem)
        t_key = t_tok.structural_key

        if w_key == t_key:
            result.append(_generalize_match(w_elem, t_tok, ctr))
            wi += 1
            ti += 1
            continue

        # --- Tag mismatch ---

        # Check if a recent Optional can be promoted to Repeat.
        # Handles the case where a previous fold created Optional (from a
        # 0-vs-N mismatch) and the current page has more matching items.
        promote = _try_promote_optional(result, page_tokens, ti, ctr)
        if promote is not None:
            result, ti = promote
            continue

        # Try iterator hypothesis (terminal-tag search + verification)
        iter_result = _try_resolve_iterator(
            wrapper,
            wi,
            page_tokens,
            ti,
            ctr,
            result,
        )
        if iter_result is not None:
            result, wi, ti = iter_result
            continue

        # Fall back to optional (cross-search resynchronisation)
        opt_result = _try_resolve_optional(
            wrapper,
            wi,
            page_tokens,
            ti,
            ctr,
        )
        if opt_result is not None:
            opt_elems, new_wi, new_ti = opt_result
            result.extend(opt_elems)
            wi = new_wi
            ti = new_ti
            continue

        # Both hypotheses failed — incompatible structures
        return None

    # Trailing wrapper elements → optional
    if wi < len(wrapper):
        result.append(Optional(_vars_from_wrapper(wrapper[wi:], ctr)))
    # Trailing sample tokens → optional
    if ti < len(page_tokens):
        result.append(Optional(_vars_from_tokens(page_tokens[ti:], ctr)))

    return result


# ---------------------------------------------------------------------------
# Value generalisation (string mismatch)
# ---------------------------------------------------------------------------


def _generalize_match(
    w_elem: UFREElement,
    t_tok: Token,
    ctr: _Counter,
) -> UFREElement:
    """Generalize a structurally-matching wrapper element against a sample token."""
    if isinstance(w_elem, Literal):
        if _is_value_token(w_elem.token.kind):
            if w_elem.token.value != t_tok.value:
                return Var(
                    ctr.next(),
                    w_elem.token.kind,
                    w_elem.token.tag,
                    w_elem.token.attr_name,
                    w_elem.token.position_key,
                    always_has_value=bool(w_elem.token.value) and bool(t_tok.value),
                )
        return w_elem
    if isinstance(w_elem, Var):
        if w_elem.always_has_value and not t_tok.value:
            return Var(
                w_elem.name,
                w_elem.kind,
                w_elem.tag,
                w_elem.attr_name,
                w_elem.position_key,
                always_has_value=False,
            )
        return w_elem
    return w_elem  # pragma: no cover


# ---------------------------------------------------------------------------
# Consuming existing Repeat / Optional during ACME
# ---------------------------------------------------------------------------


def _consume_repeat(
    repeat: Repeat,
    tokens: list[Token],
    ti: int,
    ctr: _Counter,
) -> tuple[Repeat, int]:
    """Consume iterations of an existing Repeat, generalising the body."""
    body = list(repeat.elements)
    new_ti = ti
    while new_ti < len(tokens):
        match = _try_match_and_generalize_body(body, tokens, new_ti, ctr)
        if match is None:
            break
        body, consumed = match
        if consumed == 0:
            break
        new_ti += consumed
    return Repeat(body, repeat.var_name), new_ti


def _try_match_and_generalize_body(
    body: list[UFREElement],
    tokens: list[Token],
    start: int,
    ctr: _Counter,
) -> tuple[list[UFREElement], int] | None:
    """Match one iteration of *body* against *tokens[start:]*, generalising
    Literals to Vars where values differ.

    Returns ``(new_body, tokens_consumed)`` or ``None``.
    """
    pos = start
    new_body: list[UFREElement] = []
    for elem in body:
        if isinstance(elem, Repeat):
            # Nested repeat: consume matching iterations from tokens
            inner_body = list(elem.elements)
            while pos < len(tokens):
                match = _try_match_and_generalize_body(inner_body, tokens, pos, ctr)
                if match is None:
                    break
                inner_body, consumed = match
                if consumed == 0:
                    break
                pos += consumed
            new_body.append(Repeat(inner_body, elem.var_name))
        elif isinstance(elem, Optional):
            consumed = _try_consume_optional(elem.elements, tokens, pos)
            new_body.append(elem)
            pos += consumed
        else:
            if pos >= len(tokens):
                return None
            tok = tokens[pos]
            key = _structural_key_of(elem)
            if key is None or key != tok.structural_key:
                return None
            new_body.append(_generalize_match(elem, tok, ctr))
            pos += 1
    return new_body, pos - start


def _try_consume_optional(
    elements: list[UFREElement],
    tokens: list[Token],
    ti: int,
) -> int:
    """Try to match Optional content against tokens. Returns consumed (0 = skip)."""
    pos = ti
    for elem in elements:
        if pos >= len(tokens):
            return 0
        tok = tokens[pos]
        if isinstance(elem, Literal):
            if tok.structural_key != elem.token.structural_key:
                return 0
            if _is_value_token(elem.token.kind) and elem.token.value != tok.value:
                return 0
        elif isinstance(elem, Var):
            if tok.structural_key != elem.structural_key:
                return 0
        else:
            return 0
        pos += 1
    return pos - ti


# ---------------------------------------------------------------------------
# Iterator hypothesis (try first at every tag mismatch)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Optional → Repeat promotion
# ---------------------------------------------------------------------------


def _try_promote_optional(
    result: list[UFREElement],
    tokens: list[Token],
    ti: int,
    ctr: _Counter,
) -> tuple[list[UFREElement], int] | None:
    """Promote a recent Optional to Repeat if the sample has more matching items.

    When a previous fold created Optional([body]) from a 0-vs-N mismatch,
    and the current fold's sample has additional items matching the body,
    convert the Optional to Repeat and greedily consume all iterations.

    Returns ``(new_result, new_ti)`` or ``None``.
    """
    # Search backward for a recent Optional (within a few elements)
    for j in range(len(result) - 1, max(len(result) - 6, -1), -1):
        if not isinstance(result[j], Optional):
            continue
        opt_elem = result[j]
        assert isinstance(opt_elem, Optional)
        body = opt_elem.elements
        if not body or any(isinstance(e, (Optional, Repeat)) for e in body):
            continue
        # Does the sample at ti match the Optional body?
        match = _try_match_and_generalize_body(body, tokens, ti, ctr)
        if match is None:
            continue
        # Check that elements between j+1 and end of result are all
        # fixed (Literal/Var) and DON'T match the body — otherwise we'd
        # be absorbing trailing fixed content into the Repeat.
        trailing = result[j + 1 :]
        if trailing and _structural_match(trailing[: len(body)], body):
            continue  # trailing content looks like the body — ambiguous
        # Promote: consume all matching iterations from sample
        current_body, consumed = match
        new_ti = ti + consumed
        while new_ti < len(tokens):
            m = _try_match_and_generalize_body(current_body, tokens, new_ti, ctr)
            if m is None:
                break
            current_body, consumed = m
            new_ti += consumed
        repeat = Repeat(current_body, ctr.next())
        new_result = result[:j] + [repeat] + result[j + 1 :]
        return new_result, new_ti
    return None


# ---------------------------------------------------------------------------
# Iterator hypothesis (try first at every tag mismatch)
# ---------------------------------------------------------------------------


def _try_resolve_iterator(
    wrapper: list[UFREElement],
    wi: int,
    tokens: list[Token],
    ti: int,
    ctr: _Counter,
    result: list[UFREElement],
) -> tuple[list[UFREElement], int, int] | None:
    """Try to resolve a tag mismatch as an iterator (repeating pattern).

    Uses terminal-tag search and backward square verification per the paper.
    Returns ``(new_result, new_wi, new_ti)`` or ``None``.
    """
    # Need a terminal key from the last matched element
    if not result:
        return None
    terminal_key = _structural_key_of(result[-1])
    if terminal_key is None:
        return None

    # === Case 1: Extra iteration(s) in sample ===
    sample_candidates: list[tuple[int, int]] = []
    for j in range(ti, min(ti + _MAX_LOOK, len(tokens))):
        if tokens[j].structural_key == terminal_key:
            sq_len = j - ti + 1
            if len(result) >= sq_len:
                sample_candidates.append((j, sq_len))
                if len(sample_candidates) >= _MAX_CANDIDATES:
                    break

    for end_pos, sq_len in sample_candidates:
        saved_n = ctr._n
        prev_elems = result[-sq_len:]

        # Only flat elements (no nested Optional / Repeat)
        if any(isinstance(e, (Optional, Repeat)) for e in prev_elems):
            continue

        body = _verify_square(prev_elems, tokens[ti : end_pos + 1], ctr)
        if body is None:
            ctr._n = saved_n
            continue

        # Count preceding matching iterations in result
        n_extra = _count_preceding_iterations(result, sq_len, body)
        total_pop = (1 + n_extra) * sq_len

        # Consume additional iterations from sample, generalising body
        current_body = list(body)
        new_ti = end_pos + 1
        additional = 0
        while new_ti < len(tokens):
            m = _try_match_and_generalize_body(current_body, tokens, new_ti, ctr)
            if m is None:
                break
            current_body, consumed = m
            new_ti += consumed
            additional += 1

        total_iterations = 1 + n_extra + 1 + additional

        # Peek-ahead: when evidence is minimal (exactly 2 iterations, no
        # extra consumed from sample), verify the wrapper can continue after
        # the Repeat.  Prevents false Repeats from two identical fixed
        # elements (e.g. br, input, br → false Repeat(br)).
        if total_iterations == 2 and wi < len(wrapper):
            w_next_key = _structural_key_of(wrapper[wi])
            if w_next_key is not None and new_ti < len(tokens):
                if tokens[new_ti].structural_key != w_next_key:
                    ctr._n = saved_n
                    continue

        var_name = ctr.next()
        repeat = Repeat(current_body, var_name)
        new_result = result[:-total_pop] + [repeat]
        return (new_result, wi, new_ti)

    # === Case 2: Extra iteration(s) in wrapper ===
    wrapper_candidates: list[tuple[int, int]] = []
    for k in range(wi, min(wi + _MAX_LOOK, len(wrapper))):
        wk = _structural_key_of(wrapper[k])
        if wk == terminal_key:
            sq_len = k - wi + 1
            if ti >= sq_len:
                wrapper_candidates.append((k, sq_len))
                if len(wrapper_candidates) >= _MAX_CANDIDATES:
                    break

    for end_pos, sq_len in wrapper_candidates:
        saved_n = ctr._n
        wrapper_elems = wrapper[wi : end_pos + 1]

        if any(isinstance(e, (Optional, Repeat)) for e in wrapper_elems):
            continue

        prev_tokens = tokens[ti - sq_len : ti]
        body = _verify_square(wrapper_elems, prev_tokens, ctr)
        if body is None:
            ctr._n = saved_n
            continue

        # Pop the shared iteration that was already in result
        n_pop = 0
        if len(result) >= sq_len:
            tail_elems = result[-sq_len:]
            if not any(
                isinstance(e, (Optional, Repeat)) for e in tail_elems
            ) and _structural_match(tail_elems, body):
                n_extra = _count_preceding_iterations(result, sq_len, body)
                n_pop = (1 + n_extra) * sq_len

        var_name = ctr.next()
        repeat = Repeat(body, var_name)
        new_result = (result[:-n_pop] if n_pop > 0 else list(result)) + [repeat]

        # Skip additional matching iterations in wrapper
        new_wi = end_pos + 1
        while new_wi + sq_len <= len(wrapper):
            chunk = wrapper[new_wi : new_wi + sq_len]
            if any(isinstance(e, (Optional, Repeat)) for e in chunk):
                break
            if _structural_match(chunk, body):
                new_wi += sq_len
            else:
                break

        return (new_result, new_wi, ti)

    return None


def _verify_square(
    elems: list[UFREElement],
    toks: list[Token],
    ctr: _Counter,
) -> list[UFREElement] | None:
    """Backward square verification: check *elems* and *toks* are structurally
    compatible and return the generalised body, or ``None``."""
    if len(elems) != len(toks):
        return None
    body: list[UFREElement] = []
    for elem, tok in zip(elems, toks):
        key = _structural_key_of(elem)
        if key is None or key != tok.structural_key:
            return None
        body.append(_generalize_match(elem, tok, ctr))
    return body


def _count_preceding_iterations(
    result: list[UFREElement],
    sq_len: int,
    body: list[UFREElement],
) -> int:
    """Count consecutive iterations matching *body* that precede the known
    iteration at the tail of *result*."""
    count = 0
    pos = len(result) - sq_len
    while pos >= sq_len:
        pos -= sq_len
        candidate = result[pos : pos + sq_len]
        if any(isinstance(e, (Optional, Repeat)) for e in candidate):
            break
        if _structural_match(candidate, body):
            count += 1
        else:
            break
    return count


# ---------------------------------------------------------------------------
# Optional hypothesis (fallback when iterator fails)
# ---------------------------------------------------------------------------


def _try_resolve_optional(
    wrapper: list[UFREElement],
    wi: int,
    tokens: list[Token],
    ti: int,
    ctr: _Counter,
) -> tuple[list[UFREElement], int, int] | None:
    """Try to resolve a tag mismatch as an optional (one side has extra content).

    Cross-search: look for the wrapper's mismatch key in the sample and the
    sample's mismatch key in the wrapper.  Prefer tag-level (open/close)
    matches over generic value tokens (text/tail).

    Returns ``(optional_elements, new_wi, new_ti)`` or ``None``.
    """
    w_key = _structural_key_of(wrapper[wi])
    t_key = tokens[ti].structural_key

    # (source, sample_skip, wrapper_skip, priority)
    candidates: list[tuple[str, int, int, int]] = []

    # Search for w_key in tokens[ti+1:]  →  sample-side optional
    for j in range(ti + 1, min(ti + _MAX_LOOK, len(tokens))):
        if tokens[j].structural_key == w_key:
            skip = j - ti
            kind = w_key[0]
            priority = 0 if kind in ("open", "close") else 1
            candidates.append(("sample", skip, 0, priority))
            break

    # Search for t_key in wrapper[wi+1:]  →  wrapper-side optional
    for k in range(wi + 1, min(wi + _MAX_LOOK, len(wrapper))):
        wk = _structural_key_of(wrapper[k])
        if wk is not None and wk == t_key:
            skip = k - wi
            kind = t_key[0]
            priority = 0 if kind in ("open", "close") else 1
            candidates.append(("wrapper", 0, skip, priority))
            break

    if not candidates:
        return None

    candidates.sort(key=lambda c: (c[3], c[1] + c[2]))

    for source, sample_skip, wrapper_skip, _ in candidates:
        if source == "sample":
            opt = _vars_from_tokens(tokens[ti : ti + sample_skip], ctr)
            return ([Optional(opt)], wi, ti + sample_skip)
        else:
            opt = _vars_from_wrapper(wrapper[wi : wi + wrapper_skip], ctr)
            return ([Optional(opt)], wi + wrapper_skip, ti)

    return None  # pragma: no cover


# ---------------------------------------------------------------------------
# Helpers for building Optional content
# ---------------------------------------------------------------------------


def _vars_from_tokens(toks: list[Token], ctr: _Counter) -> list[UFREElement]:
    """Convert sample tokens to UFRE elements, with value tokens as Vars."""
    result: list[UFREElement] = []
    for t in toks:
        if _is_value_token(t.kind):
            result.append(Var(ctr.next(), t.kind, t.tag, t.attr_name, t.position_key))
        else:
            result.append(Literal(t))
    return result


def _vars_from_wrapper(
    elems: list[UFREElement],
    ctr: _Counter,
) -> list[UFREElement]:
    """Convert wrapper elements for Optional use — value Literals → Vars."""
    result: list[UFREElement] = []
    for e in elems:
        if isinstance(e, Literal) and _is_value_token(e.token.kind):
            result.append(
                Var(
                    ctr.next(),
                    e.token.kind,
                    e.token.tag,
                    e.token.attr_name,
                    e.token.position_key,
                )
            )
        else:
            result.append(e)
    return result


# ---------------------------------------------------------------------------
# Post-fold repeat detection
# ---------------------------------------------------------------------------
#
# When one page has 0 loop iterations and another has N, ACME creates an
# Optional for the N items rather than a Repeat (because there is no
# "previous iteration" to verify against during terminal-tag search).
# This post-processing pass converts such Optionals to Repeats by:
#   1. Detecting internal repetition within an Optional.
#   2. Absorbing an Optional into structurally-matching preceding elements.
#   3. Merging adjacent structurally-matching Optionals.


def _detect_repeats(ufre: list[UFREElement], ctr: _Counter) -> list[UFREElement]:
    """Convert Optional elements to Repeat where a repeating pattern exists."""
    result = list(ufre)
    changed = True
    while changed:
        changed = False
        for i in range(len(result)):
            if not isinstance(result[i], Optional):
                continue
            opt_i = result[i]
            assert isinstance(opt_i, Optional)
            unit = opt_i.elements
            unit_len = len(unit)
            if unit_len == 0:
                continue
            is_flat = not any(isinstance(e, (Optional, Repeat)) for e in unit)

            # --- Case 1: Optional matches preceding elements ---
            if is_flat:
                count = 0
                pos = i - unit_len
                while pos >= 0:
                    preceding = result[pos : pos + unit_len]
                    if any(isinstance(e, (Optional, Repeat)) for e in preceding):
                        break
                    if _structural_match(preceding, unit):
                        count += 1
                        pos -= unit_len
                    else:
                        break

                if count > 0:
                    start = i - count * unit_len
                    template = result[start : start + unit_len]
                    repeat = Repeat(template, ctr.next())
                    result = result[:start] + [repeat] + result[i + 1 :]
                    changed = True
                    break

            # --- Case 1b: flat Optional matches preceding block with Repeats ---
            # When inner loops created Repeat elements in preceding blocks,
            # the flat Optional body won't length-match.  Try matching the
            # Optional body against a preceding block using Repeat-aware
            # template matching.
            if is_flat:
                for block_len in range(1, i + 1):
                    preceding = result[i - block_len : i]
                    if not any(isinstance(e, Repeat) for e in preceding):
                        continue
                    if not _matches_template_block(preceding, unit):
                        continue
                    # Found a match.  Count additional matching blocks before it.
                    count = 1
                    pos = i - block_len
                    while pos >= block_len:
                        prev = result[pos - block_len : pos]
                        if _deep_structural_match(prev, preceding):
                            count += 1
                            pos -= block_len
                        else:
                            break
                    start = i - count * block_len
                    template = result[start : start + block_len]
                    repeat = Repeat(template, ctr.next())
                    result = result[:start] + [repeat] + result[i + 1 :]
                    changed = True
                    break
                if changed:
                    break

            # --- Case 2: internal repetition within the Optional ---
            if is_flat:
                for sub_len in range(1, unit_len // 2 + 1):
                    if unit_len % sub_len != 0:
                        continue
                    sub_unit = unit[:sub_len]
                    if any(isinstance(e, (Optional, Repeat)) for e in sub_unit):
                        continue
                    if all(
                        _structural_match(unit[j : j + sub_len], sub_unit)
                        for j in range(sub_len, unit_len, sub_len)
                    ):
                        repeat = Repeat(list(sub_unit), ctr.next())
                        result = result[:i] + [repeat] + result[i + 1 :]
                        changed = True
                        break
                if changed:
                    break

            # --- Case 3: adjacent Optionals with matching structure ---
            if is_flat and i + 1 < len(result) and isinstance(result[i + 1], Optional):
                next_opt = result[i + 1]
                assert isinstance(next_opt, Optional)
                next_elems = next_opt.elements
                if next_elems and not any(
                    isinstance(e, (Optional, Repeat)) for e in next_elems
                ):
                    if _structural_match(unit, next_elems):
                        repeat = Repeat(list(unit), ctr.next())
                        result = result[:i] + [repeat] + result[i + 2 :]
                        changed = True
                        break
                    for sl in range(1, max(unit_len, len(next_elems)) + 1):
                        if unit_len % sl != 0 or len(next_elems) % sl != 0:
                            continue
                        su = unit[:sl]
                        if any(isinstance(e, (Optional, Repeat)) for e in su):
                            continue
                        a_ok = all(
                            _structural_match(unit[j : j + sl], su)
                            for j in range(sl, unit_len, sl)
                        )
                        b_ok = all(
                            _structural_match(next_elems[j : j + sl], su)
                            for j in range(0, len(next_elems), sl)
                        )
                        if a_ok and b_ok:
                            repeat = Repeat(list(su), ctr.next())
                            result = result[:i] + [repeat] + result[i + 2 :]
                            changed = True
                            break
                    if changed:
                        break

    # Recurse into Repeat bodies to detect inner nested repeats
    for i, elem in enumerate(result):
        if isinstance(elem, Repeat):
            new_body = _detect_repeats(elem.elements, ctr)
            if new_body is not elem.elements:
                result[i] = Repeat(new_body, elem.var_name)

    return result


# ---------------------------------------------------------------------------
# Token-level unit matching (used by extract / fixed_mask)
# ---------------------------------------------------------------------------


def _try_match_unit_tokens(
    unit: list[UFREElement],
    tokens: list[Token],
    start: int,
) -> int:
    """Try to match one iteration of a repeat unit. Return tokens consumed (0 if no match)."""
    ti = start
    for elem in unit:
        if ti >= len(tokens):
            return 0
        tok = tokens[ti]
        if isinstance(elem, Literal):
            if tok.structural_key != elem.token.structural_key:
                return 0
            if _is_value_token(elem.token.kind) and elem.token.value != tok.value:
                return 0
            ti += 1
        elif isinstance(elem, Var):
            if tok.structural_key != elem.structural_key:
                return 0
            ti += 1
        else:
            return 0
    return ti - start


# ---------------------------------------------------------------------------
# Template (public API)
# ---------------------------------------------------------------------------
# RELAX NG generation from UFRE
# ---------------------------------------------------------------------------

_RNG_NS = "http://relaxng.org/ns/structure/1.0"


def _ufre_to_relax_ng(ufre: list[UFREElement]) -> str:
    """Convert a UFRE element list to a RELAX NG schema string."""
    from xml.etree.ElementTree import Element as XE, SubElement, tostring

    grammar = XE("grammar", xmlns=_RNG_NS)
    grammar.set("datatypeLibrary", "http://www.w3.org/2001/XMLSchema-datatypes")
    start = SubElement(grammar, "start")
    _rng_from_ufre(ufre, 0, start)
    return tostring(grammar, encoding="unicode", xml_declaration=True)


def _rng_from_ufre(
    elements: list[UFREElement],
    pos: int,
    parent: Any,
) -> int:
    """Walk the flat UFRE list starting at *pos*, appending RELAX NG
    nodes to *parent*.  Returns the index past the last consumed element."""
    from xml.etree.ElementTree import SubElement

    while pos < len(elements):
        elem = elements[pos]

        if isinstance(elem, Literal):
            tok = elem.token
            if tok.kind == "open":
                el = SubElement(parent, "element", name=tok.tag)
                pos += 1
                pos = _rng_from_ufre(elements, pos, el)
                if len(el) == 0:
                    SubElement(el, "empty")
                continue
            elif tok.kind == "close":
                return pos + 1
            elif tok.kind in ("text", "tail"):
                if tok.value:
                    SubElement(parent, "text")
                pos += 1
                continue
            elif tok.kind == "attr":
                attr_el = SubElement(parent, "attribute", name=tok.attr_name)
                SubElement(attr_el, "text")
                pos += 1
                continue

        elif isinstance(elem, Var):
            if elem.kind in ("text", "tail"):
                SubElement(parent, "text")
            elif elem.kind == "attr":
                attr_el = SubElement(parent, "attribute", name=elem.attr_name)
                SubElement(attr_el, "text")
            pos += 1
            continue

        elif isinstance(elem, Repeat):
            zm = SubElement(parent, "zeroOrMore")
            _rng_from_ufre(elem.elements, 0, zm)
            pos += 1
            continue

        elif isinstance(elem, Optional):
            opt = SubElement(parent, "optional")
            _rng_from_ufre(elem.elements, 0, opt)
            pos += 1
            continue

        pos += 1

    return pos


# ---------------------------------------------------------------------------


class RoadRunnerTemplate:
    """Template inferred by the RoadRunner algorithm."""

    def __init__(self, ufre: list[UFREElement]) -> None:
        self._ufre = ufre

    def serialize(self) -> dict[str, Any]:
        from westlean.serialization import RoadRunnerTemplateModel, ufre_to_model

        model = RoadRunnerTemplateModel(ufre=ufre_to_model(self._ufre))
        return model.model_dump()

    def get_relax_ng(self) -> str:
        return _ufre_to_relax_ng(self._ufre)

    @classmethod
    def restore(cls, data: dict[str, Any]) -> RoadRunnerTemplate:
        from westlean.serialization import RoadRunnerTemplateModel

        model = RoadRunnerTemplateModel.model_validate(data)
        return model.to_internal()

    # -- extract ----------------------------------------------------------

    def extract(self, page: etree._Element) -> dict[str, Any] | None:
        tokens = linearize(page)
        result: dict[str, Any] = {}
        ti = self._match_extract(tokens, result)
        if ti is None or ti != len(tokens):
            return None
        return result

    def _match_extract(
        self,
        tokens: list[Token],
        out: dict[str, Any],
    ) -> int | None:
        ti = 0
        for elem in self._ufre:
            if isinstance(elem, Literal):
                if ti >= len(tokens):
                    return None
                tok = tokens[ti]
                if tok.structural_key != elem.token.structural_key:
                    return None
                if _is_value_token(elem.token.kind) and elem.token.value != tok.value:
                    return None
                ti += 1

            elif isinstance(elem, Var):
                if ti >= len(tokens):
                    return None
                tok = tokens[ti]
                if tok.structural_key != elem.structural_key:
                    return None
                if elem.always_has_value and not tok.value:
                    return None
                if tok.value:
                    out[elem.name] = tok.value
                ti += 1

            elif isinstance(elem, Repeat):
                items: list[dict[str, Any]] = []
                while ti < len(tokens):
                    consumed, item = self._try_repeat_extract(elem.elements, tokens, ti)
                    if consumed == 0:
                        break
                    items.append(item)
                    ti += consumed
                out[elem.var_name] = items

            elif isinstance(elem, Optional):
                ti += self._try_optional_extract(elem.elements, tokens, ti, out)

        return ti

    def _try_repeat_extract(
        self,
        elements: list[UFREElement],
        tokens: list[Token],
        start: int,
    ) -> tuple[int, dict[str, Any]]:
        """Try matching one iteration of a repeat unit. Return (consumed, item_dict)."""
        ti = start
        item: dict[str, Any] = {}
        for elem in elements:
            if isinstance(elem, Repeat):
                # Nested repeat: consume inner iterations
                inner_items: list[dict[str, Any]] = []
                while ti < len(tokens):
                    consumed, inner_item = self._try_repeat_extract(
                        elem.elements,
                        tokens,
                        ti,
                    )
                    if consumed == 0:
                        break
                    inner_items.append(inner_item)
                    ti += consumed
                item[elem.var_name] = inner_items
            elif isinstance(elem, Optional):
                ti += self._try_optional_extract(elem.elements, tokens, ti, item)
            elif isinstance(elem, Literal):
                if ti >= len(tokens):
                    return 0, {}
                tok = tokens[ti]
                if tok.structural_key != elem.token.structural_key:
                    return 0, {}
                if _is_value_token(elem.token.kind) and elem.token.value != tok.value:
                    return 0, {}
                ti += 1
            elif isinstance(elem, Var):
                if ti >= len(tokens):
                    return 0, {}
                tok = tokens[ti]
                if tok.structural_key != elem.structural_key:
                    return 0, {}
                if tok.value:
                    item[elem.name] = tok.value
                ti += 1
            else:
                return 0, {}
        return ti - start, item

    def _try_optional_extract(
        self,
        elements: list[UFREElement],
        tokens: list[Token],
        start: int,
        out: dict[str, Any],
    ) -> int:
        """Try matching optional elements. Return tokens consumed (0 to skip)."""
        ti = start
        tentative: dict[str, Any] = {}
        for elem in elements:
            if ti >= len(tokens):
                return 0
            tok = tokens[ti]
            if isinstance(elem, Literal):
                if tok.structural_key != elem.token.structural_key:
                    return 0
                if _is_value_token(elem.token.kind) and elem.token.value != tok.value:
                    return 0
                ti += 1
            elif isinstance(elem, Var):
                if tok.structural_key != elem.structural_key:
                    return 0
                if tok.value:
                    tentative[elem.name] = tok.value
                ti += 1
            else:
                return 0  # nested Optional not supported in v1
        out.update(tentative)
        return ti - start

    # -- fixed_mask -------------------------------------------------------

    def fixed_mask(self, page: etree._Element) -> dict[str, bool] | None:
        tokens = linearize(page)
        mask: dict[str, bool] = {}
        ti = self._match_mask(tokens, mask)
        if ti is None or ti != len(tokens):
            return None
        return mask

    def _match_mask(
        self,
        tokens: list[Token],
        mask: dict[str, bool],
    ) -> int | None:
        ti = 0
        for elem in self._ufre:
            if isinstance(elem, Literal):
                if ti >= len(tokens):
                    return None
                tok = tokens[ti]
                if tok.structural_key != elem.token.structural_key:
                    return None
                if _is_value_token(elem.token.kind):
                    if elem.token.value != tok.value:
                        return None
                    if tok.kind == "attr" or tok.value:
                        mask[tok.position_key] = True
                ti += 1

            elif isinstance(elem, Var):
                if ti >= len(tokens):
                    return None
                tok = tokens[ti]
                if tok.structural_key != elem.structural_key:
                    return None
                if elem.always_has_value and not tok.value:
                    return None
                if tok.kind == "attr" or tok.value:
                    mask[tok.position_key] = False
                ti += 1

            elif isinstance(elem, Repeat):
                while ti < len(tokens):
                    consumed = self._try_repeat_mask(elem.elements, tokens, ti, mask)
                    if consumed == 0:
                        break
                    ti += consumed

            elif isinstance(elem, Optional):
                ti += self._try_optional_mask(elem.elements, tokens, ti, mask)

        return ti

    def _try_repeat_mask(
        self,
        elements: list[UFREElement],
        tokens: list[Token],
        start: int,
        mask: dict[str, bool],
    ) -> int:
        """Try matching one repeat iteration for mask. Returns tokens consumed."""
        ti = start
        tentative: dict[str, bool] = {}
        for elem in elements:
            if isinstance(elem, Repeat):
                # Nested repeat: consume inner iterations, all variable
                while ti < len(tokens):
                    consumed = self._try_repeat_mask(
                        elem.elements,
                        tokens,
                        ti,
                        tentative,
                    )
                    if consumed == 0:
                        break
                    ti += consumed
            elif isinstance(elem, Optional):
                ti += self._try_optional_mask(elem.elements, tokens, ti, tentative)
            elif isinstance(elem, Literal):
                if ti >= len(tokens):
                    return 0
                tok = tokens[ti]
                if tok.structural_key != elem.token.structural_key:
                    return 0
                if _is_value_token(elem.token.kind) and elem.token.value != tok.value:
                    return 0
                # All positions within a repeat are variable
                if _is_value_token(tok.kind) and (tok.kind == "attr" or tok.value):
                    tentative[tok.position_key] = False
                ti += 1
            elif isinstance(elem, Var):
                if ti >= len(tokens):
                    return 0
                tok = tokens[ti]
                if tok.structural_key != elem.structural_key:
                    return 0
                if tok.kind == "attr" or tok.value:
                    tentative[tok.position_key] = False
                ti += 1
            else:
                return 0
        mask.update(tentative)
        return ti - start

    def _try_optional_mask(
        self,
        elements: list[UFREElement],
        tokens: list[Token],
        start: int,
        mask: dict[str, bool],
    ) -> int:
        ti = start
        tentative: dict[str, bool] = {}
        for elem in elements:
            if ti >= len(tokens):
                return 0
            tok = tokens[ti]
            if isinstance(elem, Literal):
                if tok.structural_key != elem.token.structural_key:
                    return 0
                if _is_value_token(elem.token.kind) and elem.token.value != tok.value:
                    return 0
                if _is_value_token(tok.kind) and (tok.kind == "attr" or tok.value):
                    tentative[tok.position_key] = True
                ti += 1
            elif isinstance(elem, Var):
                if tok.structural_key != elem.structural_key:
                    return 0
                if tok.kind == "attr" or tok.value:
                    tentative[tok.position_key] = False
                ti += 1
            else:
                return 0
        mask.update(tentative)
        return ti - start


# ---------------------------------------------------------------------------
# Inferer
# ---------------------------------------------------------------------------


class RoadRunnerInferer:
    """RoadRunner template inferer (Crescenzi, Mecca & Merialdo, VLDB 2001).

    Folds pairwise ACME across all input pages to produce a UFRE grammar.
    Iterator detection happens both during ACME (terminal-tag search with
    backward square verification) and as a post-fold pass that converts
    Optionals with repeating content to Repeat elements.
    """

    def infer(
        self,
        pages: Sequence[etree._Element],
    ) -> RoadRunnerTemplate | EmptyTemplate:
        ctr = _Counter()
        tokens = linearize(pages[0])
        wrapper = _tokens_to_ufre(tokens)

        for page in pages[1:]:
            page_tokens = linearize(page)
            result = _acme(wrapper, page_tokens, ctr)
            if result is None:
                return EmptyTemplate()
            # Post-fold: convert Optionals with repeating content to Repeats
            # so subsequent folds handle them correctly.
            wrapper = _detect_repeats(result, ctr)

        return RoadRunnerTemplate(wrapper)
