from lxml.html import fragment_fromstring

from westlean.harness import MAX_DEPTH, InferenceTestSuite
from westlean.algorithms.exalg import ExalgInferer
from westlean.strategies import template_and_schema


# ---------------------------------------------------------------------------
# Unit tests for _analyze_gap_posstring and gap analysis
# ---------------------------------------------------------------------------


class TestExalgGapAnalysis:
    """Unit tests for _analyze_gap_posstring and related gap analysis."""

    def test_optional_element_in_gap(self):
        """Gap with an element present in some pages but not others."""
        pages = [
            fragment_fromstring("<div><h1>T</h1><em>Sub</em><p>D</p></div>"),
            fragment_fromstring("<div><h1>T</h1><p>E</p></div>"),
        ]
        tpl = ExalgInferer().infer(pages)
        # With optional present
        r1 = tpl.extract(fragment_fromstring("<div><h1>T</h1><em>X</em><p>F</p></div>"))
        assert r1 is not None
        # Without optional
        r2 = tpl.extract(fragment_fromstring("<div><h1>T</h1><p>G</p></div>"))
        assert r2 is not None

    def test_optional_with_variable_text(self):
        """Optional element's text should be Var, not Literal."""
        pages = [
            fragment_fromstring("<div><h1>T</h1><em>Alpha</em><p>D</p></div>"),
            fragment_fromstring("<div><h1>T</h1><em>Beta</em><p>E</p></div>"),
            fragment_fromstring("<div><h1>T</h1><p>F</p></div>"),
        ]
        tpl = ExalgInferer().infer(pages)
        # Must accept unseen optional text
        r = tpl.extract(
            fragment_fromstring("<div><h1>T</h1><em>Gamma</em><p>G</p></div>")
        )
        assert r is not None

    def test_mixed_fixed_and_optional(self):
        """Gap with a fixed element followed by an optional element."""
        pages = [
            fragment_fromstring(
                "<div><h1>T</h1><b>Fixed</b><em>Opt</em><p>D</p></div>"
            ),
            fragment_fromstring("<div><h1>T</h1><b>Fixed</b><p>E</p></div>"),
        ]
        tpl = ExalgInferer().infer(pages)
        # With optional
        r1 = tpl.extract(
            fragment_fromstring("<div><h1>T</h1><b>Fixed</b><em>X</em><p>F</p></div>")
        )
        assert r1 is not None
        # Without optional
        r2 = tpl.extract(
            fragment_fromstring("<div><h1>T</h1><b>Fixed</b><p>G</p></div>")
        )
        assert r2 is not None

    def test_two_optionals_in_gap(self):
        """Gap with two independent optional elements."""
        pages = [
            fragment_fromstring(
                "<div><h1>T</h1><em>A</em><span>X</span><p>D</p></div>"
            ),
            fragment_fromstring("<div><h1>T</h1><span>Y</span><p>E</p></div>"),
            fragment_fromstring("<div><h1>T</h1><em>B</em><p>F</p></div>"),
        ]
        tpl = ExalgInferer().infer(pages)
        # Both present
        r1 = tpl.extract(
            fragment_fromstring("<div><h1>T</h1><em>C</em><span>Z</span><p>G</p></div>")
        )
        assert r1 is not None
        # Neither present
        r2 = tpl.extract(fragment_fromstring("<div><h1>T</h1><p>H</p></div>"))
        assert r2 is not None

    def test_loop_not_confused_with_optional(self):
        """Gap with repeated same-tag elements should be Set, not Optional."""
        pages = [
            fragment_fromstring(
                "<div><h1>T</h1><p>a</p><p>b</p><footer>E</footer></div>"
            ),
            fragment_fromstring(
                "<div><h1>T</h1><p>c</p><p>d</p><p>e</p><footer>E</footer></div>"
            ),
        ]
        tpl = ExalgInferer().infer(pages)
        # Should generalize to different item counts
        r = tpl.extract(
            fragment_fromstring("<div><h1>T</h1><p>x</p><footer>E</footer></div>")
        )
        assert r is not None
        r2 = tpl.extract(
            fragment_fromstring(
                "<div><h1>T</h1><p>x</p><p>y</p><p>z</p><p>w</p><footer>E</footer></div>"
            )
        )
        assert r2 is not None

    def test_depth_limit_prevents_infinite_recursion(self):
        """Deep nesting doesn't cause infinite recursion."""
        import sys

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(200)
        try:
            pages = [
                fragment_fromstring(
                    "<div><h1>T</h1><em>A</em><span>X</span><p>D</p></div>"
                ),
                fragment_fromstring("<div><h1>T</h1><span>Y</span><p>E</p></div>"),
                fragment_fromstring("<div><h1>T</h1><em>B</em><p>F</p></div>"),
            ]
            # Should not raise RecursionError
            tpl = ExalgInferer().infer(pages)
            assert tpl is not None
        finally:
            sys.setrecursionlimit(old_limit)

    def test_nested_loop_in_gap(self):
        """Gap with varying outer count and inner variable-count children."""
        pages = [
            fragment_fromstring(
                "<div><h1>T</h1><a><abbr>x</abbr><abbr>y</abbr></a><footer>E</footer></div>"
            ),
            fragment_fromstring(
                "<div><h1>T</h1><a><abbr>p</abbr></a><a><abbr>q</abbr><abbr>r</abbr></a><footer>E</footer></div>"
            ),
        ]
        tpl = ExalgInferer().infer(pages)
        # Different outer count than training
        test = fragment_fromstring(
            "<div><h1>T</h1><a><abbr>m</abbr><abbr>n</abbr></a><a><abbr>o</abbr></a><a><abbr>w</abbr></a><footer>E</footer></div>"
        )
        r = tpl.extract(test)
        assert r is not None

    # -- Context conflation tests (DiffFormat) --

    def test_same_tag_siblings_different_roles(self):
        """Two siblings with same tag: one loop, one fixed content."""
        pages = [
            fragment_fromstring(
                "<div><table>"
                "<tbody><tr><td>A</td></tr></tbody>"
                "<tbody><tr><td>X</td></tr><tr><td>Y</td></tr></tbody>"
                "</table></div>"
            ),
            fragment_fromstring(
                "<div><table>"
                "<tbody><tr><td>B</td></tr><tr><td>C</td></tr></tbody>"
                "<tbody><tr><td>X</td></tr><tr><td>Y</td></tr></tbody>"
                "</table></div>"
            ),
        ]
        tpl = ExalgInferer().infer(pages)
        test = fragment_fromstring(
            "<div><table>"
            "<tbody></tbody>"
            "<tbody><tr><td>X</td></tr><tr><td>Y</td></tr></tbody>"
            "</table></div>"
        )
        r = tpl.extract(test)
        assert r is not None
        mask = tpl.fixed_mask(test)
        assert mask is not None

    def test_fixed_loop_same_tag_in_sibling_parents(self):
        """Fixed element + loop of same tag inside different parent instances."""
        pages = [
            fragment_fromstring(
                "<div>"
                "<address><a>Fixed</a></address>"
                "<address><a>L1</a><a>L2</a></address>"
                "</div>"
            ),
            fragment_fromstring(
                "<div><address><a>Fixed</a></address><address><a>M1</a></address></div>"
            ),
        ]
        tpl = ExalgInferer().infer(pages)
        test = fragment_fromstring(
            "<div><address><a>Fixed</a></address><address></address></div>"
        )
        r = tpl.extract(test)
        assert r is not None

    def test_nested_fixed_count_parents(self):
        """Nested fixed-count parents with conflated grandchild contexts."""
        pages = [
            fragment_fromstring(
                "<div><table>"
                "<tbody><tr><td>A</td></tr><tr><td>B</td></tr></tbody>"
                "<tbody><tr><td>X</td></tr></tbody>"
                "</table></div>"
            ),
            fragment_fromstring(
                "<div><table>"
                "<tbody><tr><td>C</td></tr></tbody>"
                "<tbody><tr><td>X</td></tr></tbody>"
                "</table></div>"
            ),
            fragment_fromstring(
                "<div><table>"
                "<tbody><tr><td>D</td></tr><tr><td>E</td></tr><tr><td>F</td></tr></tbody>"
                "<tbody><tr><td>X</td></tr></tbody>"
                "</table></div>"
            ),
        ]
        tpl = ExalgInferer().infer(pages)
        test = fragment_fromstring(
            "<div><table>"
            "<tbody></tbody>"
            "<tbody><tr><td>X</td></tr></tbody>"
            "</table></div>"
        )
        r = tpl.extract(test)
        assert r is not None

    def test_all_loops_empty_simultaneous(self):
        """Page where every loop has 0 items simultaneously."""
        pages = [
            fragment_fromstring(
                "<div><h1>T</h1>"
                "<ul><li>A</li><li>B</li></ul>"
                "<ol><li>X</li></ol>"
                "<footer>F</footer></div>"
            ),
            fragment_fromstring(
                "<div><h1>T</h1>"
                "<ul><li>C</li></ul>"
                "<ol><li>Y</li><li>Z</li></ol>"
                "<footer>F</footer></div>"
            ),
        ]
        tpl = ExalgInferer().infer(pages)
        test = fragment_fromstring(
            "<div><h1>T</h1><ul></ul><ol></ol><footer>F</footer></div>"
        )
        r = tpl.extract(test)
        assert r is not None


class TestExalg(InferenceTestSuite):
    def make_inferer(self):
        return ExalgInferer()

    def _template_strategy(self):
        # EXALG works on fixed-structure pages for v1
        return template_and_schema(
            max_depth=MAX_DEPTH, allow_loops=False, allow_conditionals=False
        )


class TestExalgV2:
    """Deterministic tests for loop and conditional support."""

    def test_loop_extract(self):
        """Varying product count: 2 vs 3 items, extract from 4."""
        pages = [
            fragment_fromstring(
                "<div><h1>Products</h1><p>Gizmo - $10</p><p>Widget - $5</p></div>"
            ),
            fragment_fromstring(
                "<div><h1>Products</h1><p>Sprocket - $7</p><p>Gadget - $3</p><p>Doohickey - $9</p></div>"
            ),
        ]
        tpl = ExalgInferer().infer(pages)
        test = fragment_fromstring(
            "<div><h1>Products</h1><p>A - $1</p><p>B - $2</p><p>C - $3</p><p>D - $4</p></div>"
        )
        result = tpl.extract(test)
        assert result is not None
        loop_values = [v for v in result.values() if isinstance(v, list)]
        assert len(loop_values) == 1
        assert len(loop_values[0]) == 4

    def test_loop_fixed_mask(self):
        """Fixed mask marks loop positions as variable."""
        pages = [
            fragment_fromstring("<div><h1>Products</h1><p>A</p><p>B</p></div>"),
            fragment_fromstring("<div><h1>Products</h1><p>X</p><p>Y</p><p>Z</p></div>"),
        ]
        tpl = ExalgInferer().infer(pages)
        test = fragment_fromstring("<div><h1>Products</h1><p>Q</p><p>R</p></div>")
        mask = tpl.fixed_mask(test)
        assert mask is not None
        assert mask["0/text"] is True  # h1 text is fixed
        assert mask["1/text"] is False  # loop items are variable
        assert mask["2/text"] is False

    def test_loop_generalizes(self):
        """Template trained on 2-3 items recognizes 1 and 5 items."""
        pages = [
            fragment_fromstring("<div><h1>T</h1><p>A</p><p>B</p></div>"),
            fragment_fromstring("<div><h1>T</h1><p>X</p><p>Y</p><p>Z</p></div>"),
        ]
        tpl = ExalgInferer().infer(pages)
        assert (
            tpl.extract(fragment_fromstring("<div><h1>T</h1><p>Q</p></div>"))
            is not None
        )
        assert (
            tpl.extract(
                fragment_fromstring(
                    "<div><h1>T</h1><p>1</p><p>2</p><p>3</p><p>4</p><p>5</p></div>"
                )
            )
            is not None
        )

    def test_loop_zero_items(self):
        """Empty page (0 loop items) still recognized."""
        pages = [
            fragment_fromstring("<div><h1>T</h1><p>A</p><p>B</p></div>"),
            fragment_fromstring("<div><h1>T</h1><p>X</p><p>Y</p><p>Z</p></div>"),
        ]
        tpl = ExalgInferer().infer(pages)
        assert tpl.extract(fragment_fromstring("<div><h1>T</h1></div>")) is not None

    def test_loop_with_fixed_siblings(self):
        """Loop between fixed header and footer."""
        pages = [
            fragment_fromstring(
                "<div><h1>Title</h1><p>A</p><p>B</p><footer>End</footer></div>"
            ),
            fragment_fromstring(
                "<div><h1>Title</h1><p>X</p><p>Y</p><p>Z</p><footer>End</footer></div>"
            ),
        ]
        tpl = ExalgInferer().infer(pages)
        test = fragment_fromstring(
            "<div><h1>Title</h1><p>Q</p><footer>End</footer></div>"
        )
        result = tpl.extract(test)
        assert result is not None

    def test_discriminates_wrong_structure(self):
        """Rejects pages with completely different structure."""
        pages = [
            fragment_fromstring("<div><h1>Products</h1><p>A</p><p>B</p></div>"),
            fragment_fromstring("<div><h1>Products</h1><p>X</p><p>Y</p><p>Z</p></div>"),
        ]
        tpl = ExalgInferer().infer(pages)
        # Wrong root tag
        assert (
            tpl.extract(
                fragment_fromstring("<section><h1>Products</h1><p>A</p></section>")
            )
            is None
        )
        # Wrong fixed text
        assert (
            tpl.extract(fragment_fromstring("<div><h1>Wrong</h1><p>A</p></div>"))
            is None
        )

    def test_serialization_roundtrip_with_loop(self):
        """Serialize/restore preserves loop extraction capability."""
        from westlean.protocol import restore_template
        import json

        pages = [
            fragment_fromstring("<div><h1>T</h1><p>A</p><p>B</p></div>"),
            fragment_fromstring("<div><h1>T</h1><p>X</p><p>Y</p><p>Z</p></div>"),
        ]
        tpl = ExalgInferer().infer(pages)
        data = tpl.serialize()
        restored = restore_template(json.loads(json.dumps(data)))

        test = fragment_fromstring(
            "<div><h1>T</h1><p>Q</p><p>R</p><p>S</p><p>T</p></div>"
        )
        assert tpl.extract(test) == restored.extract(test)

    def test_two_loops_same_level(self):
        """Two different repeating tags at same parent level."""
        pages = [
            fragment_fromstring(
                "<div><h1>A</h1><p>x</p><p>y</p><h2>B</h2><span>a</span><span>b</span><span>c</span></div>"
            ),
            fragment_fromstring(
                "<div><h1>A</h1><p>q</p><h2>B</h2><span>d</span><span>e</span></div>"
            ),
        ]
        tpl = ExalgInferer().infer(pages)
        test = fragment_fromstring(
            "<div><h1>A</h1><p>m</p><p>n</p><p>o</p><h2>B</h2><span>r</span><span>s</span></div>"
        )
        result = tpl.extract(test)
        assert result is not None
        loop_values = [v for v in result.values() if isinstance(v, list)]
        assert len(loop_values) == 2
        lengths = sorted(len(lv) for lv in loop_values)
        assert lengths == [2, 3]

    def test_conditional_and_loop(self):
        """Optional element + loop at same parent level, separated by backbone."""
        pages = [
            fragment_fromstring(
                "<div><h1>Title</h1><em>Sub</em><h2>Items</h2><p>A</p><p>B</p></div>"
            ),
            fragment_fromstring(
                "<div><h1>Title</h1><h2>Items</h2><p>X</p><p>Y</p><p>Z</p></div>"
            ),
        ]
        tpl = ExalgInferer().infer(pages)
        assert (
            tpl.extract(
                fragment_fromstring(
                    "<div><h1>Title</h1><em>S</em><h2>Items</h2><p>Q</p></div>"
                )
            )
            is not None
        )
        assert (
            tpl.extract(
                fragment_fromstring(
                    "<div><h1>Title</h1><h2>Items</h2><p>Q</p><p>R</p></div>"
                )
            )
            is not None
        )

    def test_tag_collision_same_structural_key(self):
        """Fixed element + loop body with identical tag and no distinguishing structure."""
        pages = [
            fragment_fromstring("<div><p>Fixed</p><p>Loop1</p><p>Loop2</p></div>"),
            fragment_fromstring("<div><p>Fixed</p><p>LoopA</p></div>"),
        ]
        tpl = ExalgInferer().infer(pages)
        test = fragment_fromstring("<div><p>Fixed</p><p>X</p><p>Y</p><p>Z</p></div>")
        result = tpl.extract(test)
        assert result is not None
        loop_values = [v for v in result.values() if isinstance(v, list)]
        assert len(loop_values) == 1
        assert len(loop_values[0]) == 3

    def test_tag_collision_different_attrs(self):
        """Fixed element with attribute + loop body without — different structural keys."""
        pages = [
            fragment_fromstring(
                '<div><p class="title">Fixed</p><p>Loop1</p><p>Loop2</p></div>'
            ),
            fragment_fromstring('<div><p class="title">Fixed</p><p>LoopA</p></div>'),
        ]
        tpl = ExalgInferer().infer(pages)
        test = fragment_fromstring(
            '<div><p class="title">Fixed</p><p>X</p><p>Y</p></div>'
        )
        result = tpl.extract(test)
        assert result is not None

    def test_adjacent_loops_no_separator(self):
        """Two loops with different tags, no fixed Element between them."""
        pages = [
            fragment_fromstring("<div><p>a</p><p>b</p><span>x</span></div>"),
            fragment_fromstring("<div><p>c</p><span>y</span><span>z</span></div>"),
        ]
        tpl = ExalgInferer().infer(pages)
        test = fragment_fromstring(
            "<div><p>m</p><p>n</p><p>o</p><span>q</span><span>r</span></div>"
        )
        result = tpl.extract(test)
        assert result is not None
        loop_values = [v for v in result.values() if isinstance(v, list)]
        assert len(loop_values) == 2


class TestExalgV2Props(InferenceTestSuite):
    """Property-based tests with loop/conditional templates."""

    def make_inferer(self):
        return ExalgInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH,
            allow_loops=True,
            allow_conditionals=False,
            single_element_loops=True,
        )


class TestExalgV3Props(InferenceTestSuite):
    """Property-based tests with nested loop templates."""

    def make_inferer(self):
        return ExalgInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH,
            allow_loops=True,
            allow_conditionals=False,
            single_element_loops=True,
            max_loop_depth=2,
        )
