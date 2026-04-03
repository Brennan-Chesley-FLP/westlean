from lxml.html import fragment_fromstring

from westlean.algorithms.anti_unification import AntiUnificationInferer
from westlean.harness import MAX_DEPTH, InferenceTestSuite
from westlean.strategies import template_and_schema


class TestAntiUnification(InferenceTestSuite):
    def make_inferer(self):
        return AntiUnificationInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH, allow_loops=False, allow_conditionals=False
        )


class TestAntiUnificationBasic:
    """Deterministic unit tests for basic anti-unification behavior."""

    def test_text_variable(self):
        p1 = fragment_fromstring("<div>Hello</div>")
        p2 = fragment_fromstring("<div>World</div>")
        tpl = AntiUnificationInferer().infer([p1, p2])

        result = tpl.extract(fragment_fromstring("<div>Test</div>"))
        assert result is not None
        assert "Test" in result.values()

    def test_fixed_text_preserved(self):
        p1 = fragment_fromstring("<div>Same</div>")
        p2 = fragment_fromstring("<div>Same</div>")
        tpl = AntiUnificationInferer().infer([p1, p2])

        assert tpl.extract(fragment_fromstring("<div>Same</div>")) == {}
        assert tpl.extract(fragment_fromstring("<div>Different</div>")) is None

    def test_different_tag_rejected(self):
        p1 = fragment_fromstring("<div><span>A</span></div>")
        p2 = fragment_fromstring("<div><span>B</span></div>")
        tpl = AntiUnificationInferer().infer([p1, p2])

        assert tpl.extract(fragment_fromstring("<div><p>C</p></div>")) is None

    def test_attribute_variable(self):
        p1 = fragment_fromstring('<div class="a">text</div>')
        p2 = fragment_fromstring('<div class="b">text</div>')
        tpl = AntiUnificationInferer().infer([p1, p2])

        result = tpl.extract(fragment_fromstring('<div class="c">text</div>'))
        assert result is not None
        assert "c" in result.values()

    def test_fixed_mask(self):
        p1 = fragment_fromstring("<div>Hello</div>")
        p2 = fragment_fromstring("<div>World</div>")
        tpl = AntiUnificationInferer().infer([p1, p2])

        mask = tpl.fixed_mask(fragment_fromstring("<div>Test</div>"))
        assert mask is not None
        assert mask["text"] is False

    def test_fold_across_three(self):
        p1 = fragment_fromstring('<div class="x">A</div>')
        p2 = fragment_fromstring('<div class="x">B</div>')
        p3 = fragment_fromstring('<div class="y">C</div>')
        tpl = AntiUnificationInferer().infer([p1, p2, p3])

        result = tpl.extract(fragment_fromstring('<div class="z">D</div>'))
        assert result is not None
        assert "D" in result.values()
        assert "z" in result.values()


class TestAntiUnificationV2:
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
        tpl = AntiUnificationInferer().infer(pages)
        test = fragment_fromstring(
            "<div><h1>Products</h1><p>A - $1</p><p>B - $2</p><p>C - $3</p><p>D - $4</p></div>"
        )
        result = tpl.extract(test)
        assert result is not None
        # Should have a loop variable with 4 items
        loop_values = [v for v in result.values() if isinstance(v, list)]
        assert len(loop_values) == 1
        assert len(loop_values[0]) == 4

    def test_loop_fixed_mask(self):
        """Fixed mask marks loop positions as variable."""
        pages = [
            fragment_fromstring("<div><h1>Products</h1><p>A</p><p>B</p></div>"),
            fragment_fromstring("<div><h1>Products</h1><p>X</p><p>Y</p><p>Z</p></div>"),
        ]
        tpl = AntiUnificationInferer().infer(pages)
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
        tpl = AntiUnificationInferer().infer(pages)
        # 1 item
        assert (
            tpl.extract(fragment_fromstring("<div><h1>T</h1><p>Q</p></div>"))
            is not None
        )
        # 5 items
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
        tpl = AntiUnificationInferer().infer(pages)
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
        tpl = AntiUnificationInferer().infer(pages)
        test = fragment_fromstring(
            "<div><h1>Title</h1><p>Q</p><footer>End</footer></div>"
        )
        result = tpl.extract(test)
        assert result is not None

    def test_conditional_optional_child(self):
        """Optional child element (conditional)."""
        pages = [
            fragment_fromstring(
                "<div><h1>Profile</h1><p>Bio text</p><span>Alice</span></div>"
            ),
            fragment_fromstring("<div><h1>Profile</h1><span>Bob</span></div>"),
        ]
        tpl = AntiUnificationInferer().infer(pages)
        # With optional element
        assert (
            tpl.extract(
                fragment_fromstring(
                    "<div><h1>Profile</h1><p>New bio</p><span>Carol</span></div>"
                )
            )
            is not None
        )
        # Without optional element
        assert (
            tpl.extract(
                fragment_fromstring("<div><h1>Profile</h1><span>Dave</span></div>")
            )
            is not None
        )

    def test_discriminates_wrong_structure(self):
        """Rejects pages with completely different structure."""
        pages = [
            fragment_fromstring("<div><h1>Products</h1><p>A</p><p>B</p></div>"),
            fragment_fromstring("<div><h1>Products</h1><p>X</p><p>Y</p><p>Z</p></div>"),
        ]
        tpl = AntiUnificationInferer().infer(pages)
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
        tpl = AntiUnificationInferer().infer(pages)
        data = tpl.serialize()
        restored = restore_template(json.loads(json.dumps(data)))

        test = fragment_fromstring(
            "<div><h1>T</h1><p>Q</p><p>R</p><p>S</p><p>T</p></div>"
        )
        assert tpl.extract(test) == restored.extract(test)

    def test_three_page_fold_with_loop(self):
        """Loop detection works across 3 pages with varying counts."""
        pages = [
            fragment_fromstring("<div><h1>T</h1><p>A</p><p>B</p></div>"),
            fragment_fromstring("<div><h1>T</h1><p>X</p><p>Y</p><p>Z</p></div>"),
            fragment_fromstring(
                "<div><h1>T</h1><p>1</p><p>2</p><p>3</p><p>4</p></div>"
            ),
        ]
        tpl = AntiUnificationInferer().infer(pages)
        result = tpl.extract(fragment_fromstring("<div><h1>T</h1><p>Q</p></div>"))
        assert result is not None

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
        tpl = AntiUnificationInferer().infer(pages)
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
        tpl = AntiUnificationInferer().infer(pages)
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
        tpl = AntiUnificationInferer().infer(pages)
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
        tpl = AntiUnificationInferer().infer(pages)
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
        tpl = AntiUnificationInferer().infer(pages)
        test = fragment_fromstring(
            "<div><p>m</p><p>n</p><p>o</p><span>q</span><span>r</span></div>"
        )
        result = tpl.extract(test)
        assert result is not None
        loop_values = [v for v in result.values() if isinstance(v, list)]
        assert len(loop_values) == 2

    def test_loop_detected_from_empty_and_nonempty_pages(self):
        """Loop detected when most training pages have 0 items and some have 1+.

        Pairwise folding must not collapse children to a variable when the
        only mismatch is 0-vs-N children — the N children form a detectable
        repeating (or optional-promoted-to-repeating) region.
        """
        # 3 empty pages, then 1-item and 2-item pages
        pages = [
            fragment_fromstring("<div></div>"),
            fragment_fromstring("<div></div>"),
            fragment_fromstring("<div></div>"),
            fragment_fromstring(
                '<div><img src="http://a.aa/" alt="0" class="a0"></div>'
            ),
            fragment_fromstring(
                '<div><img src="http://a.aa/x" alt="0" class="a0"><img src="http://a.aa/y" alt="0" class="a0"></div>'
            ),
        ]
        tpl = AntiUnificationInferer().infer(pages)
        # Should detect a repeating region, not collapse to variable
        test = fragment_fromstring(
            '<div><img src="http://test.com/" alt="0" class="a0"></div>'
        )
        result = tpl.extract(test)
        assert result is not None
        # The src attribute value should be extracted
        from westlean.evaluation import flatten_values

        vals = flatten_values(result)
        assert any("http://test.com/" in v for v in vals)


class TestAntiUnificationV2Props(InferenceTestSuite):
    """Property-based tests with loop/conditional templates."""

    def make_inferer(self):
        return AntiUnificationInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH,
            allow_loops=True,
            allow_conditionals=False,
            single_element_loops=True,
        )


class TestAntiUnificationLGGCompatibleV2Props(InferenceTestSuite):
    """Property-based tests with loops that are always non-empty.

    LGG (Plotkin 1970) requires compatible inputs — same tag and arity at
    every node.  When loops always have >= 1 iteration, the DOM tree shape
    is stable across pages (the loop body element is always present), so
    the pairwise fold never encounters a shape mismatch.  This is the
    class of templates where anti-unification is theoretically sound.
    """

    def make_inferer(self):
        return AntiUnificationInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH,
            allow_loops=True,
            allow_conditionals=False,
            single_element_loops=True,
            min_loop_length=1,
        )


class TestAntiUnificationLGGCompatibleV3Props(InferenceTestSuite):
    """LGG-compatible nested loop tests (all loops have >= 1 iteration)."""

    def make_inferer(self):
        return AntiUnificationInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH,
            allow_loops=True,
            allow_conditionals=False,
            single_element_loops=True,
            max_loop_depth=2,
            min_loop_length=1,
        )


class TestAntiUnificationV3Props(InferenceTestSuite):
    """Property-based tests with nested loop templates."""

    def make_inferer(self):
        return AntiUnificationInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH,
            allow_loops=True,
            allow_conditionals=False,
            single_element_loops=True,
            max_loop_depth=2,
        )
