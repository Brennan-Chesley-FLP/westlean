from lxml.html import fragment_fromstring

from westlean.harness import MAX_DEPTH, InferenceTestSuite
from westlean.algorithms.tree_automata import KTestableInferer
from westlean.strategies import template_and_schema


class TestKTestable(InferenceTestSuite):
    def make_inferer(self):
        return KTestableInferer(k=2)

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH, allow_loops=False, allow_conditionals=False
        )


class TestKTestableK3(InferenceTestSuite):
    def make_inferer(self):
        return KTestableInferer(k=3)

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH, allow_loops=False, allow_conditionals=False
        )


class TestKTestableV2:
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
        tpl = KTestableInferer(k=2).infer(pages)
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
        tpl = KTestableInferer(k=2).infer(pages)
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
        tpl = KTestableInferer(k=2).infer(pages)
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
        tpl = KTestableInferer(k=2).infer(pages)
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
        tpl = KTestableInferer(k=2).infer(pages)
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
        tpl = KTestableInferer(k=2).infer(pages)
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
        tpl = KTestableInferer(k=2).infer(pages)
        data = tpl.serialize()
        restored = restore_template(json.loads(json.dumps(data)))

        test = fragment_fromstring(
            "<div><h1>T</h1><p>Q</p><p>R</p><p>S</p><p>T</p></div>"
        )
        assert tpl.extract(test) == restored.extract(test)

    def test_k3_loop(self):
        """k=3 also handles loops with compressed patterns."""
        pages = [
            fragment_fromstring("<div><h1>T</h1><p>A</p><p>B</p></div>"),
            fragment_fromstring("<div><h1>T</h1><p>X</p><p>Y</p><p>Z</p></div>"),
        ]
        tpl = KTestableInferer(k=3).infer(pages)
        assert (
            tpl.extract(fragment_fromstring("<div><h1>T</h1><p>Q</p></div>"))
            is not None
        )
        assert (
            tpl.extract(
                fragment_fromstring(
                    "<div><h1>T</h1><p>1</p><p>2</p><p>3</p><p>4</p></div>"
                )
            )
            is not None
        )

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
        tpl = KTestableInferer(k=2).infer(pages)
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
        tpl = KTestableInferer(k=2).infer(pages)
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
        tpl = KTestableInferer(k=2).infer(pages)
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
        tpl = KTestableInferer(k=2).infer(pages)
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
        tpl = KTestableInferer(k=2).infer(pages)
        test = fragment_fromstring(
            "<div><p>m</p><p>n</p><p>o</p><span>q</span><span>r</span></div>"
        )
        result = tpl.extract(test)
        assert result is not None
        loop_values = [v for v in result.values() if isinstance(v, list)]
        assert len(loop_values) == 2

    def test_loop_vs_fixed_single_contentless_element(self):
        """Loop of contentless elements vs fixed single — fundamental ambiguity.

        Template A: section(loop(br)), accepts 0+ br's.
        Template B: section(br), always has 1 br.
        A page with 1 br from Template B is indistinguishable from a
        1-item rendering of Template A.
        """
        train_a = [
            fragment_fromstring("<section></section>"),
            fragment_fromstring("<section></section>"),
            fragment_fromstring("<section><br><br></section>"),
        ]
        tpl = KTestableInferer(k=2).infer(train_a)
        neg = fragment_fromstring("<section><br></section>")
        # This is a fundamental limitation: the neg page IS a valid
        # rendering of template A with 1 item.  No tree-based algorithm
        # can distinguish this without external context.
        assert tpl.extract(neg) is not None

    def test_optional_inner_structure_mismatch(self):
        """Optional child with wrong inner structure should be rejected.

        With 2 pages (0 vs 1 item), the child is optional.
        The optional child's inner structure is fixed (3 inputs).
        A test page with the optional child but 0 inputs should be rejected.
        """
        small_pages = [
            fragment_fromstring("<div></div>"),
            fragment_fromstring(
                '<div><abbr class="a0"><input><input><input>'
                '<a class="a0">x</a></abbr></div>'
            ),
        ]
        tpl = KTestableInferer(k=2).infer(small_pages)
        # Test page has the optional abbr but with wrong inner structure
        test = fragment_fromstring(
            '<div><abbr class="a0"><a class="a0">q</a></abbr></div>'
        )
        assert tpl.extract(test) is None

    def test_stability_optional_to_repeating(self):
        """Adding more training pages should not degrade recognition.

        With 2 pages (0 vs 1 item), child is optional.
        With 5 pages (0,0,0,1,2 items), child becomes repeating.
        Both templates should behave consistently on test pages that
        match the learned inner structure.
        """
        small_pages = [
            fragment_fromstring("<div></div>"),
            fragment_fromstring(
                '<div><abbr class="a0"><input><input><input>'
                '<a class="a0">x</a></abbr></div>'
            ),
        ]
        large_pages = small_pages + [
            fragment_fromstring("<div></div>"),
            fragment_fromstring("<div></div>"),
            fragment_fromstring(
                '<div><abbr class="a0"><input><input><input>'
                '<a class="a0">y</a></abbr>'
                '<abbr class="a0"><input><input><input>'
                '<a class="a0">z</a></abbr></div>'
            ),
        ]
        # Test page with correct inner structure
        test_ok = fragment_fromstring(
            '<div><abbr class="a0"><input><input><input>'
            '<a class="a0">q</a></abbr></div>'
        )
        tpl_small = KTestableInferer(k=2).infer(small_pages)
        tpl_large = KTestableInferer(k=2).infer(large_pages)
        small_ok = tpl_small.extract(test_ok) is not None
        large_ok = tpl_large.extract(test_ok) is not None
        if small_ok:
            assert large_ok, "Recognition degraded after adding more training pages"

        # Test page with wrong inner structure — both should reject
        test_bad = fragment_fromstring(
            '<div><abbr class="a0"><a class="a0">q</a></abbr></div>'
        )
        assert tpl_small.extract(test_bad) is None
        assert tpl_large.extract(test_bad) is None

    def test_loop_with_variable_children(self):
        """Loop rows whose cells have variable children should still match.

        When the third <td> has an <a class="x"> in one page but a bare <a>
        in another, the attribute name mismatch makes _build_uta_tree give up
        on that cell's children, falling back to children=None (fully
        variable).  The structural matcher must then accept page <td> elements
        that have children.
        """
        pages = [
            fragment_fromstring(
                "<table><tbody>"
                '<tr><td>A</td><td>1</td><td><a class="x">link1</a></td></tr>'
                '<tr><td>B</td><td>2</td><td><a class="x">link2</a></td></tr>'
                "</tbody></table>"
            ),
            fragment_fromstring(
                "<table><tbody>"
                "<tr><td>C</td><td>3</td><td><a>link3</a></td></tr>"
                "</tbody></table>"
            ),
        ]
        tpl = KTestableInferer(k=2).infer(pages)
        test = fragment_fromstring(
            "<table><tbody>"
            '<tr><td>X</td><td>9</td><td><a class="y">link4</a></td></tr>'
            '<tr><td>Y</td><td>8</td><td><a class="z">link5</a></td></tr>'
            "</tbody></table>"
        )
        result = tpl.extract(test)
        assert result is not None, "Failed to match loop with variable-children cells"
        loop_values = [v for v in result.values() if isinstance(v, list)]
        assert len(loop_values) == 1
        assert len(loop_values[0]) == 2


class TestKTestableV2Props(InferenceTestSuite):
    """Property-based tests with loop/conditional templates."""

    def make_inferer(self):
        return KTestableInferer(k=2)

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH,
            allow_loops=True,
            allow_conditionals=False,
            single_element_loops=True,
        )


class TestKTestableV3Props(InferenceTestSuite):
    """Property-based tests with nested loop templates."""

    def make_inferer(self):
        return KTestableInferer(k=2)

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH,
            allow_loops=True,
            allow_conditionals=False,
            single_element_loops=True,
            max_loop_depth=2,
        )
