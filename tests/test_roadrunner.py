import pytest
from lxml.html import fragment_fromstring

from westlean.harness import MAX_DEPTH, InferenceTestSuite
from westlean.algorithms.roadrunner import RoadRunnerInferer
from westlean.strategies import template_and_schema


class TestRoadRunner(InferenceTestSuite):
    def make_inferer(self):
        return RoadRunnerInferer()

    def _template_strategy(self):
        # RoadRunner v1 handles fixed-structure (no loops/conditionals)
        return template_and_schema(
            max_depth=MAX_DEPTH, allow_loops=False, allow_conditionals=False
        )


class TestRoadRunnerV2:
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
        tpl = RoadRunnerInferer().infer(pages)
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
        tpl = RoadRunnerInferer().infer(pages)
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
        tpl = RoadRunnerInferer().infer(pages)
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
        tpl = RoadRunnerInferer().infer(pages)
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
        tpl = RoadRunnerInferer().infer(pages)
        test = fragment_fromstring(
            "<div><h1>Title</h1><p>Q</p><footer>End</footer></div>"
        )
        result = tpl.extract(test)
        assert result is not None

    def test_conditional_optional_child(self):
        """Optional child element (conditional) via resync."""
        pages = [
            fragment_fromstring(
                "<div><h1>Profile</h1><p>Bio text</p><span>Alice</span></div>"
            ),
            fragment_fromstring("<div><h1>Profile</h1><span>Bob</span></div>"),
        ]
        tpl = RoadRunnerInferer().infer(pages)
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
        tpl = RoadRunnerInferer().infer(pages)
        # Wrong root tag
        assert (
            tpl.extract(
                fragment_fromstring("<section><h1>Products</h1><p>A</p></section>")
            )
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
        tpl = RoadRunnerInferer().infer(pages)
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
        tpl = RoadRunnerInferer().infer(pages)
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
        tpl = RoadRunnerInferer().infer(pages)
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
        tpl = RoadRunnerInferer().infer(pages)
        test = fragment_fromstring("<div><p>Fixed</p><p>X</p><p>Y</p><p>Z</p></div>")
        result = tpl.extract(test)
        assert result is not None

    def test_tag_collision_different_attrs(self):
        """Fixed element with attribute + loop body without — different structural keys."""
        pages = [
            fragment_fromstring(
                '<div><p class="title">Fixed</p><p>Loop1</p><p>Loop2</p></div>'
            ),
            fragment_fromstring('<div><p class="title">Fixed</p><p>LoopA</p></div>'),
        ]
        tpl = RoadRunnerInferer().infer(pages)
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
        tpl = RoadRunnerInferer().infer(pages)
        test = fragment_fromstring(
            "<div><p>m</p><p>n</p><p>o</p><span>q</span><span>r</span></div>"
        )
        result = tpl.extract(test)
        assert result is not None
        loop_values = [v for v in result.values() if isinstance(v, list)]
        assert len(loop_values) == 2


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Algorithm-level limitations with some loop templates. "
        "See limitations_and_deviations.md."
    ),
)
class TestRoadRunnerV2Props(InferenceTestSuite):
    """Property-based tests with loop/conditional templates."""

    def make_inferer(self):
        return RoadRunnerInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH,
            allow_loops=True,
            allow_conditionals=False,
            single_element_loops=True,
        )


@pytest.mark.xfail(
    strict=False, reason=("Nested loop support is experimental. See v4_algs.md.")
)
class TestRoadRunnerV3Props(InferenceTestSuite):
    """Property-based tests with nested loop templates."""

    def make_inferer(self):
        return RoadRunnerInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH,
            allow_loops=True,
            allow_conditionals=False,
            single_element_loops=True,
            max_loop_depth=2,
        )
