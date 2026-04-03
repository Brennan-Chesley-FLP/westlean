"""Tests for the evaluation harness using an oracle inferer.

The oracle "cheats" — it stores the original template and uses the real
renderer/mask to extract data.  This verifies the harness itself works
before any real algorithms exist.
"""

from __future__ import annotations

from typing import Any, Sequence

import hypothesis.strategies as st
from hypothesis import given, settings
from lxml import etree

from westlean.evaluation import (
    EvaluationResult,
    build_position_map,
    evaluate,
    flatten_values,
    ground_truth_mask,
    parse_html,
)
from westlean.harness import MAX_DEPTH, InferenceTestSuite
from westlean.protocol import InferredTemplate, TemplateInferer
from westlean.renderer import render
from westlean.strategies import template_and_schema, template_with_data
from westlean.template_ast import Element
from westlean.field_strategies import field_strategy


# ---------------------------------------------------------------------------
# Oracle implementation — perfect inferer for testing the harness
# ---------------------------------------------------------------------------


class OracleTemplate:
    """An InferredTemplate that uses ground-truth knowledge to extract/classify."""

    def __init__(self, template: Element, known_data: list[dict]) -> None:
        self._template = template
        self._known_pages: dict[str, dict] = {}
        for d in known_data:
            html = render(template, d)
            self._known_pages[html] = d
        # Store rendered page strings for matching
        self._page_strings = set(self._known_pages.keys())
        # Store the structural signature for matching new pages
        self._signature = self._compute_signature(template)

    def _compute_signature(self, elem: Element) -> tuple:
        """Compute a structural signature (tag + child structure) for matching."""
        child_sigs: list[tuple | str] = []
        for child in elem.children:
            if isinstance(child, Element):
                child_sigs.append(self._compute_signature(child))
            else:
                child_sigs.append(type(child).__name__)
        return (elem.tag, tuple(sorted(elem.attributes.keys())), tuple(child_sigs))

    def _page_signature(self, page: etree._Element) -> tuple:
        """Compute a structural signature from a rendered DOM tree."""
        child_sigs = []
        for child in page:
            child_sigs.append(self._page_signature(child))
        return (page.tag, tuple(sorted(page.attrib.keys())), tuple(child_sigs))

    def extract(self, page: etree._Element) -> dict[str, Any] | None:
        """Extract data by matching against known pages or structure."""
        from lxml.html import tostring

        html = tostring(page, encoding="unicode")  # type: ignore[call-overload]

        # Exact match against known pages
        if html in self._known_pages:
            return dict(self._known_pages[html])

        # Structural match: try to render with each known data and see if
        # the structure is compatible. For the oracle, we accept any page
        # whose tag structure matches.
        page_sig = self._page_signature(page)
        for known_html, known_data in self._known_pages.items():
            known_page = parse_html(known_html)
            if self._page_signature(known_page) == page_sig:
                # Structure matches — extract by diffing text content
                extracted: dict[str, Any] = {}
                self._diff_extract(known_page, page, extracted, 0)
                return extracted

        return None

    def _diff_extract(
        self,
        known: etree._Element,
        page: etree._Element,
        out: dict[str, Any],
        counter: int,
    ) -> int:
        """Extract variable values by comparing a known page against a new page."""
        # Compare text
        if (known.text or "") != (page.text or ""):
            out[f"var_{counter}"] = page.text or ""
            counter += 1
        # Compare attributes
        for attr in sorted(set(known.attrib) | set(page.attrib)):
            if known.get(attr, "") != page.get(attr, ""):
                out[f"var_{counter}"] = page.get(attr, "")
                counter += 1
        # Compare children
        for known_child, page_child in zip(known, page):
            counter = self._diff_extract(known_child, page_child, out, counter)
            if (known_child.tail or "") != (page_child.tail or ""):
                out[f"var_{counter}"] = page_child.tail or ""
                counter += 1
        return counter

    def fixed_mask(self, page: etree._Element) -> dict[str, bool] | None:
        """Compute mask by comparing against a known page."""
        # Find a known page with matching structure
        page_sig = self._page_signature(page)
        known_page = None
        for known_html in self._known_pages:
            kp = parse_html(known_html)
            if self._page_signature(kp) == page_sig:
                known_page = kp
                break

        if known_page is None:
            return None

        # Build mask by diffing: positions that differ are variable
        mask: dict[str, bool] = {}
        page_map = build_position_map(page)
        known_map = build_position_map(known_page)

        for key in page_map:
            if key in known_map:
                mask[key] = page_map[key] == known_map[key]
            else:
                mask[key] = False  # new position = variable
        return mask

    def serialize(self) -> dict:
        return {"algorithm": "oracle"}

    def get_relax_ng(self) -> str:
        """Generate a permissive schema from the known page structure."""
        from westlean.algorithms.anti_unification import (
            AntiUnificationInferer,
        )

        if not self._known_pages:
            return ""
        pages = [parse_html(html) for html in self._known_pages]
        tpl = AntiUnificationInferer().infer(pages)
        from westlean.algorithms.anti_unification import AntiUnifiedTemplate

        if isinstance(tpl, AntiUnifiedTemplate):
            return tpl.get_relax_ng()
        return ""


class OracleInferer:
    """A TemplateInferer that uses ground-truth knowledge."""

    def __init__(self, template: Element, all_data: list[dict]) -> None:
        self._template = template
        self._all_data = all_data

    def infer(self, pages: Sequence[etree._Element]) -> OracleTemplate:
        return OracleTemplate(self._template, self._all_data)


# ---------------------------------------------------------------------------
# Tests for the oracle + harness
# ---------------------------------------------------------------------------


class TestOracleInferer:
    """Verify the oracle itself works, so we can trust it for harness testing."""

    @given(triple=template_with_data(max_depth=2))
    @settings(deadline=None, max_examples=30)
    def test_oracle_recognizes_known_pages(self, triple):
        template, schema, data = triple
        oracle = OracleTemplate(template, [data])
        page = parse_html(render(template, data))
        assert oracle.extract(page) is not None

    @given(triple=template_with_data(max_depth=2))
    @settings(deadline=None, max_examples=30)
    def test_oracle_produces_mask(self, triple):
        template, schema, data = triple
        oracle = OracleTemplate(template, [data])
        page = parse_html(render(template, data))
        mask = oracle.fixed_mask(page)
        assert mask is not None


class TestGroundTruthMask:
    """Verify ground_truth_mask works on basic cases."""

    @given(triple=template_with_data(max_depth=2))
    @settings(deadline=None, max_examples=30)
    def test_ground_truth_mask_runs(self, triple):
        template, schema, data = triple
        mask = ground_truth_mask(template, data)
        assert isinstance(mask, dict)
        # All values should be booleans
        for v in mask.values():
            assert isinstance(v, bool)


class TestEvaluation:
    """Verify the evaluate() function runs end-to-end."""

    @given(data=st.data())
    @settings(deadline=None, max_examples=10)
    def test_evaluate_runs(self, data):
        template, schema = data.draw(
            template_and_schema(
                max_depth=2, allow_loops=False, allow_conditionals=False
            )
        )
        all_data = [data.draw(field_strategy(schema)) for _ in range(5)]

        train_data = all_data[:3]
        test_data = all_data[3:]

        inferer = OracleInferer(template, train_data)
        result = evaluate(inferer, template, schema, train_data, test_data, [])

        assert isinstance(result, EvaluationResult)
        assert result.recognition_rate >= 0.0
        assert result.n_train == 3
        assert result.n_test == 2


class TestBuildPositionMap:
    def test_simple_element(self):
        from lxml import etree

        root = etree.Element("div")
        root.text = "hello"
        child = etree.SubElement(root, "span")
        child.text = "world"
        child.tail = " after"

        pos = build_position_map(root)
        assert pos["text"] == "hello"
        assert pos["0/text"] == "world"
        assert pos["0/tail"] == " after"

    def test_attributes(self):
        from lxml import etree

        root = etree.Element("a")
        root.set("href", "http://example.com")
        root.set("class", "link")
        root.text = "click"

        pos = build_position_map(root)
        assert pos["@class"] == "link"
        assert pos["@href"] == "http://example.com"
        assert pos["text"] == "click"


class TestFlattenValues:
    def test_flat_dict(self):
        assert flatten_values({"a": "hello", "b": "world"}) == {"hello", "world"}

    def test_nested_dict(self):
        assert flatten_values({"user": {"name": "Alice"}}) == {"Alice"}

    def test_list_of_dicts(self):
        result = flatten_values({"items": [{"x": "a"}, {"x": "b"}]})
        assert result == {"a", "b"}

    def test_bools_excluded(self):
        assert flatten_values({"show": True, "name": "x"}) == {"x"}

    def test_ints_stringified(self):
        assert flatten_values({"count": 42}) == {"42"}

    def test_empty_strings_excluded(self):
        assert flatten_values({"a": ""}) == set()


# ---------------------------------------------------------------------------
# Harness test suite with the oracle — verifies the mixin tests pass
# ---------------------------------------------------------------------------


class TestOracleViaHarness(InferenceTestSuite):
    """Run the standardized test suite using a diff-based oracle.

    Uses templates without loops/conditionals so the diff-based approach
    (which requires fixed tree structure) can work.  If these fail,
    the harness has a bug — not the algorithm.
    """

    def make_inferer(self) -> TemplateInferer:
        return DeferredOracleInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH, allow_loops=False, allow_conditionals=False
        )


class DeferredOracleInferer:
    """An inferer that builds an oracle from the pages it receives."""

    def infer(self, pages: Sequence[etree._Element]) -> InferredTemplate:
        return DiffBasedTemplate(list(pages))


class DiffBasedTemplate:
    """A template inferred by diffing pages — the simplest real strategy.

    Identifies "variable" positions by finding values that differ across
    training pages.  On extract(), returns the values at those positions.
    """

    def __init__(self, pages: list[etree._Element]) -> None:
        self._pages = pages
        # Pre-compute position maps and variable positions
        self._maps = [build_position_map(p) for p in pages]
        self._variable_keys: set[str] = set()
        if len(self._maps) >= 2:
            ref = self._maps[0]
            for m in self._maps[1:]:
                for key in ref:
                    if key in m and ref[key] != m[key]:
                        self._variable_keys.add(key)
                # Keys present in one but not another are also variable
                for key in set(ref) ^ set(m):
                    self._variable_keys.add(key)

    def extract(self, page: etree._Element) -> dict[str, Any] | None:
        if not self._pages:
            return None
        if not self._structurally_compatible(self._pages[0], page):
            return None

        page_map = build_position_map(page)
        extracted: dict[str, Any] = {}
        counter = 0
        for key in sorted(self._variable_keys):
            if key in page_map:
                extracted[f"var_{counter}"] = page_map[key]
                counter += 1
        return extracted

    def fixed_mask(self, page: etree._Element) -> dict[str, bool] | None:
        if not self._pages:
            return None
        if not self._structurally_compatible(self._pages[0], page):
            return None

        page_map = build_position_map(page)
        mask: dict[str, bool] = {}
        for key in page_map:
            mask[key] = key not in self._variable_keys
        return mask

    def serialize(self) -> dict:
        return {"algorithm": "diff_based"}

    def get_relax_ng(self) -> str:
        from westlean.algorithms.anti_unification import (
            AntiUnificationInferer,
            AntiUnifiedTemplate,
        )

        if not self._pages:
            return ""
        tpl = AntiUnificationInferer().infer(self._pages)
        if isinstance(tpl, AntiUnifiedTemplate):
            return tpl.get_relax_ng()
        return ""

    def _structurally_compatible(self, a: etree._Element, b: etree._Element) -> bool:
        """Check if two trees have the same tag + attribute-set + child structure."""
        if a.tag != b.tag:
            return False
        if len(a) != len(b):
            return False
        if set(a.attrib) != set(b.attrib):
            return False
        # Check text/no-text at same positions
        if bool(a.text) != bool(b.text):
            return False
        return all(self._structurally_compatible(ca, cb) for ca, cb in zip(a, b))


# ---------------------------------------------------------------------------
# Nested loop generation tests
# ---------------------------------------------------------------------------


class TestNestedLoopGeneration:
    """Verify the generator can produce and render nested loop templates."""

    def test_manual_nested_loop_renders(self):
        """Hand-crafted nested loop: users with posts."""
        from westlean.template_ast import TextNode, TemplateVar, LoopBlock

        tpl = Element(
            tag="div",
            attributes={},
            children=(
                Element(tag="h1", attributes={}, children=(TextNode(text="Users"),)),
                LoopBlock(
                    list_path="users",
                    item_var="user",
                    children=(
                        Element(
                            tag="section",
                            attributes={},
                            children=(
                                TemplateVar(path="user.name"),
                                LoopBlock(
                                    list_path="user.posts",
                                    item_var="post",
                                    children=(
                                        Element(
                                            tag="p",
                                            attributes={},
                                            children=(TemplateVar(path="post.title"),),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        data = {
            "users": [
                {"name": "Alice", "posts": [{"title": "Hello"}, {"title": "World"}]},
                {"name": "Bob", "posts": [{"title": "Foo"}]},
            ]
        }
        html = render(tpl, data)
        page = parse_html(html)
        assert page is not None
        # Verify structure: div > h1 + section*2
        assert page.tag == "div"
        sections = [c for c in page if c.tag == "section"]
        assert len(sections) == 2
        # Alice's section has 2 <p> (posts) + text
        alice_ps = [c for c in sections[0] if c.tag == "p"]
        assert len(alice_ps) == 2
        assert alice_ps[0].text == "Hello"
        # Bob's section has 1 <p>
        bob_ps = [c for c in sections[1] if c.tag == "p"]
        assert len(bob_ps) == 1

    def test_manual_nested_loop_ground_truth_mask(self):
        """Ground truth mask works for nested loop templates."""
        from westlean.template_ast import TextNode, TemplateVar, LoopBlock

        tpl = Element(
            tag="div",
            attributes={},
            children=(
                Element(tag="h1", attributes={}, children=(TextNode(text="Title"),)),
                LoopBlock(
                    list_path="rows",
                    item_var="row",
                    children=(
                        Element(
                            tag="ul",
                            attributes={},
                            children=(
                                LoopBlock(
                                    list_path="row.cells",
                                    item_var="cell",
                                    children=(
                                        Element(
                                            tag="li",
                                            attributes={},
                                            children=(TemplateVar(path="cell.val"),),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        data = {
            "rows": [
                {"cells": [{"val": "A"}, {"val": "B"}]},
                {"cells": [{"val": "X"}]},
            ]
        }
        mask = ground_truth_mask(tpl, data)
        assert mask is not None
        # h1 text is fixed
        assert mask["0/text"] is True
        # li text values are variable
        variable_keys = [k for k, v in mask.items() if not v]
        assert len(variable_keys) > 0

    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_generated_nested_loops_render(self, data):
        """Generated nested loop templates render without errors."""
        template, schema = data.draw(
            template_and_schema(
                max_depth=4,
                allow_loops=True,
                single_element_loops=True,
                max_loop_depth=2,
            )
        )
        d = data.draw(field_strategy(schema))
        html = render(template, d)
        page = parse_html(html)
        assert page is not None

    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_generated_nested_loops_mask(self, data):
        """Ground truth mask works for generated nested loop templates."""
        template, schema = data.draw(
            template_and_schema(
                max_depth=4,
                allow_loops=True,
                single_element_loops=True,
                max_loop_depth=2,
            )
        )
        d = data.draw(field_strategy(schema))
        mask = ground_truth_mask(template, d)
        assert mask is not None
