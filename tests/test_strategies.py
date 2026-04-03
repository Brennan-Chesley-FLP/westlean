"""Property-based tests for the template/data generator."""

from hypothesis import given, settings
from lxml import html as lxml_html

from westlean.content_model import ELEMENTS, children_allowed
from westlean.renderer import render
from westlean.strategies import template_and_schema, template_with_data
from westlean.template_ast import (
    ConditionalBlock,
    Element,
    LoopBlock,
    TemplateNode,
    TemplateVar,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_var_paths(node: TemplateNode) -> list[str]:
    """Collect all TemplateVar paths reachable from *node*."""
    paths: list[str] = []
    if isinstance(node, TemplateVar):
        paths.append(node.path)
    elif isinstance(node, Element):
        for attr_val in node.attributes.values():
            for part in attr_val.parts:
                if isinstance(part, TemplateVar):
                    paths.append(part.path)
        for child in node.children:
            paths.extend(_collect_var_paths(child))
    elif isinstance(node, ConditionalBlock):
        paths.append(node.predicate_path)
        for child in node.children:
            paths.extend(_collect_var_paths(child))
        for child in node.else_children:
            paths.extend(_collect_var_paths(child))
    elif isinstance(node, LoopBlock):
        # Loop body var paths are relative to item_var, handled by renderer
        pass
    return paths


def _check_content_model(
    node: TemplateNode, parent_tag: str | None = None
) -> list[str]:
    """Return a list of content-model violations."""
    violations: list[str] = []
    if isinstance(node, Element):
        if parent_tag and parent_tag in ELEMENTS:
            parent_def = ELEMENTS[parent_tag]
            if not children_allowed(parent_def, node.tag):
                violations.append(f"<{node.tag}> not allowed in <{parent_tag}>")
        for child in node.children:
            violations.extend(_check_content_model(child, node.tag))
    elif isinstance(node, (ConditionalBlock, LoopBlock)):
        children = (
            node.children if isinstance(node, ConditionalBlock) else node.children
        )
        for child in children:
            violations.extend(_check_content_model(child, parent_tag))
        if isinstance(node, ConditionalBlock):
            for child in node.else_children:
                violations.extend(_check_content_model(child, parent_tag))
    return violations


def _has_conditionals(node: TemplateNode) -> bool:
    """Return True if the tree contains any ConditionalBlock."""
    if isinstance(node, ConditionalBlock):
        return True
    if isinstance(node, Element):
        return any(_has_conditionals(c) for c in node.children)
    if isinstance(node, LoopBlock):
        return any(_has_conditionals(c) for c in node.children)
    return False


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestGeneration:
    @given(data=template_and_schema(max_depth=3))
    @settings(deadline=None)
    def test_generates_valid_template_and_schema(self, data):
        root, schema = data
        assert isinstance(root, Element)
        assert root.tag in ELEMENTS

    @given(triple=template_with_data(max_depth=3))
    @settings(deadline=None)
    def test_render_produces_parseable_html(self, triple):
        root, schema, data = triple
        html_str = render(root, data)
        # Must parse without error
        lxml_html.fragment_fromstring(html_str, create_parent="div")

    @given(triple=template_with_data(max_depth=3))
    @settings(deadline=None)
    def test_render_is_deterministic(self, triple):
        root, schema, data = triple
        assert render(root, data) == render(root, data)

    @given(data=template_and_schema(max_depth=3))
    @settings(deadline=None)
    def test_content_model_compliance(self, data):
        root, _schema = data
        violations = _check_content_model(root)
        assert violations == [], f"Content model violations: {violations}"

    @given(triple=template_with_data(max_depth=3))
    @settings(deadline=None)
    def test_void_elements_no_children_in_output(self, triple):
        root, schema, data = triple
        html_str = render(root, data)
        doc = lxml_html.fragment_fromstring(html_str, create_parent="div")
        void_tags = {t for t, e in ELEMENTS.items() if e.void}
        for elem in doc.iter():
            if elem.tag in void_tags:
                assert len(elem) == 0, f"<{elem.tag}> has children in output"

    @given(triple=template_with_data(max_depth=3))
    @settings(deadline=None)
    def test_schema_generates_conforming_data(self, triple):
        root, schema, instance = triple
        # All top-level var paths should resolve (loop-scoped paths are tested
        # implicitly via the render test).
        for path in _collect_var_paths(root):
            parts = path.split(".")
            # Skip loop-scoped vars (e.g. "item.name") -- they only resolve
            # inside a LoopBlock during rendering.
            if len(parts) == 1:
                assert parts[0] in instance, (
                    f"Missing key '{parts[0]}' in generated data"
                )
