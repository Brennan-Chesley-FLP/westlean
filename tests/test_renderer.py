import pytest

from westlean.renderer import RenderError, render
from westlean.template_ast import (
    AttributeValue,
    ConditionalBlock,
    Element,
    LoopBlock,
    TemplateVar,
    TextNode,
)


class TestRenderStatic:
    def test_simple_div_with_text(self):
        root = Element(tag="div", children=(TextNode("hello"),))
        assert render(root, {}) == "<div>hello</div>"

    def test_nested_elements(self):
        root = Element(
            tag="div",
            children=(
                Element(tag="p", children=(TextNode("one"),)),
                Element(tag="p", children=(TextNode("two"),)),
            ),
        )
        html = render(root, {})
        assert "<div>" in html
        assert "<p>one</p>" in html
        assert "<p>two</p>" in html

    def test_void_element(self):
        root = Element(tag="div", children=(Element(tag="br"),))
        html = render(root, {})
        assert "<br>" in html

    def test_static_attributes(self):
        root = Element(
            tag="div",
            attributes={"class": AttributeValue(parts=("container",))},
            children=(TextNode("x"),),
        )
        html = render(root, {})
        assert 'class="container"' in html

    def test_interleaved_text_and_elements(self):
        root = Element(
            tag="div",
            children=(
                TextNode("before "),
                Element(tag="span", children=(TextNode("mid"),)),
                TextNode(" after"),
            ),
        )
        html = render(root, {})
        assert "before " in html
        assert "<span>mid</span>" in html
        assert " after" in html


class TestRenderVars:
    def test_text_var(self):
        root = Element(tag="div", children=(TemplateVar(path="name"),))
        assert render(root, {"name": "Alice"}) == "<div>Alice</div>"

    def test_attr_var(self):
        root = Element(
            tag="a",
            attributes={"href": AttributeValue(parts=(TemplateVar(path="url"),))},
            children=(TextNode("link"),),
        )
        html = render(root, {"url": "http://example.com"})
        assert 'href="http://example.com"' in html

    def test_mixed_attr(self):
        root = Element(
            tag="div",
            attributes={
                "class": AttributeValue(parts=("item-", TemplateVar(path="id"), "-end"))
            },
            children=(TextNode("x"),),
        )
        html = render(root, {"id": "42"})
        assert 'class="item-42-end"' in html

    def test_missing_var_raises(self):
        root = Element(tag="div", children=(TemplateVar(path="missing"),))
        with pytest.raises(RenderError):
            render(root, {})

    def test_nested_path(self):
        root = Element(tag="div", children=(TemplateVar(path="user.name"),))
        html = render(root, {"user": {"name": "Bob"}})
        assert "Bob" in html


class TestRenderConditional:
    def test_true_shows_children(self):
        root = Element(
            tag="div",
            children=(
                ConditionalBlock(
                    predicate_path="visible",
                    children=(TextNode("yes"),),
                    else_children=(TextNode("no"),),
                ),
            ),
        )
        assert "yes" in render(root, {"visible": True})

    def test_false_shows_else(self):
        root = Element(
            tag="div",
            children=(
                ConditionalBlock(
                    predicate_path="visible",
                    children=(TextNode("yes"),),
                    else_children=(TextNode("no"),),
                ),
            ),
        )
        assert "no" in render(root, {"visible": False})

    def test_false_no_else_is_empty(self):
        root = Element(
            tag="div",
            children=(
                ConditionalBlock(
                    predicate_path="visible",
                    children=(Element(tag="span", children=(TextNode("x"),)),),
                ),
            ),
        )
        html = render(root, {"visible": False})
        assert "<span>" not in html


class TestRenderLoop:
    def test_basic_loop(self):
        root = Element(
            tag="ul",
            children=(
                LoopBlock(
                    list_path="items",
                    item_var="item",
                    children=(
                        Element(
                            tag="li",
                            children=(TemplateVar(path="item.name"),),
                        ),
                    ),
                ),
            ),
        )
        html = render(root, {"items": [{"name": "a"}, {"name": "b"}]})
        assert "<li>a</li>" in html
        assert "<li>b</li>" in html

    def test_empty_loop(self):
        root = Element(
            tag="div",
            children=(
                LoopBlock(
                    list_path="items",
                    item_var="item",
                    children=(TextNode("x"),),
                ),
            ),
        )
        html = render(root, {"items": []})
        assert html == "<div></div>"

    def test_loop_non_list_raises(self):
        root = Element(
            tag="div",
            children=(
                LoopBlock(
                    list_path="items",
                    item_var="item",
                    children=(TextNode("x"),),
                ),
            ),
        )
        with pytest.raises(RenderError):
            render(root, {"items": "not a list"})
