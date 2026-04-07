from __future__ import annotations

from lxml import etree
from lxml.html import tostring as html_tostring

from westlean.template_ast import (
    AttributeValue,
    CommentNode,
    ConditionalBlock,
    Element,
    LoopBlock,
    TemplateNode,
    TemplateVar,
    TextNode,
)


class RenderError(Exception):
    """Raised when rendering fails due to missing or invalid data."""


def _resolve_path(data: dict, path: str) -> object:
    """Resolve a dot-separated path against a nested data dict."""
    parts = path.split(".")
    current: object = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise RenderError(f"Missing data at path '{path}': key '{part}' not found")
        current = current[part]
    return current


def _render_attr_value(attr_val: AttributeValue, data: dict) -> str:
    """Render an attribute value, resolving any template vars."""
    pieces: list[str] = []
    for part in attr_val.parts:
        if isinstance(part, str):
            pieces.append(part)
        elif isinstance(part, TemplateVar):
            pieces.append(str(_resolve_path(data, part.path)))
    return "".join(pieces)


def _render_node(node: TemplateNode, data: dict) -> list[etree._Element | str]:
    """Render a template node to a list of lxml elements and/or strings.

    Returns a list because control-flow blocks (conditionals, loops) may
    expand to zero or many siblings.
    """
    if isinstance(node, TextNode):
        return [node.text]

    if isinstance(node, TemplateVar):
        return [str(_resolve_path(data, node.path))]

    if isinstance(node, Element):
        elem = etree.Element(node.tag)
        for attr_name, attr_val in node.attributes.items():
            elem.set(attr_name, _render_attr_value(attr_val, data))
        _append_children(elem, node.children, data)
        return [elem]

    if isinstance(node, ConditionalBlock):
        branch = (
            node.children
            if _resolve_path(data, node.predicate_path)
            else node.else_children
        )
        results: list[etree._Element | str] = []
        for child in branch:
            results.extend(_render_node(child, data))
        return results

    if isinstance(node, LoopBlock):
        items = _resolve_path(data, node.list_path)
        if not isinstance(items, list):
            raise RenderError(
                f"Expected list at '{node.list_path}', got {type(items).__name__}"
            )
        results = []
        for item in items:
            loop_data = {**data, node.item_var: item}
            for child in node.children:
                results.extend(_render_node(child, loop_data))
        return results

    if isinstance(node, CommentNode):
        pieces: list[str] = []
        for part in node.children:
            if isinstance(part, TextNode):
                pieces.append(part.text)
            elif isinstance(part, TemplateVar):
                pieces.append(str(_resolve_path(data, part.path)))
        return [etree.Comment("".join(pieces))]

    raise RenderError(f"Unknown node type: {type(node)}")  # pragma: no cover


def _append_children(
    parent: etree._Element,
    children: tuple[TemplateNode, ...],
    data: dict,
) -> None:
    """Append rendered children to *parent*, respecting lxml's text model.

    lxml stores text in two places:
    * ``element.text`` — text before the first child element
    * ``child.tail``   — text after that child element (before the next sibling)
    """
    last_elem: etree._Element | None = None

    for child_node in children:
        for item in _render_node(child_node, data):
            if isinstance(item, str):
                if not item:
                    continue
                if last_elem is None:
                    parent.text = (parent.text or "") + item
                else:
                    last_elem.tail = (last_elem.tail or "") + item
            else:
                parent.append(item)
                last_elem = item


def render(root: Element, data: dict) -> str:
    """Render a template root element with *data* and return an HTML string."""
    results = _render_node(root, data)
    if len(results) != 1 or not isinstance(results[0], etree._Element):
        raise RenderError("Template root must render to exactly one element")
    return html_tostring(results[0], encoding="unicode")  # type: ignore[call-overload]
