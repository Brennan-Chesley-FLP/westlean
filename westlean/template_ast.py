from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class TextNode:
    """Static text content."""

    text: str


@dataclass(frozen=True)
class TemplateVar:
    """A template hole filled from the data dict.

    ``path`` is a dot-separated key into the data dict, e.g. ``"title"``
    or ``"item.name"`` when inside a loop whose ``item_var`` is ``"item"``.
    """

    path: str


@dataclass(frozen=True)
class AttributeValue:
    """An attribute value composed of static strings and/or template variables.

    A purely static attribute: ``parts = ("value",)``
    A purely dynamic attribute: ``parts = (TemplateVar("x"),)``
    Mixed: ``parts = ("prefix-", TemplateVar("id"), "-suffix")``
    """

    parts: tuple[Union[str, TemplateVar], ...]


@dataclass(frozen=True)
class Element:
    """An HTML element with tag, attributes, and children."""

    tag: str
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    children: tuple[TemplateNode, ...] = ()


@dataclass(frozen=True)
class ConditionalBlock:
    """Renders children only when the boolean at ``predicate_path`` is truthy.

    ``else_children`` provides the fallback branch.
    """

    predicate_path: str
    children: tuple[TemplateNode, ...] = ()
    else_children: tuple[TemplateNode, ...] = ()


@dataclass(frozen=True)
class LoopBlock:
    """Repeats children once per item in the list at ``list_path``.

    Inside children, template variables prefixed with ``item_var.`` resolve
    against the current iteration item.
    """

    list_path: str
    item_var: str
    children: tuple[TemplateNode, ...] = ()


@dataclass(frozen=True)
class CommentNode:
    """An HTML comment node. Contains only text content (static or variable)."""

    children: tuple[Union[TextNode, TemplateVar], ...] = ()


# Union of every node that can appear in a template tree.
TemplateNode = Union[
    TextNode, TemplateVar, Element, ConditionalBlock, LoopBlock, CommentNode
]
