"""Compatibility utilities for lxml element handling."""

from __future__ import annotations

from lxml import etree

COMMENT_TAG = "#comment"


def element_tag(elem: etree._Element) -> str:
    """Return a stable string tag for any lxml element, including comments."""
    if elem.tag is etree.Comment:
        return COMMENT_TAG
    return str(elem.tag)
