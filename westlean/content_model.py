from __future__ import annotations

import enum
from dataclasses import dataclass


class ContentCategory(enum.Flag):
    """HTML5 content categories as a flag enum for set-like operations."""

    NONE = 0
    METADATA = enum.auto()
    FLOW = enum.auto()
    SECTIONING = enum.auto()
    HEADING = enum.auto()
    PHRASING = enum.auto()
    EMBEDDED = enum.auto()
    INTERACTIVE = enum.auto()


@dataclass(frozen=True)
class AttributeSpec:
    """Definition of a valid HTML attribute."""

    name: str
    required: bool = False
    values: tuple[str, ...] = ()
    templatable: bool = True


@dataclass(frozen=True)
class ElementDef:
    """Complete definition of an HTML5 element for generation purposes."""

    tag: str
    categories: ContentCategory = ContentCategory.NONE
    permitted_content: ContentCategory = ContentCategory.NONE
    permitted_children: frozenset[str] = frozenset()
    forbidden_children: frozenset[str] = frozenset()
    void: bool = False
    accepts_text: bool = True
    strict_children: bool = False
    attributes: tuple[AttributeSpec, ...] = ()
    min_children: int = 0


# ---------------------------------------------------------------------------
# Shortcuts
# ---------------------------------------------------------------------------
_F = ContentCategory.FLOW
_P = ContentCategory.PHRASING
_S = ContentCategory.SECTIONING
_H = ContentCategory.HEADING
_E = ContentCategory.EMBEDDED
_I = ContentCategory.INTERACTIVE
_M = ContentCategory.METADATA

# ---------------------------------------------------------------------------
# Element registry
# ---------------------------------------------------------------------------

ELEMENTS: dict[str, ElementDef] = {}


def _reg(e: ElementDef) -> None:
    ELEMENTS[e.tag] = e


# -- Document structure (only valid as children of <html> via strict_children) --
_reg(
    ElementDef(
        "html",
        ContentCategory.NONE,
        _M | _F,
        permitted_children=frozenset({"head", "body"}),
        strict_children=True,
        accepts_text=False,
    )
)
_reg(ElementDef("head", ContentCategory.NONE, _M, accepts_text=False))
_reg(ElementDef("body", ContentCategory.NONE, _F))
_reg(ElementDef("title", _M, _P))

# -- Sectioning / flow containers ------------------------------------------
_reg(ElementDef("div", _F, _F))
_reg(ElementDef("section", _F | _S, _F))
_reg(ElementDef("article", _F | _S, _F))
_reg(ElementDef("aside", _F | _S, _F))
_reg(ElementDef("nav", _F | _S, _F))
_reg(ElementDef("header", _F, _F, forbidden_children=frozenset({"header", "footer"})))
_reg(ElementDef("footer", _F, _F, forbidden_children=frozenset({"header", "footer"})))
_reg(ElementDef("main", _F, _F))
_reg(
    ElementDef("address", _F, _P)
)  # HTML5 allows flow; lxml's HTML4 parser only allows phrasing

# -- Headings ---------------------------------------------------------------
for _n in range(1, 7):
    _reg(ElementDef(f"h{_n}", _F | _H, _P))

# -- Phrasing / inline ------------------------------------------------------
_reg(ElementDef("span", _F | _P, _P))
_reg(
    ElementDef(
        "a",
        _F | _P | _I,
        _P,  # transparent in spec; use _P (safe conservative)
        forbidden_children=frozenset({"a"}),
        attributes=(AttributeSpec("href"),),
    )
)
_reg(ElementDef("strong", _F | _P, _P))
_reg(ElementDef("em", _F | _P, _P))
_reg(ElementDef("code", _F | _P, _P))
_reg(ElementDef("b", _F | _P, _P))
_reg(ElementDef("i", _F | _P, _P))
_reg(ElementDef("small", _F | _P, _P))
_reg(ElementDef("mark", _F | _P, _P))
_reg(ElementDef("time", _F | _P, _P, attributes=(AttributeSpec("datetime"),)))
_reg(ElementDef("abbr", _F | _P, _P))
_reg(ElementDef("cite", _F | _P, _P))
_reg(ElementDef("q", _F | _P, _P))
_reg(ElementDef("s", _F | _P, _P))
_reg(ElementDef("u", _F | _P, _P))
_reg(ElementDef("sub", _F | _P, _P))
_reg(ElementDef("sup", _F | _P, _P))
_reg(ElementDef("data", _F | _P, _P, attributes=(AttributeSpec("value"),)))

# -- Text blocks ------------------------------------------------------------
_reg(ElementDef("p", _F, _P))
_reg(ElementDef("blockquote", _F, _F))
_reg(ElementDef("pre", _F, _P))
_reg(ElementDef("figure", _F, _F, permitted_children=frozenset({"figcaption"})))
_reg(ElementDef("figcaption", ContentCategory.NONE, _F))

# -- Lists ------------------------------------------------------------------
_reg(
    ElementDef(
        "ul",
        _F,
        ContentCategory.NONE,
        permitted_children=frozenset({"li"}),
        strict_children=True,
        accepts_text=False,
        min_children=1,
    )
)
_reg(
    ElementDef(
        "ol",
        _F,
        ContentCategory.NONE,
        permitted_children=frozenset({"li"}),
        strict_children=True,
        accepts_text=False,
        min_children=1,
    )
)
_reg(ElementDef("li", ContentCategory.NONE, _F))
_reg(
    ElementDef(
        "dl",
        _F,
        ContentCategory.NONE,
        permitted_children=frozenset({"dt", "dd"}),
        strict_children=True,
        accepts_text=False,
        min_children=1,
    )
)
_reg(ElementDef("dt", ContentCategory.NONE, _P))
_reg(ElementDef("dd", ContentCategory.NONE, _F))

# -- Tables -----------------------------------------------------------------
_reg(
    ElementDef(
        "table",
        _F,
        ContentCategory.NONE,
        permitted_children=frozenset({"caption", "thead", "tbody", "tfoot", "tr"}),
        strict_children=True,
        accepts_text=False,
        min_children=1,
    )
)
_reg(ElementDef("caption", ContentCategory.NONE, _F))
_reg(
    ElementDef(
        "thead",
        ContentCategory.NONE,
        ContentCategory.NONE,
        permitted_children=frozenset({"tr"}),
        strict_children=True,
        accepts_text=False,
        min_children=1,
    )
)
_reg(
    ElementDef(
        "tbody",
        ContentCategory.NONE,
        ContentCategory.NONE,
        permitted_children=frozenset({"tr"}),
        strict_children=True,
        accepts_text=False,
        min_children=1,
    )
)
_reg(
    ElementDef(
        "tfoot",
        ContentCategory.NONE,
        ContentCategory.NONE,
        permitted_children=frozenset({"tr"}),
        strict_children=True,
        accepts_text=False,
        min_children=1,
    )
)
_reg(
    ElementDef(
        "tr",
        ContentCategory.NONE,
        ContentCategory.NONE,
        permitted_children=frozenset({"td", "th"}),
        strict_children=True,
        accepts_text=False,
        min_children=1,
    )
)
_reg(ElementDef("td", ContentCategory.NONE, _F))
_reg(ElementDef("th", ContentCategory.NONE, _F))

# -- Void elements ----------------------------------------------------------
_reg(ElementDef("br", _F | _P, ContentCategory.NONE, void=True, accepts_text=False))
_reg(ElementDef("hr", _F, ContentCategory.NONE, void=True, accepts_text=False))
_reg(
    ElementDef(
        "img",
        _F | _P | _E,
        ContentCategory.NONE,
        void=True,
        accepts_text=False,
        attributes=(
            AttributeSpec("src", required=True),
            AttributeSpec("alt", required=True),
        ),
    )
)
_reg(
    ElementDef(
        "input",
        _F | _P | _I,
        ContentCategory.NONE,
        void=True,
        accepts_text=False,
        attributes=(
            AttributeSpec(
                "type",
                values=("text", "number", "email", "hidden", "checkbox", "radio"),
            ),
            AttributeSpec("name"),
            AttributeSpec("value"),
        ),
    )
)
_reg(
    ElementDef(
        "meta",
        _M,
        ContentCategory.NONE,
        void=True,
        accepts_text=False,
        attributes=(AttributeSpec("name"), AttributeSpec("content")),
    )
)
_reg(
    ElementDef(
        "link",
        _M,
        ContentCategory.NONE,
        void=True,
        accepts_text=False,
        attributes=(
            AttributeSpec("rel", required=True),
            AttributeSpec("href", required=True),
        ),
    )
)
_reg(ElementDef("wbr", _F | _P, ContentCategory.NONE, void=True, accepts_text=False))

# -- Forms ------------------------------------------------------------------
_reg(ElementDef("form", _F | _I, _F, forbidden_children=frozenset({"form"})))
_reg(ElementDef("label", _F | _P | _I, _P, attributes=(AttributeSpec("for"),)))
_reg(
    ElementDef(
        "select",
        _F | _P | _I,
        ContentCategory.NONE,
        permitted_children=frozenset({"option", "optgroup"}),
        strict_children=True,
        accepts_text=False,
        min_children=1,
    )
)
_reg(
    ElementDef(
        "option",
        ContentCategory.NONE,
        ContentCategory.NONE,
        attributes=(AttributeSpec("value"),),
    )
)
_reg(
    ElementDef(
        "optgroup",
        ContentCategory.NONE,
        ContentCategory.NONE,
        permitted_children=frozenset({"option"}),
        strict_children=True,
        accepts_text=False,
        attributes=(AttributeSpec("label", required=True),),
    )
)
_reg(
    ElementDef(
        "textarea",
        _F | _P | _I,
        ContentCategory.NONE,
        attributes=(AttributeSpec("name"),),
    )
)
_reg(
    ElementDef(
        "button",
        _F | _P | _I,
        _P,
        forbidden_children=frozenset({"a", "button", "input", "select", "textarea"}),
    )
)
_reg(ElementDef("fieldset", _F, _F, permitted_children=frozenset({"legend"})))
_reg(ElementDef("legend", ContentCategory.NONE, _P))

# -- Details / summary ------------------------------------------------------
_reg(ElementDef("details", _F | _I, _F, permitted_children=frozenset({"summary"})))
_reg(ElementDef("summary", ContentCategory.NONE, _P))

# -- Misc flow --------------------------------------------------------------
_reg(ElementDef("dialog", _F, _F))


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def children_allowed(parent: ElementDef, child_tag: str) -> bool:
    """Return True if ``child_tag`` is a valid child of ``parent``."""
    if child_tag not in ELEMENTS:
        return False
    child_def = ELEMENTS[child_tag]

    if child_tag in parent.forbidden_children:
        return False

    if parent.strict_children:
        return child_tag in parent.permitted_children

    # Category-based check: the child must belong to at least one category
    # that the parent permits.
    if parent.permitted_content and (child_def.categories & parent.permitted_content):
        return True

    # Explicit permitted children supplement category checks.
    if child_tag in parent.permitted_children:
        return True

    return False


def get_valid_children(parent_tag: str) -> list[str]:
    """Return sorted list of tags that are valid children of ``parent_tag``."""
    if parent_tag not in ELEMENTS:
        return []
    parent = ELEMENTS[parent_tag]
    if parent.void:
        return []
    return sorted(tag for tag in ELEMENTS if children_allowed(parent, tag))
