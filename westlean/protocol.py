"""Standard interface for template inference algorithms."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from lxml import etree


@runtime_checkable
class InferredTemplate(Protocol):
    """Result of template inference.

    Implementations must provide both behavioral extraction and structural
    classification of fixed vs. variable positions, plus JSON serialization.
    """

    def extract(self, page: etree._Element) -> dict[str, Any] | None:
        """Extract variable data from a page.

        Returns a dict mapping algorithm-chosen variable names to extracted
        values, or ``None`` if the page does not match this template.

        Values may be:
        * ``str`` — text content or attribute values
        * ``list[dict[str, Any]]`` — repeated sections (loops)
        """
        ...

    def fixed_mask(self, page: etree._Element) -> dict[str, bool] | None:
        """Classify each text/attribute position in *page* as fixed or variable.

        Returns a dict mapping canonical position keys to booleans
        (``True`` = fixed template content, ``False`` = variable data),
        or ``None`` if the page does not match.

        Position key format (child-index path from root)::

            "0/1/text"   — text content of root[0][1]
            "0/1/@href"  — href attribute of root[0][1]
            "0/1/tail"   — tail text after root[0][1]
            "text"       — text content of root itself
            "@class"     — attribute on root itself
        """
        ...

    def serialize(self) -> dict[str, Any]:
        """Serialize this template to a JSON-friendly dict.

        The dict includes an ``"algorithm"`` key for dispatch during
        restoration.
        """
        ...

    def get_relax_ng(self) -> str:
        """Generate a RELAX NG schema (XML syntax) for this template.

        The schema should validate any page that this template would
        accept via ``extract()``.

        Returns an XML string containing a ``<grammar>`` element in the
        RELAX NG namespace.
        """
        ...


@runtime_checkable
class TemplateInferer(Protocol):
    """Protocol that all template inference implementations must satisfy."""

    def infer(self, pages: Sequence[etree._Element]) -> InferredTemplate:
        """Infer a template from a sequence of example DOM trees.

        Implementations may assume ``len(pages) >= 2``.
        """
        ...


# ---------------------------------------------------------------------------
# Shared empty template
# ---------------------------------------------------------------------------


class EmptyTemplate:
    """Placeholder that matches nothing (pages were structurally incompatible)."""

    def extract(self, page: etree._Element) -> dict[str, Any] | None:
        return None

    def fixed_mask(self, page: etree._Element) -> dict[str, bool] | None:
        return None

    def serialize(self) -> dict[str, Any]:
        return {"algorithm": "empty"}

    def get_relax_ng(self) -> str:
        return ""


# ---------------------------------------------------------------------------
# Restoration dispatch
# ---------------------------------------------------------------------------


def restore_template(data: dict[str, Any]) -> InferredTemplate:
    """Restore an :class:`InferredTemplate` from a serialized dict.

    Dispatches on ``data["algorithm"]`` to the appropriate Pydantic model
    and reconstruction logic.
    """
    from pydantic import TypeAdapter

    from westlean.serialization import TemplateModel

    adapter = TypeAdapter(TemplateModel)
    model = adapter.validate_python(data)

    from westlean.serialization import EmptyTemplateModel

    if isinstance(model, EmptyTemplateModel):
        return EmptyTemplate()

    return model.to_internal()  # type: ignore[union-attr]
