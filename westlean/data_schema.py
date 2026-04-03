from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class StringField:
    """A string value.  Generated as text safe for embedding in HTML."""

    min_length: int = 1
    max_length: int = 50


@dataclass(frozen=True)
class IntField:
    """An integer value, rendered as its string representation."""

    min_value: int = 0
    max_value: int = 10_000


@dataclass(frozen=True)
class BoolField:
    """A boolean value, used for conditional blocks."""


@dataclass(frozen=True)
class UrlField:
    """A URL value for href/src attributes."""


@dataclass(frozen=True)
class ListField:
    """A list of items, each conforming to ``item_schema``."""

    item_schema: ObjectField
    min_length: int = 0
    max_length: int = 5


@dataclass(frozen=True)
class ObjectField:
    """A dict of named fields.  The top-level data schema is also an ObjectField."""

    fields: dict[str, SchemaField] = field(default_factory=dict)


# Union of every field type.
SchemaField = Union[StringField, IntField, BoolField, UrlField, ListField, ObjectField]

# Convenience alias: the top-level schema is always an ObjectField.
DataSchema = ObjectField
