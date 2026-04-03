"""Hypothesis strategies for generating data from schema field types.

Separated from :mod:`data_schema` so that ``hypothesis`` is only imported
by test infrastructure, not by the inference/tracing code path.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any

import hypothesis.strategies as st

from westlean.data_schema import (
    BoolField,
    IntField,
    ListField,
    ObjectField,
    SchemaField,
    StringField,
    UrlField,
)


@singledispatch
def field_strategy(field: SchemaField) -> st.SearchStrategy:
    """Return a Hypothesis strategy for the given schema field."""
    raise TypeError(f"No strategy for {type(field)}")


@field_strategy.register
def _string(field: StringField) -> st.SearchStrategy[str]:
    alphabet = st.characters(
        whitelist_categories=("L", "N", "Zs"),
        blacklist_characters="<>&\"'",
    )
    return st.text(alphabet, min_size=field.min_length, max_size=field.max_length)


@field_strategy.register
def _int(field: IntField) -> st.SearchStrategy[int]:
    return st.integers(min_value=field.min_value, max_value=field.max_value)


@field_strategy.register
def _bool(field: BoolField) -> st.SearchStrategy[bool]:
    return st.booleans()


@field_strategy.register
def _url(field: UrlField) -> st.SearchStrategy[str]:
    return st.builds(
        lambda proto, domain, path: f"{proto}://{domain}/{path}",
        st.sampled_from(["http", "https"]),
        st.from_regex(r"[a-z]{3,10}\.[a-z]{2,4}", fullmatch=True),
        st.from_regex(r"[a-z0-9/]{0,20}", fullmatch=True),
    )


@field_strategy.register
def _list(field: ListField) -> st.SearchStrategy[list[dict[str, Any]]]:
    return st.lists(
        field_strategy(field.item_schema),
        min_size=field.min_length,
        max_size=field.max_length,
    )


@field_strategy.register
def _object(field: ObjectField) -> st.SearchStrategy[dict[str, Any]]:
    if not field.fields:
        return st.just({})
    return st.fixed_dictionaries(
        {name: field_strategy(f) for name, f in field.fields.items()}
    )
