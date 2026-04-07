"""Standardized test suite for template inference algorithms.

Subclass :class:`InferenceTestSuite` and implement :meth:`make_inferer`
to get property-based tests for free::

    class TestMyAlgorithm(InferenceTestSuite):
        def make_inferer(self):
            return MyInferer()
"""

from __future__ import annotations


import hypothesis.strategies as st
from hypothesis import HealthCheck, given, settings, assume
from lxml import etree

from westlean.evaluation import (
    flatten_values,
    ground_truth_mask,
    parse_html,
)
from westlean.protocol import TemplateInferer
from westlean.renderer import render
from westlean.strategies import template_and_schema
from westlean.field_strategies import field_strategy
from westlean.template_ast import Element
from westlean.data_schema import (
    BoolField,
    DataSchema,
    IntField,
    ListField,
    ObjectField,
    StringField,
    UrlField,
)

import os

_SETTINGS = dict(deadline=None, suppress_health_check=[HealthCheck.differing_executors])
MAX_DEPTH = int(os.getenv("WESTLEAN_MAX_DEPTH", "3"))


def _minimal_data(schema: DataSchema) -> dict:
    """Generate minimal data for a schema: empty lists, short strings."""
    return _minimal_fields(schema)


def _single_item_data(schema: DataSchema) -> dict:
    """Generate data with all lists having exactly 1 item.

    Loop templates accept any item count.  Checking only the 0-item
    rendering (``_minimal_data``) misses overlap when Template B produces
    a page that looks like a 1-item rendering of Template A.
    """
    return _single_item_fields(schema)


def _minimal_fields(obj: ObjectField) -> dict:
    result: dict = {}
    for name, field_type in obj.fields.items():
        if isinstance(field_type, StringField):
            result[name] = "x"
        elif isinstance(field_type, IntField):
            result[name] = 0
        elif isinstance(field_type, BoolField):
            result[name] = True
        elif isinstance(field_type, UrlField):
            result[name] = "http://x"
        elif isinstance(field_type, ListField):
            result[name] = []
        elif isinstance(field_type, ObjectField):
            result[name] = _minimal_fields(field_type)
    return result


def _single_item_fields(obj: ObjectField) -> dict:
    result: dict = {}
    for name, field_type in obj.fields.items():
        if isinstance(field_type, StringField):
            result[name] = "x"
        elif isinstance(field_type, IntField):
            result[name] = 0
        elif isinstance(field_type, BoolField):
            result[name] = True
        elif isinstance(field_type, UrlField):
            result[name] = "http://x"
        elif isinstance(field_type, ListField):
            result[name] = [_single_item_fields(field_type.item_schema)]
        elif isinstance(field_type, ObjectField):
            result[name] = _single_item_fields(field_type)
    return result


def _empty_values_data(schema: DataSchema) -> dict:
    """Generate data with empty strings and single-item lists.

    Covers the case where Template B has no text content but Template A
    has a variable that happens to be empty — the rendered DOM is
    identical so the negative page is a valid match.
    """
    return _empty_values_fields(schema)


def _empty_values_fields(obj: ObjectField) -> dict:
    result: dict = {}
    for name, field_type in obj.fields.items():
        if isinstance(field_type, StringField):
            result[name] = ""
        elif isinstance(field_type, IntField):
            result[name] = 0
        elif isinstance(field_type, BoolField):
            result[name] = True
        elif isinstance(field_type, UrlField):
            result[name] = ""
        elif isinstance(field_type, ListField):
            result[name] = [_empty_values_fields(field_type.item_schema)]
        elif isinstance(field_type, ObjectField):
            result[name] = _empty_values_fields(field_type)
    return result


def _tree_signature(elem) -> tuple:
    """Compute a structural signature for DOM tree comparison."""
    return (
        elem.tag,
        tuple(sorted(elem.attrib)),
        bool(elem.text),
        tuple(_tree_signature(c) for c in elem),
    )


def _template_signature(node) -> tuple:
    """Compute a structural signature for template AST comparison.

    Unlike _tree_signature which operates on rendered DOM, this compares
    the template structure directly — so two identical templates with
    different data won't appear as different templates.
    """
    from westlean.template_ast import (
        Element as TElement,
        LoopBlock,
        ConditionalBlock,
        TextNode,
        TemplateVar,
    )

    if isinstance(node, TElement):
        return (
            "elem",
            node.tag,
            tuple(sorted(node.attributes)),
            tuple(_template_signature(c) for c in node.children),
        )
    if isinstance(node, LoopBlock):
        return ("loop", tuple(_template_signature(c) for c in node.children))
    if isinstance(node, ConditionalBlock):
        return (
            "cond",
            tuple(_template_signature(c) for c in node.children),
            tuple(_template_signature(c) for c in node.else_children),
        )
    if isinstance(node, TextNode):
        return ("text",)
    if isinstance(node, TemplateVar):
        return ("var",)
    return ("other",)


def _render_pages(template: Element, schema: DataSchema, data_list: list[dict]) -> list:
    """Render and parse a list of data dicts into lxml element trees."""
    return [parse_html(render(template, d)) for d in data_list]


def _draw_varied_training(
    data: st.DataObject, schema: DataSchema, n: int = 5
) -> list[dict]:
    """Draw training data ensuring every schema field has variation.

    Without variation, diff-based algorithms can't detect variable positions.
    For list fields, also requires at least two different non-zero lengths
    so loop-capable algorithms can detect the repeating pattern.
    """

    train_data = [data.draw(field_strategy(schema)) for _ in range(n)]
    _check_field_variation(schema, train_data)
    # Reject if all data dicts are identical — algorithms can't learn
    # anything from pages that are all the same.
    assume(len({str(d) for d in train_data}) >= 2)
    return train_data


def _check_field_variation(
    schema: DataSchema,
    train_data: list[dict],
) -> None:
    """Recursively check that all fields (including nested list items) vary."""
    from westlean.data_schema import ListField

    for field_name, field_type in schema.fields.items():
        vals = {str(d.get(field_name)) for d in train_data if field_name in d}
        assume(len(vals) >= 2)
        if isinstance(field_type, ListField):
            lengths = {len(d.get(field_name, [])) for d in train_data}
            # Need 2+ different lengths with at least one >= 2
            # so loop detection can distinguish loops from conditionals
            assume(len(lengths) >= 2 and any(n >= 2 for n in lengths))
            # Also need variation within item fields so the repeating
            # child template has variable (not fixed) slots
            for (
                item_field_name,
                item_field_type,
            ) in field_type.item_schema.fields.items():
                all_item_vals: set[str] = set()
                for d in train_data:
                    for item in d.get(field_name, []):
                        if isinstance(item, dict) and item_field_name in item:
                            all_item_vals.add(str(item[item_field_name]))
                assume(len(all_item_vals) >= 2)
                # Recurse into nested list fields
                if isinstance(item_field_type, ListField):
                    nested_items: list[dict] = []
                    for d in train_data:
                        for item in d.get(field_name, []):
                            if isinstance(item, dict):
                                nested_items.append(item)
                    if nested_items:
                        _check_field_variation(
                            field_type.item_schema,
                            nested_items,
                        )


class InferenceTestSuite:
    """Mixin providing standardized property-based tests.

    Subclasses must override :meth:`make_inferer`.  All ``test_*`` methods
    are discovered by pytest automatically.

    Override :meth:`_template_strategy` to restrict the kinds of templates
    generated (e.g. disable loops for algorithms that can't handle them).
    """

    def make_inferer(self) -> TemplateInferer:
        raise NotImplementedError

    def _template_strategy(self):
        """Return the strategy used to generate (template, schema) pairs.

        Tree depth is controlled by the ``WESTLEAN_MAX_DEPTH`` env var
        (default 3).
        """
        return template_and_schema(max_depth=MAX_DEPTH)

    # ------------------------------------------------------------------
    # 1. Recognition: pages from the same template should be recognised
    # ------------------------------------------------------------------

    @given(data=st.data())
    @settings(**_SETTINGS)
    def test_recognizes_matching_pages(self, data: st.DataObject) -> None:
        """Infer from N pages, then extract() on a new page should not be None."""
        template, schema = data.draw(self._template_strategy())
        train_data = _draw_varied_training(data, schema)
        test_datum = data.draw(field_strategy(schema))

        inferer = self.make_inferer()
        train_pages = _render_pages(template, schema, train_data)
        inferred = inferer.infer(train_pages)

        test_page = parse_html(render(template, test_datum))
        result = inferred.extract(test_page)
        assert result is not None, (
            "Inferred template failed to recognise a matching page"
        )

    # ------------------------------------------------------------------
    # 2. Discrimination: pages from a different template should be rejected
    # ------------------------------------------------------------------

    @given(data=st.data())
    @settings(**_SETTINGS)
    def test_discriminates_different_templates(self, data: st.DataObject) -> None:
        """Infer from template A, extract() on template B should return None."""
        template_a, schema_a = data.draw(self._template_strategy())
        template_b, schema_b = data.draw(self._template_strategy())

        # Ensure the two templates are structurally different — both at the
        # template AST level (catches identical templates with different data)
        # and at the rendered DOM level (catches loop-induced variation).
        assume(_template_signature(template_a) != _template_signature(template_b))
        page_a = parse_html(render(template_a, data.draw(field_strategy(schema_a))))
        page_b = parse_html(render(template_b, data.draw(field_strategy(schema_b))))
        assume(_tree_signature(page_a) != _tree_signature(page_b))

        train_data_a = _draw_varied_training(data, schema_a, n=3)
        inferer = self.make_inferer()
        train_pages = _render_pages(template_a, schema_a, train_data_a)
        inferred = inferer.infer(train_pages)

        neg_page = parse_html(render(template_b, data.draw(field_strategy(schema_b))))
        # The negative page must be structurally different from ALL pages
        # template A could produce — including with empty and single-item
        # loops.  A loop template correctly accepts any item count, so
        # renderings with 0 or 1 items are valid matches, not
        # discrimination failures.
        template_a_sigs = {_tree_signature(p) for p in train_pages}
        template_a_sigs.add(
            _tree_signature(parse_html(render(template_a, _minimal_data(schema_a))))
        )
        template_a_sigs.add(
            _tree_signature(parse_html(render(template_a, _single_item_data(schema_a))))
        )
        template_a_sigs.add(
            _tree_signature(
                parse_html(render(template_a, _empty_values_data(schema_a)))
            )
        )
        assume(_tree_signature(neg_page) not in template_a_sigs)
        result = inferred.extract(neg_page)
        assert result is None, (
            "Inferred template matched a page from a different template"
        )

    # ------------------------------------------------------------------
    # 3. Value extraction: extracted values should contain ground truth
    # ------------------------------------------------------------------

    @given(data=st.data())
    @settings(**_SETTINGS)
    def test_extracts_ground_truth_values(self, data: st.DataObject) -> None:
        """Extracted values should contain the actual data values."""
        template, schema = data.draw(self._template_strategy())
        train_data = _draw_varied_training(data, schema)
        test_datum = data.draw(field_strategy(schema))

        truth_vals = flatten_values(test_datum)
        assume(len(truth_vals) > 0)  # skip templates with no extractable values

        inferer = self.make_inferer()
        inferred = inferer.infer(_render_pages(template, schema, train_data))

        test_page = parse_html(render(template, test_datum))
        extraction = inferred.extract(test_page)
        assert extraction is not None, "Failed to recognise matching page"

        extracted_vals = flatten_values(extraction)
        # Use substring containment: lxml may merge adjacent text/var nodes
        # into a single text field, so extracted values may be concatenations.
        for tv in truth_vals:
            assert any(tv in ev for ev in extracted_vals), (
                f"Ground truth value {tv!r} not found in any extracted value"
            )

    # ------------------------------------------------------------------
    # 4. Structural mask: variable positions should be identified
    # ------------------------------------------------------------------

    @given(data=st.data())
    @settings(**_SETTINGS)
    def test_fixed_mask_identifies_variables(self, data: st.DataObject) -> None:
        """fixed_mask() should label variable positions as variable."""
        template, schema = data.draw(self._template_strategy())
        train_data = _draw_varied_training(data, schema)
        test_datum = data.draw(field_strategy(schema))

        inferer = self.make_inferer()
        inferred = inferer.infer(_render_pages(template, schema, train_data))

        test_page = parse_html(render(template, test_datum))
        inf_mask = inferred.fixed_mask(test_page)
        assert inf_mask is not None, "fixed_mask() returned None for matching page"

        truth_mask = ground_truth_mask(template, test_datum)

        # Check: truly variable positions should be labeled variable
        for key, is_fixed in truth_mask.items():
            if not is_fixed and key in inf_mask:
                assert not inf_mask[key], (
                    f"Position '{key}' is variable but was labeled fixed"
                )

    # ------------------------------------------------------------------
    # 5. Generalisation: should work on unseen data
    # ------------------------------------------------------------------

    @given(data=st.data())
    @settings(**_SETTINGS)
    def test_generalizes_to_unseen_data(self, data: st.DataObject) -> None:
        """Template inferred from N pages should recognise M new pages."""
        template, schema = data.draw(self._template_strategy())
        train_data = _draw_varied_training(data, schema)

        inferer = self.make_inferer()
        inferred = inferer.infer(_render_pages(template, schema, train_data))

        for _ in range(5):
            new_datum = data.draw(field_strategy(schema))
            page = parse_html(render(template, new_datum))
            assert inferred.extract(page) is not None, (
                "Inferred template failed to generalise to unseen data"
            )

    # ------------------------------------------------------------------
    # 6. Stability: more examples should not degrade quality
    # ------------------------------------------------------------------

    @given(data=st.data())
    @settings(**_SETTINGS)
    def test_stable_under_more_examples(self, data: st.DataObject) -> None:
        """Adding more training pages should not break recognition."""
        template, schema = data.draw(self._template_strategy())
        small_data = [data.draw(field_strategy(schema)) for _ in range(2)]
        extra_data = [data.draw(field_strategy(schema)) for _ in range(3)]
        test_datum = data.draw(field_strategy(schema))

        inferer = self.make_inferer()

        # Infer from small set
        inferred_small = inferer.infer(_render_pages(template, schema, small_data))
        # Infer from larger set
        inferred_large = inferer.infer(
            _render_pages(template, schema, small_data + extra_data)
        )

        test_page = parse_html(render(template, test_datum))

        small_ok = inferred_small.extract(test_page) is not None
        large_ok = inferred_large.extract(test_page) is not None

        if small_ok:
            assert large_ok, "Recognition degraded after adding more training pages"

    # ------------------------------------------------------------------
    # 7. RELAX NG: schema should validate all training pages
    # ------------------------------------------------------------------

    @given(data=st.data())
    @settings(**_SETTINGS)
    def test_relax_ng_validates_training_pages(self, data: st.DataObject) -> None:
        """RELAX NG schema from inferred template should validate training pages."""
        template, schema = data.draw(self._template_strategy())
        train_data = _draw_varied_training(data, schema)

        inferer = self.make_inferer()
        train_pages = _render_pages(template, schema, train_data)
        inferred = inferer.infer(train_pages)

        rng_str = inferred.get_relax_ng()
        assert rng_str, "get_relax_ng() returned empty string"

        try:
            rng_doc = etree.fromstring(rng_str.encode())
            relaxng = etree.RelaxNG(rng_doc)
        except etree.RelaxNGParseError as exc:
            raise AssertionError(
                f"get_relax_ng() produced invalid schema: {exc}\n{rng_str[:500]}"
            ) from exc

        for i, page in enumerate(train_pages):
            assert relaxng.validate(page), (
                f"Training page {i} failed RELAX NG validation: {relaxng.error_log}"
            )
