"""Round-trip serialization tests for all template inference algorithms."""

from __future__ import annotations

import json

import hypothesis.strategies as st
from hypothesis import HealthCheck, given, settings

from westlean.evaluation import parse_html
from westlean.harness import MAX_DEPTH, _draw_varied_training, _render_pages
from westlean.protocol import EmptyTemplate, restore_template
from westlean.field_strategies import field_strategy
from westlean.strategies import template_and_schema

from westlean.algorithms.exalg import ExalgInferer
from westlean.algorithms.anti_unification import AntiUnificationInferer
from westlean.algorithms.fivatech import FiVaTechInferer
from westlean.algorithms.roadrunner import RoadRunnerInferer
from westlean.algorithms.tree_automata import KTestableInferer
from westlean.renderer import render


_SETTINGS = dict(deadline=None, suppress_health_check=[HealthCheck.differing_executors])


def _fixed_structure_strategy():
    return template_and_schema(
        max_depth=MAX_DEPTH, allow_loops=False, allow_conditionals=False
    )


# ---------------------------------------------------------------------------
# Round-trip helpers
# ---------------------------------------------------------------------------


def _roundtrip_extract(inferer, template, schema, train_data, test_datum):
    """Infer, serialize, restore, and verify extract() matches."""
    train_pages = _render_pages(template, schema, train_data)
    inferred = inferer.infer(train_pages)

    # Serialize through JSON (proves it's JSON-safe)
    serialized = inferred.serialize()
    json_str = json.dumps(serialized)
    restored_data = json.loads(json_str)

    # Restore via dispatch
    restored = restore_template(restored_data)

    # Both should extract the same thing
    test_page = parse_html(render(template, test_datum))
    original_result = inferred.extract(test_page)
    restored_result = restored.extract(test_page)

    assert (original_result is None) == (restored_result is None), (
        "Original and restored templates disagree on recognition"
    )
    if original_result is not None:
        assert original_result == restored_result, (
            "Extracted values differ after round-trip"
        )

    # fixed_mask should also match
    original_mask = inferred.fixed_mask(test_page)
    restored_mask = restored.fixed_mask(test_page)
    assert original_mask == restored_mask, "fixed_mask differs after round-trip"


# ---------------------------------------------------------------------------
# ExAlg
# ---------------------------------------------------------------------------


class TestExalgSerialization:
    @given(data=st.data())
    @settings(**_SETTINGS)
    def test_roundtrip(self, data: st.DataObject) -> None:
        template, schema = data.draw(_fixed_structure_strategy())
        train_data = _draw_varied_training(data, schema)
        test_datum = data.draw(field_strategy(schema))
        _roundtrip_extract(ExalgInferer(), template, schema, train_data, test_datum)

    def test_serialize_is_json(self) -> None:
        from lxml.html import fragment_fromstring

        pages = [
            fragment_fromstring("<div><p>Hello</p></div>"),
            fragment_fromstring("<div><p>World</p></div>"),
        ]
        tpl = ExalgInferer().infer(pages)
        serialized = tpl.serialize()
        assert serialized["algorithm"] == "exalg"
        # Must be JSON-serializable
        json.dumps(serialized)


# ---------------------------------------------------------------------------
# Anti-Unification
# ---------------------------------------------------------------------------


class TestAntiUnificationSerialization:
    @given(data=st.data())
    @settings(**_SETTINGS)
    def test_roundtrip(self, data: st.DataObject) -> None:
        template, schema = data.draw(_fixed_structure_strategy())
        train_data = _draw_varied_training(data, schema)
        test_datum = data.draw(field_strategy(schema))
        _roundtrip_extract(
            AntiUnificationInferer(), template, schema, train_data, test_datum
        )


# ---------------------------------------------------------------------------
# FiVaTech
# ---------------------------------------------------------------------------


class TestFiVaTechSerialization:
    @given(data=st.data())
    @settings(**_SETTINGS)
    def test_roundtrip(self, data: st.DataObject) -> None:
        template, schema = data.draw(_fixed_structure_strategy())
        train_data = _draw_varied_training(data, schema)
        test_datum = data.draw(field_strategy(schema))
        _roundtrip_extract(FiVaTechInferer(), template, schema, train_data, test_datum)


# ---------------------------------------------------------------------------
# RoadRunner
# ---------------------------------------------------------------------------


class TestRoadRunnerSerialization:
    @given(data=st.data())
    @settings(**_SETTINGS)
    def test_roundtrip(self, data: st.DataObject) -> None:
        template, schema = data.draw(_fixed_structure_strategy())
        train_data = _draw_varied_training(data, schema)
        test_datum = data.draw(field_strategy(schema))
        _roundtrip_extract(
            RoadRunnerInferer(), template, schema, train_data, test_datum
        )


# ---------------------------------------------------------------------------
# k-Testable
# ---------------------------------------------------------------------------


class TestKTestableSerialization:
    @given(data=st.data())
    @settings(**_SETTINGS)
    def test_roundtrip_k2(self, data: st.DataObject) -> None:
        template, schema = data.draw(_fixed_structure_strategy())
        train_data = _draw_varied_training(data, schema)
        test_datum = data.draw(field_strategy(schema))
        _roundtrip_extract(
            KTestableInferer(k=2), template, schema, train_data, test_datum
        )

    @given(data=st.data())
    @settings(**_SETTINGS)
    def test_roundtrip_k3(self, data: st.DataObject) -> None:
        template, schema = data.draw(_fixed_structure_strategy())
        train_data = _draw_varied_training(data, schema)
        test_datum = data.draw(field_strategy(schema))
        _roundtrip_extract(
            KTestableInferer(k=3), template, schema, train_data, test_datum
        )


# ---------------------------------------------------------------------------
# Empty template
# ---------------------------------------------------------------------------


class TestEmptyTemplateSerialization:
    def test_roundtrip(self) -> None:
        tpl = EmptyTemplate()
        serialized = tpl.serialize()
        assert serialized == {"algorithm": "empty"}

        restored = restore_template(serialized)
        assert isinstance(restored, EmptyTemplate)

    def test_json_roundtrip(self) -> None:
        serialized = EmptyTemplate().serialize()
        json_str = json.dumps(serialized)
        restored = restore_template(json.loads(json_str))
        assert isinstance(restored, EmptyTemplate)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_exalg_dispatch(self) -> None:
        from lxml.html import fragment_fromstring

        pages = [
            fragment_fromstring("<div><p>A</p></div>"),
            fragment_fromstring("<div><p>B</p></div>"),
        ]
        tpl = ExalgInferer().infer(pages)
        data = tpl.serialize()
        assert data["algorithm"] == "exalg"
        restored = restore_template(data)
        # Should produce same results
        test = fragment_fromstring("<div><p>C</p></div>")
        assert tpl.extract(test) == restored.extract(test)
