"""Tests for the tracing infrastructure and TracingExalgInferer."""

from __future__ import annotations

import json

from hypothesis import HealthCheck
from lxml.html import fragment_fromstring

from westlean.tracer import Tracer, tracing, get_tracer
from westlean.algorithms.tracing_exalg import TracingExalgInferer
from westlean.algorithms.tracing_anti_unification import TracingAntiUnificationInferer
from westlean.algorithms.tracing_fivatech import TracingFiVaTechInferer
from westlean.algorithms.tracing_roadrunner import TracingRoadRunnerInferer
from westlean.algorithms.tracing_tree_automata import TracingKTestableInferer
from westlean.harness import MAX_DEPTH, InferenceTestSuite
from westlean.strategies import template_and_schema


_SETTINGS = dict(deadline=None, suppress_health_check=[HealthCheck.differing_executors])


# ---------------------------------------------------------------------------
# Core tracer tests
# ---------------------------------------------------------------------------


class TestTracer:
    def test_emit_collects_steps(self) -> None:
        tracer = Tracer()
        tracer.emit("test", "phase1", "doing something", {"x": 1})
        tracer.emit("test", "phase2", "doing more", {"y": 2})
        assert len(tracer.steps) == 2
        assert tracer.steps[0].step_index == 0
        assert tracer.steps[1].step_index == 1

    def test_to_json_is_serializable(self) -> None:
        tracer = Tracer()
        tracer.emit("test", "p", "desc", {"nested": {"a": [1, 2]}})
        result = tracer.to_json()
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert len(parsed) == 1
        assert parsed[0]["algorithm"] == "test"

    def test_context_manager(self) -> None:
        assert get_tracer() is None
        with tracing() as tracer:
            assert get_tracer() is tracer
            tracer.emit("test", "p", "desc", {})
        assert get_tracer() is None
        assert len(tracer.steps) == 1

    def test_nested_context_restores(self) -> None:
        with tracing() as outer:
            outer.emit("outer", "p", "d", {})
            with tracing() as inner:
                inner.emit("inner", "p", "d", {})
                assert get_tracer() is inner
            assert get_tracer() is outer
        assert get_tracer() is None
        assert len(outer.steps) == 1
        assert len(inner.steps) == 1


# ---------------------------------------------------------------------------
# TracingExalgInferer tests
# ---------------------------------------------------------------------------


class TestTracingExalg:
    def test_emits_all_phases(self) -> None:
        pages = [
            fragment_fromstring("<div><h1>Title</h1><p>Hello Alice</p></div>"),
            fragment_fromstring("<div><h1>Title</h1><p>Hello Bob</p></div>"),
        ]
        with tracing() as tracer:
            inferer = TracingExalgInferer()
            inferer.infer(pages)

        phases = [s.phase for s in tracer.steps]
        assert phases == [
            "tokenization",
            "equivalence_classes",
            "diffformat",
            "handinv",
            "diffeq",
            "skeleton",
            "result",
        ]

        # All steps should be for exalg
        assert all(s.algorithm == "exalg" for s in tracer.steps)

    def test_structure_mismatch_emits_empty(self) -> None:
        pages = [
            fragment_fromstring("<div><p>Hello</p></div>"),
            fragment_fromstring("<section><span>World</span></section>"),
        ]
        with tracing() as tracer:
            inferer = TracingExalgInferer()
            inferer.infer(pages)

        # Different root tags → EmptyTemplate before any analysis
        assert len(tracer.steps) == 0

    def test_trace_is_json_serializable(self) -> None:
        pages = [
            fragment_fromstring("<div><p>A</p></div>"),
            fragment_fromstring("<div><p>B</p></div>"),
        ]
        with tracing() as tracer:
            TracingExalgInferer().infer(pages)

        json_str = json.dumps(tracer.to_json())
        parsed = json.loads(json_str)
        assert len(parsed) == 7

    def test_without_tracer_still_works(self) -> None:
        """TracingExalgInferer works fine without an active tracer."""
        pages = [
            fragment_fromstring("<div><p>Hello</p></div>"),
            fragment_fromstring("<div><p>World</p></div>"),
        ]
        inferer = TracingExalgInferer()
        template = inferer.infer(pages)
        result = template.extract(fragment_fromstring("<div><p>Test</p></div>"))
        assert result is not None

    def test_equivalence_class_data(self) -> None:
        pages = [
            fragment_fromstring("<div><p>Same</p><span>A</span></div>"),
            fragment_fromstring("<div><p>Same</p><span>B</span></div>"),
        ]
        with tracing() as tracer:
            TracingExalgInferer().infer(pages)

        eq_step = next(s for s in tracer.steps if s.phase == "equivalence_classes")
        # Should have template constants (including "Same" text)
        assert eq_step.data["template_constants"] > 0


# ---------------------------------------------------------------------------
# TracingExalgInferer correctness (reuses InferenceTestSuite)
# ---------------------------------------------------------------------------


class TestTracingExalgCorrectness(InferenceTestSuite):
    """Verify TracingExalgInferer produces identical results to ExalgInferer."""

    def make_inferer(self):
        return TracingExalgInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH, allow_loops=False, allow_conditionals=False
        )


class TestTracingAntiUnificationCorrectness(InferenceTestSuite):
    def make_inferer(self):
        return TracingAntiUnificationInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH, allow_loops=False, allow_conditionals=False
        )


class TestTracingFiVaTechCorrectness(InferenceTestSuite):
    def make_inferer(self):
        return TracingFiVaTechInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH, allow_loops=False, allow_conditionals=False
        )


class TestTracingRoadRunnerCorrectness(InferenceTestSuite):
    def make_inferer(self):
        return TracingRoadRunnerInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH, allow_loops=False, allow_conditionals=False
        )


class TestTracingKTestableCorrectness(InferenceTestSuite):
    def make_inferer(self):
        return TracingKTestableInferer()

    def _template_strategy(self):
        return template_and_schema(
            max_depth=MAX_DEPTH, allow_loops=False, allow_conditionals=False
        )
