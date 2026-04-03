"""Tracing infrastructure for recording algorithm intermediate steps.

The tracer is opt-in via a context manager.  Algorithm subclasses emit
:class:`TraceStep` events during execution; the :class:`Tracer` collects
them.  The resulting trace is a JSON-serializable list of step dicts,
suitable for replay in a browser-based D3 visualization.

Usage::

    from westlean.tracer import tracing
    from westlean.algorithms.tracing_exalg import TracingExalgInferer

    with tracing() as tracer:
        inferer = TracingExalgInferer()
        template = inferer.infer(pages)

    steps = tracer.to_json()  # list[dict] ready for JSON serialization
"""

from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Generator


@dataclass
class TraceStep:
    """A single recorded step in an algorithm trace."""

    algorithm: str
    phase: str
    step_index: int
    description: str
    data: dict[str, Any]


class Tracer:
    """Collects :class:`TraceStep` events during algorithm execution."""

    def __init__(self) -> None:
        self.steps: list[TraceStep] = []

    def emit(
        self,
        algorithm: str,
        phase: str,
        description: str,
        data: dict[str, Any],
    ) -> None:
        step = TraceStep(
            algorithm=algorithm,
            phase=phase,
            step_index=len(self.steps),
            description=description,
            data=data,
        )
        self.steps.append(step)

    def to_json(self) -> list[dict[str, Any]]:
        return [asdict(s) for s in self.steps]


_active_tracer: ContextVar[Tracer | None] = ContextVar("_active_tracer", default=None)


@contextmanager
def tracing() -> Generator[Tracer]:
    """Context manager that activates a :class:`Tracer` for the duration."""
    tracer = Tracer()
    token = _active_tracer.set(tracer)
    try:
        yield tracer
    finally:
        _active_tracer.reset(token)


def get_tracer() -> Tracer | None:
    """Return the active tracer, or ``None`` if tracing is not active."""
    return _active_tracer.get()
