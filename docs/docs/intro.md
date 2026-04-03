---
slug: /
sidebar_position: 1
---

# Westlean

Westlean is a research project for studying **HTML template inference algorithms** -- algorithms that take multiple HTML pages generated from the same template and recover the underlying template structure, identifying which parts are fixed (template content) and which are variable (data).

## What is Template Inference?

Given a set of HTML pages that were generated from the same template with different data:

```html
<!-- Page 1 -->
<div><h1>Welcome</h1><p>Hello Alice</p></div>

<!-- Page 2 -->
<div><h1>Welcome</h1><p>Hello Bob</p></div>

<!-- Page 3 -->
<div><h1>Welcome</h1><p>Hello Carol</p></div>
```

A template inference algorithm determines:
- **Fixed positions**: `<h1>Welcome</h1>` -- identical across all pages
- **Variable positions**: The name after "Hello " -- differs between pages

## Algorithms

Westlean implements five template inference algorithms from the academic literature:

| Algorithm | Approach | Paper |
|-----------|----------|-------|
| [ExAlg](algorithms/exalg) | Position frequency classification | Arasu & Garcia-Molina |
| [Anti-Unification](algorithms/anti-unification) | Pairwise tree generalization (LGG) | Plotkin / Reynolds 1970 |
| [FiVaTech](algorithms/fivatech) | Simultaneous N-way tree merge with LCS | Kayed & Chang 2010 |
| [RoadRunner](algorithms/roadrunner) | Token stream alignment (ACME/UFRE) | Crescenzi, Mecca & Merialdo 2001 |
| [k-Testable](algorithms/k-testable) | k-subtree pattern automata | Garcia & Oncina 1993 |

## Interactive Demos

Each algorithm page includes an **interactive visualization** powered by [Pyodide](https://pyodide.org/) (Python in the browser) and D3.js. You can:

- See the algorithm run step-by-step on example HTML pages
- Watch intermediate data structures evolve
- Modify the input HTML and re-run the algorithm
- Step forward/backward through algorithm phases

## Unified Protocol

All algorithms implement a shared protocol with JSON serialization:

```python
class InferredTemplate(Protocol):
    def extract(self, page) -> dict[str, Any] | None: ...
    def fixed_mask(self, page) -> dict[str, bool] | None: ...
    def serialize(self) -> dict[str, Any]: ...
```

See the [Serialization API](api/serialization) and [Tracer API](api/tracer) for details.
