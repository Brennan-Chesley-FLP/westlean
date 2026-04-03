# westlean

**Alpha** -- Experimental repository for serializable HTML template inference implementations.

Given a set of HTML pages generated from the same template, these algorithms infer the underlying template structure and can then extract variable data from new pages matching that template. All inferred templates are serializable to JSON and back.

## Algorithms

Five template inference algorithms, each following a common protocol:

| Algorithm | Paper | Approach |
|-----------|-------|----------|
| **Anti-unification** | Plotkin 1970 | Pairwise Least General Generalization fold |
| **ExAlg** | Arasu & Garcia-Molina 2003 | Token-frequency equivalence classes |
| **RoadRunner** | Crescenzi, Mecca & Merialdo 2001 | ACME alignment on linearized token streams |
| **k-Testable** | Kosala et al. 2003 | Local unranked tree automata |
| **FiVaTech** | Kayed & Chang 2010 | Simultaneous N-way tree merge |

## Common interface

```python
from westlean.algorithms.exalg import ExalgInferer
from lxml.html import fragment_fromstring

pages = [
    fragment_fromstring("<div><h1>Products</h1><p>Widget - $5</p></div>"),
    fragment_fromstring("<div><h1>Products</h1><p>Gadget - $9</p></div>"),
]

template = ExalgInferer().infer(pages)

# Extract data from a new page
result = template.extract(
    fragment_fromstring("<div><h1>Products</h1><p>Gizmo - $12</p></div>")
)

# Classify positions as fixed or variable
mask = template.fixed_mask(
    fragment_fromstring("<div><h1>Products</h1><p>Gizmo - $12</p></div>")
)

# Serialize to JSON and restore
import json
from westlean.protocol import restore_template

data = json.dumps(template.serialize())
restored = restore_template(json.loads(data))
```

## Testing

All algorithms are verified against the same Hypothesis-based property test suite covering recognition, discrimination, value extraction, structural masking, generalization, stability, and RELAX NG validation.

```bash
uv sync

# Run all tests
uv run python -m pytest tests/ -v

# Longer Hypothesis runs
HYPOTHESIS_PROFILE=ci uv run python -m pytest tests/ -v

# Deeper generated HTML trees (default 3)
WESTLEAN_MAX_DEPTH=5 uv run python -m pytest tests/ -v
```

## Status

Alpha. Interfaces may change. Loop and conditional support varies by algorithm.
