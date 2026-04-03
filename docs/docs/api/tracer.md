---
sidebar_position: 2
---

# Tracer API

The tracer records intermediate algorithm steps as JSON-serializable events, enabling step-by-step visualization in the interactive documentation.

## Usage

```python
from westlean.tracer import tracing
from westlean.algorithms.tracing_exalg import TracingExalgInferer

with tracing() as tracer:
    inferer = TracingExalgInferer()
    template = inferer.infer(pages)

# Get all trace steps as JSON
steps = tracer.to_json()
```

## TraceStep Format

Each step is a dictionary with:

```json
{
  "algorithm": "exalg",
  "phase": "equivalence_classes",
  "step_index": 1,
  "description": "Compute occurrence vectors and classify structural keys into template/loop/optional",
  "data": { ... }
}
```

## ExAlg Phases

The `TracingExalgInferer` emits 7 phases corresponding to the paper's ECGM sub-modules and ConstTemp:

### 1. `tokenization`

Linearize each page into a token stream with structural keys and tag-path contexts.

```json
{
  "page_count": 3,
  "tokens_per_page": [15, 15, 15],
  "sample_tokens": [
    {"kind": "open", "tag": "div", "attr_name": "", "value": "", "position_key": "", "context": "div"},
    {"kind": "text", "tag": "", "attr_name": "", "value": "", "position_key": "text", "context": "div"}
  ]
}
```

### 2. `equivalence_classes`

FindEq: compute occurrence vectors, form equivalence classes, classify as template/loop/optional.

```json
{
  "total_structural_keys": 12,
  "template_constants": 8,
  "loop_markers": 2,
  "optional_markers": 0,
  "sample_vectors": {"open div [div]": [1, 1]},
  "equivalence_class_count": 3,
  "largest_classes": [
    {"vector": [1, 1], "size": 8, "sample_members": ["open div [div]", "text [div]"]}
  ]
}
```

### 3. `diffformat`

DiffFormat: refine contexts for fixed-count sibling disambiguation (Observation 4.4).

```json
{
  "contexts_refined": true,
  "changed_contexts": [
    {"context": "table/tbody#0/tr", "status": "added (sibling-indexed)"}
  ],
  "template_constants_before": 8,
  "template_constants_after": 12,
  "loop_markers_before": 4,
  "loop_markers_after": 2
}
```

### 4. `handinv`

HandInv: demote skeleton tokens nested inside loop bodies.

```json
{
  "demoted_count": 2,
  "demoted_keys": ["attr @class [div/p]", "text [div/p]"],
  "template_constants_before": 12,
  "template_constants_after": 10
}
```

### 5. `diffeq`

DiffEq: promote first instances of loop elements with fixed values (Observation 4.5).

```json
{
  "promoted_count": 3,
  "promoted_keys": ["open tr [table/tbody]", "text [table/tbody/tr]"],
  "promotion_decisions": [
    {"context": "table/tbody/tr", "decision": "promoted", "reason": "First instance fixed value 'Header', later instances vary", "group_size": 5}
  ],
  "skeleton_size_final": 13
}
```

### 6. `skeleton`

Summary of skeleton extraction and gap regions for ConstTemp analysis.

```json
{
  "skeleton_tokens": 10,
  "gap_regions": 3,
  "template_constants": 8,
  "promoted_first_instances": 2
}
```

### 7. `result`

The full serialized template (same format as the Serialization API).

## Context Manager

The tracer uses Python's `ContextVar` for thread-safe activation:

```python
from westlean.tracer import tracing, get_tracer

# Outside tracing context
assert get_tracer() is None

with tracing() as tracer:
    # Inside tracing context
    assert get_tracer() is tracer

# Back to None
assert get_tracer() is None
```

Tracing is fully opt-in. Algorithms work identically with or without an active tracer.
