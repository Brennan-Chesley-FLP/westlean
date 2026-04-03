---
sidebar_position: 1
---

# Serialization API

All inferred templates support JSON serialization via Pydantic models. This enables saving templates to disk, sending them over the network, and running algorithms in the browser via Pyodide.

## Usage

```python
import json
from westlean.algorithms.exalg import ExalgInferer
from westlean.protocol import restore_template

# Infer a template
inferer = ExalgInferer()
template = inferer.infer(pages)

# Serialize to JSON
data = template.serialize()
json_str = json.dumps(data)

# Restore from JSON
restored = restore_template(json.loads(json_str))
result = restored.extract(new_page)  # works identically
```

## Schema Format

Every serialized template includes an `"algorithm"` discriminator field:

### ExAlg

```json
{
  "algorithm": "exalg",
  "fixed": {"text": "Welcome", "0/@class": "header"},
  "variant": {"0/text": "var_0", "1/text": "var_1"},
  "tag_structure": {
    "": {"tag": "div", "attrs": ["class"]},
    "0": {"tag": "h1", "attrs": []}
  },
  "always_present": ["text", "0/text", "1/text"]
}
```

### Anti-Unification / FiVaTech

Both use a recursive `TplNodeModel`:

```json
{
  "algorithm": "anti_unification",
  "root": {
    "tag": "div",
    "attr_names": [],
    "text": {"is_fixed": true, "value": ""},
    "tail": {"is_fixed": true, "value": ""},
    "attrs": {},
    "children": [...],
    "children_var": null,
    "text_always_present": false,
    "tail_always_present": false
  }
}
```

### RoadRunner

Uses a discriminated union of UFRE elements:

```json
{
  "algorithm": "roadrunner",
  "ufre": [
    {"type": "literal", "token": {"kind": "open", "tag": "div", ...}},
    {"type": "var", "name": "v_0", "kind": "text", ...},
    {"type": "optional", "elements": [...]}
  ]
}
```

### k-Testable

```json
{
  "algorithm": "k_testable",
  "k": 2,
  "allowed": [{"tag": "div", "attrs": [], "child_tags": ["p"], "has_text": false}],
  "root_patterns": [...],
  "fixed": {...},
  "variant": {...},
  "always_present": [...]
}
```

### Empty Template

```json
{"algorithm": "empty"}
```

## Dispatch

`restore_template()` dispatches on the `"algorithm"` field to reconstruct the correct template type:

```python
from westlean.protocol import restore_template

template = restore_template({"algorithm": "exalg", ...})
# Returns ExalgTemplate instance
```
