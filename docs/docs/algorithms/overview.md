---
sidebar_position: 1
---

# Algorithm Overview

Westlean implements five template inference algorithms. All operate on fixed-structure HTML (no loops or conditionals in v1) and implement the same `TemplateInferer` / `InferredTemplate` protocol.

## Comparison

| Algorithm | Strategy | Merge Style | Order-Dependent? | Complexity |
|-----------|----------|-------------|-------------------|------------|
| **ExAlg** | Statistical classification | All-at-once | No | O(n * p) |
| **Anti-Unification** | Tree generalization | Pairwise fold | Yes | O(n * p) |
| **FiVaTech** | LCS child alignment | Simultaneous | No | O(n * p^2) |
| **RoadRunner** | Token stream alignment | Pairwise fold | Yes | O(n * t^2) |
| **k-Testable** | Pattern acceptance | All-at-once | No | O(n * p) |

Where *n* = number of training pages, *p* = positions per page, *t* = tokens per page.

## Shared Concepts

### Position Keys

All algorithms use **canonical position keys** to identify locations in the DOM tree:

```
"text"       -- root element's text content
"@class"     -- root element's class attribute
"0/text"     -- first child's text
"0/@href"    -- first child's href attribute
"0/1/tail"   -- tail text after the second grandchild
```

### Fixed vs. Variable

Every algorithm classifies each position as:
- **Fixed** (True): constant template content, same across all pages
- **Variable** (False): data that changes between pages

### Protocol

```python
# Inference
inferer = ExalgInferer()
template = inferer.infer(pages)        # list of lxml elements

# Extraction
data = template.extract(new_page)      # {"var_0": "Alice", ...} or None
mask = template.fixed_mask(new_page)   # {"text": True, "0/text": False, ...}

# Serialization
json_data = template.serialize()       # JSON-friendly dict
restored = restore_template(json_data) # reconstruct from JSON
```
