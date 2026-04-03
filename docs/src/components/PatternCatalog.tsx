import React from "react";
import type { TraceStep } from "../hooks/usePyodide";

interface PatternCatalogProps {
  step: TraceStep;
  allSteps: TraceStep[];
  currentIndex: number;
}

export default function PatternCatalog({
  step,
  allSteps,
  currentIndex,
}: PatternCatalogProps): React.ReactElement | null {
  const extractStep = allSteps.find((s) => s.phase === "pattern_extraction");
  if (!extractStep || allSteps.indexOf(extractStep) > currentIndex) return null;

  const unionStep = allSteps.find((s) => s.phase === "pattern_union");
  const hasUnion = unionStep && allSteps.indexOf(unionStep) <= currentIndex;

  const { k, patterns_per_page, total_unique } = extractStep.data;

  return (
    <div style={{ marginTop: "1rem" }}>
      <h4>k-Subtree Patterns (k={k})</h4>

      {/* Summary stats */}
      <div
        style={{
          display: "flex",
          gap: "16px",
          marginBottom: "1rem",
          flexWrap: "wrap",
        }}
      >
        {(patterns_per_page as number[]).map((count: number, i: number) => (
          <div
            key={i}
            style={{
              padding: "8px 16px",
              background: "var(--ifm-color-emphasis-100)",
              borderRadius: "6px",
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: "1.2rem", fontWeight: 700 }}>{count}</div>
            <div style={{ fontSize: "0.75rem", opacity: 0.7 }}>Page {i + 1}</div>
          </div>
        ))}
        <div
          style={{
            padding: "8px 16px",
            background: "var(--ifm-color-primary-lightest)",
            borderRadius: "6px",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: "1.2rem", fontWeight: 700 }}>
            {total_unique}
          </div>
          <div style={{ fontSize: "0.75rem", opacity: 0.7 }}>Unique total</div>
        </div>
      </div>

      {/* Root patterns */}
      {hasUnion && unionStep && (
        <div style={{ marginTop: "1rem" }}>
          <h4>
            Root Patterns ({unionStep.data.root_pattern_count})
          </h4>
          <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
            {(unionStep.data.root_patterns as any[]).map(
              (pat: any, i: number) => (
                <div
                  key={i}
                  style={{
                    padding: "6px 12px",
                    border: "1px solid var(--ifm-color-emphasis-300)",
                    borderRadius: "6px",
                    fontFamily: "monospace",
                    fontSize: "0.82rem",
                    background: "var(--ifm-color-emphasis-100)",
                  }}
                >
                  &lt;{pat.tag}&gt;
                  {pat.attrs && pat.attrs.length > 0 && (
                    <span style={{ opacity: 0.6 }}>
                      {" "}
                      [{pat.attrs.join(", ")}]
                    </span>
                  )}
                  {pat.child_tags && (
                    <span style={{ opacity: 0.5 }}>
                      {" "}
                      → [{pat.child_tags.join(", ")}]
                    </span>
                  )}
                  {pat.has_text && (
                    <span
                      style={{
                        marginLeft: 4,
                        fontSize: "0.7rem",
                        color: "var(--ifm-color-success)",
                      }}
                    >
                      +text
                    </span>
                  )}
                </div>
              )
            )}
          </div>
        </div>
      )}
    </div>
  );
}
