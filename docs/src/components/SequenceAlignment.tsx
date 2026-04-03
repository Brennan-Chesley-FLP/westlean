import React from "react";
import type { TraceStep } from "../hooks/usePyodide";

interface SequenceAlignmentProps {
  step: TraceStep;
  allSteps: TraceStep[];
  currentIndex: number;
}

const TAG_COLORS: Record<string, string> = {
  div: "#4a90d9",
  p: "#6bb86b",
  span: "#d9a44a",
  h1: "#d94a4a",
  h2: "#d94a4a",
  h3: "#d94a4a",
  a: "#9b59b6",
  ul: "#2ecc71",
  li: "#27ae60",
  table: "#e67e22",
  tr: "#d35400",
  td: "#c0392b",
};

function tagColor(tag: string): string {
  return TAG_COLORS[tag] || "#6b7280";
}

export default function SequenceAlignment({
  step,
  allSteps,
  currentIndex,
}: SequenceAlignmentProps): React.ReactElement | null {
  const seqStep = allSteps.find((s) => s.phase === "child_sequences");
  if (!seqStep || allSteps.indexOf(seqStep) > currentIndex) return null;

  const sequences: string[][] = seqStep.data.sequences || [];

  const backboneStep = allSteps.find((s) => s.phase === "backbone_computation");
  const hasBackbone =
    backboneStep && allSteps.indexOf(backboneStep) <= currentIndex;
  const backbone: string[] = hasBackbone
    ? backboneStep!.data.backbone || []
    : [];

  const alignStep = allSteps.find((s) => s.phase === "alignment");
  const hasAlignment = alignStep && allSteps.indexOf(alignStep) <= currentIndex;
  const mappings: number[][] = hasAlignment
    ? alignStep!.data.mappings || []
    : [];

  return (
    <div style={{ marginTop: "1rem" }}>
      <h4>Child Sequence Alignment</h4>

      {/* Page sequences */}
      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        {sequences.map((seq, pageIdx) => {
          const aligned = hasAlignment && mappings[pageIdx]
            ? new Set(mappings[pageIdx])
            : new Set<number>();

          return (
            <div
              key={pageIdx}
              style={{ display: "flex", alignItems: "center", gap: "4px" }}
            >
              <span
                style={{
                  minWidth: "60px",
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  opacity: 0.7,
                }}
              >
                Page {pageIdx + 1}
              </span>
              {seq.map((tag, i) => {
                const isAligned = aligned.has(i);
                return (
                  <span
                    key={i}
                    style={{
                      padding: "4px 10px",
                      borderRadius: "4px",
                      fontFamily: "monospace",
                      fontSize: "0.82rem",
                      fontWeight: isAligned ? 700 : 400,
                      background: isAligned
                        ? tagColor(tag)
                        : "var(--ifm-color-emphasis-200)",
                      color: isAligned ? "white" : "var(--ifm-font-color-base)",
                      opacity: hasAlignment && !isAligned ? 0.4 : 1,
                      border: isAligned
                        ? "none"
                        : "1px dashed var(--ifm-color-emphasis-400)",
                    }}
                  >
                    {tag}
                  </span>
                );
              })}
            </div>
          );
        })}

        {/* Backbone row */}
        {hasBackbone && backbone.length > 0 && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "4px",
              marginTop: "4px",
              paddingTop: "8px",
              borderTop: "2px solid var(--ifm-color-emphasis-300)",
            }}
          >
            <span
              style={{
                minWidth: "60px",
                fontSize: "0.8rem",
                fontWeight: 700,
                color: "var(--ifm-color-primary)",
              }}
            >
              Backbone
            </span>
            {backbone.map((tag, i) => (
              <span
                key={i}
                style={{
                  padding: "4px 10px",
                  borderRadius: "4px",
                  fontFamily: "monospace",
                  fontSize: "0.82rem",
                  fontWeight: 700,
                  background: tagColor(tag),
                  color: "white",
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        )}

        {hasBackbone && backbone.length === 0 && (
          <div
            style={{
              padding: "8px 12px",
              background: "rgba(220, 50, 50, 0.08)",
              borderRadius: "4px",
              color: "var(--ifm-color-danger)",
              fontSize: "0.85rem",
              marginTop: "4px",
            }}
          >
            No common child subsequence — children region becomes variable
          </div>
        )}
      </div>
    </div>
  );
}
