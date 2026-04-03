import React from "react";
import type { TraceStep } from "../hooks/usePyodide";

interface Props {
  step: TraceStep;
  allSteps: TraceStep[];
  currentIndex: number;
}

function RegionDetectionViz({ step }: { step: TraceStep }) {
  const { page_index, regions } = step.data;
  if (!regions) return null;

  const backbone: string[] = regions.backbone_tags || [];
  const repeating: any[] = regions.repeating || [];
  const optional: any[] = regions.optional || [];

  const hasChanges = repeating.length > 0 || optional.length > 0;

  return (
    <div style={{ marginTop: "0.75rem" }}>
      <div style={{ fontSize: "0.82rem", marginBottom: 8, opacity: 0.7 }}>
        Folding page {page_index + 1} — child count mismatch triggered LCS region detection
      </div>

      {backbone.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontWeight: 600, fontSize: "0.8rem", marginBottom: 4 }}>
            LCS Backbone ({backbone.length} elements):
          </div>
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
            {backbone.map((tag, i) => (
              <span
                key={i}
                style={{
                  padding: "2px 8px",
                  borderRadius: 4,
                  background: "rgba(56,168,86,0.15)",
                  color: "#2d7a3e",
                  fontFamily: "monospace",
                  fontSize: "0.78rem",
                  border: "1px solid rgba(56,168,86,0.3)",
                }}
              >
                &lt;{tag}&gt;
              </span>
            ))}
          </div>
        </div>
      )}

      {repeating.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontWeight: 600, fontSize: "0.8rem", marginBottom: 4 }}>
            Repeating Regions ({repeating.length}):
          </div>
          {repeating.map((r: any, i: number) => (
            <div
              key={i}
              style={{
                padding: "6px 10px",
                marginBottom: 4,
                borderRadius: 4,
                background: "rgba(59,130,246,0.06)",
                border: "1px solid rgba(59,130,246,0.2)",
                fontSize: "0.8rem",
              }}
            >
              <span style={{ color: "#1d4ed8", fontWeight: 600 }}>
                &lt;{r.tag}&gt;+
              </span>
              {r.var_name && (
                <span style={{ marginLeft: 8, opacity: 0.6 }}>
                  → {r.var_name}
                </span>
              )}
              {r.after_pos !== undefined && (
                <span style={{ marginLeft: 8, opacity: 0.5, fontSize: "0.72rem" }}>
                  after backbone[{r.after_pos}]
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {optional.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontWeight: 600, fontSize: "0.8rem", marginBottom: 4 }}>
            Optional Regions ({optional.length}):
          </div>
          {optional.map((o: any, i: number) => (
            <div
              key={i}
              style={{
                padding: "6px 10px",
                marginBottom: 4,
                borderRadius: 4,
                background: "rgba(139,92,246,0.06)",
                border: "1px solid rgba(139,92,246,0.2)",
                fontSize: "0.8rem",
              }}
            >
              <span style={{ color: "#7c3aed", fontWeight: 600 }}>
                {(o.tags || []).map((t: string) => `<${t}>`).join(" ")}?
              </span>
              {o.after_pos !== undefined && (
                <span style={{ marginLeft: 8, opacity: 0.5, fontSize: "0.72rem" }}>
                  after backbone[{o.after_pos}]
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {!hasChanges && backbone.length === 0 && (
        <div style={{ padding: 12, background: "var(--ifm-color-emphasis-100)", borderRadius: 6, fontSize: "0.85rem" }}>
          No regions detected — child sequences are compatible without LCS alignment.
        </div>
      )}
    </div>
  );
}

export default function AntiUnifPhaseViz({ step }: Props): React.ReactElement | null {
  if (step.phase === "region_detection") {
    return <RegionDetectionViz step={step} />;
  }
  return null;
}
