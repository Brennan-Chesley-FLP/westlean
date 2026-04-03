import React from "react";
import type { TraceStep } from "../hooks/usePyodide";

interface ClassificationPanelProps {
  step: TraceStep;
  allSteps: TraceStep[];
  currentIndex: number;
}

export default function ClassificationPanel({
  step,
  allSteps,
  currentIndex,
}: ClassificationPanelProps): React.ReactElement | null {
  const classStep = allSteps.find((s) => s.phase === "classification");
  if (!classStep || allSteps.indexOf(classStep) > currentIndex) return null;

  const fixed: Record<string, string> = classStep.data.fixed || {};
  const variant: Record<string, string> = classStep.data.variant || {};

  const fixedEntries = Object.entries(fixed).sort(([a], [b]) =>
    a.localeCompare(b)
  );
  const variantEntries = Object.entries(variant).sort(([a], [b]) =>
    a.localeCompare(b)
  );

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "1rem",
        marginTop: "1rem",
      }}
    >
      {/* Fixed positions */}
      <div
        style={{
          padding: "12px",
          border: "1px solid var(--ifm-color-success)",
          borderRadius: "8px",
          background: "rgba(0, 180, 0, 0.04)",
        }}
      >
        <h4 style={{ color: "var(--ifm-color-success)", margin: "0 0 8px 0" }}>
          Fixed ({fixedEntries.length})
        </h4>
        {fixedEntries.length === 0 ? (
          <em style={{ opacity: 0.5 }}>No fixed positions</em>
        ) : (
          <ul style={{ margin: 0, paddingLeft: "16px", fontSize: "0.85rem" }}>
            {fixedEntries.map(([key, value]) => (
              <li key={key} style={{ marginBottom: "2px" }}>
                <code>{key}</code>
                {value ? (
                  <span>
                    {" "}
                    = <em>"{value}"</em>
                  </span>
                ) : (
                  <span style={{ opacity: 0.4 }}> (empty)</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Variable positions */}
      <div
        style={{
          padding: "12px",
          border: "1px solid var(--ifm-color-warning)",
          borderRadius: "8px",
          background: "rgba(230, 130, 0, 0.04)",
        }}
      >
        <h4
          style={{ color: "var(--ifm-color-warning)", margin: "0 0 8px 0" }}
        >
          Variable ({variantEntries.length})
        </h4>
        {variantEntries.length === 0 ? (
          <em style={{ opacity: 0.5 }}>No variable positions</em>
        ) : (
          <ul style={{ margin: 0, paddingLeft: "16px", fontSize: "0.85rem" }}>
            {variantEntries.map(([key, varName]) => (
              <li key={key} style={{ marginBottom: "2px" }}>
                <code>{key}</code> &rarr;{" "}
                <strong style={{ color: "var(--ifm-color-warning-darker)" }}>
                  {varName}
                </strong>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
