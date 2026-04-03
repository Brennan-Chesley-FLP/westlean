import React from "react";
import type { TraceStep } from "../hooks/usePyodide";

interface PositionMapTableProps {
  step: TraceStep;
  /** All steps up to and including the current one, for cumulative display */
  allSteps: TraceStep[];
  currentIndex: number;
}

export default function PositionMapTable({
  step,
  allSteps,
  currentIndex,
}: PositionMapTableProps): React.ReactElement | null {
  // Don't show until we've reached the position_mapping step
  const mappingStep = allSteps.find((s) => s.phase === "position_mapping");
  if (!mappingStep || allSteps.indexOf(mappingStep) > currentIndex) return null;

  const positionMaps: Record<string, string>[] =
    mappingStep.data.position_maps || [];
  if (positionMaps.length === 0) return null;

  // Collect all position keys
  const allKeys = new Set<string>();
  positionMaps.forEach((pm) => Object.keys(pm).forEach((k) => allKeys.add(k)));
  const sortedKeys = Array.from(allKeys).sort();

  // Get classification data if we've reached that phase
  const classStep = allSteps.find((s) => s.phase === "classification");
  const hasClassification =
    classStep && allSteps.indexOf(classStep) <= currentIndex;
  const fixed: Record<string, string> = hasClassification
    ? classStep!.data.fixed || {}
    : {};
  const variant: Record<string, string> = hasClassification
    ? classStep!.data.variant || {}
    : {};

  // Get frequency data if we've reached that phase
  const freqStep = allSteps.find((s) => s.phase === "frequency_analysis");
  const hasFrequency = freqStep && allSteps.indexOf(freqStep) <= currentIndex;
  const perKey: Record<string, { all_same: boolean }> = hasFrequency
    ? freqStep!.data.per_key || {}
    : {};

  return (
    <div style={{ overflowX: "auto" }}>
      <h4>Position Map</h4>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: "0.85rem",
        }}
      >
        <thead>
          <tr>
            <th
              style={{
                textAlign: "left",
                padding: "6px 8px",
                borderBottom: "2px solid var(--ifm-color-emphasis-300)",
              }}
            >
              Position Key
            </th>
            {positionMaps.map((_, i) => (
              <th
                key={i}
                style={{
                  textAlign: "left",
                  padding: "6px 8px",
                  borderBottom: "2px solid var(--ifm-color-emphasis-300)",
                }}
              >
                Page {i + 1}
              </th>
            ))}
            {hasClassification && (
              <th
                style={{
                  textAlign: "left",
                  padding: "6px 8px",
                  borderBottom: "2px solid var(--ifm-color-emphasis-300)",
                }}
              >
                Classification
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {sortedKeys.map((key) => {
            const values = positionMaps.map((pm) => pm[key] ?? "—");
            const allSame = hasFrequency ? perKey[key]?.all_same : undefined;
            const isFixed = key in fixed;
            const isVariant = key in variant;

            const rowBg =
              hasClassification && isFixed
                ? "rgba(0, 180, 0, 0.08)"
                : hasClassification && isVariant
                  ? "rgba(230, 130, 0, 0.08)"
                  : hasFrequency && allSame
                    ? "rgba(0, 180, 0, 0.05)"
                    : hasFrequency && allSame === false
                      ? "rgba(230, 130, 0, 0.05)"
                      : undefined;

            return (
              <tr key={key} style={{ background: rowBg }}>
                <td
                  style={{
                    padding: "4px 8px",
                    fontFamily: "monospace",
                    borderBottom: "1px solid var(--ifm-color-emphasis-200)",
                    whiteSpace: "nowrap",
                  }}
                >
                  {key}
                </td>
                {values.map((val, i) => (
                  <td
                    key={i}
                    style={{
                      padding: "4px 8px",
                      borderBottom: "1px solid var(--ifm-color-emphasis-200)",
                      maxWidth: "200px",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                    title={val}
                  >
                    {val || <em style={{ opacity: 0.4 }}>empty</em>}
                  </td>
                ))}
                {hasClassification && (
                  <td
                    style={{
                      padding: "4px 8px",
                      borderBottom: "1px solid var(--ifm-color-emphasis-200)",
                      fontWeight: 600,
                      color: isFixed
                        ? "var(--ifm-color-success)"
                        : "var(--ifm-color-warning)",
                    }}
                  >
                    {isFixed
                      ? "Fixed"
                      : isVariant
                        ? variant[key]
                        : "—"}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
