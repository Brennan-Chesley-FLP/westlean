import React from "react";
import type { TraceStep } from "../hooks/usePyodide";

interface TokenStreamProps {
  step: TraceStep;
  allSteps: TraceStep[];
  currentIndex: number;
}

const KIND_COLORS: Record<string, { bg: string; fg: string }> = {
  open: { bg: "#3b82f6", fg: "white" },
  close: { bg: "#93c5fd", fg: "#1e3a5f" },
  text: { bg: "#22c55e", fg: "white" },
  tail: { bg: "#14b8a6", fg: "white" },
  attr: { bg: "#a855f7", fg: "white" },
};

function kindStyle(kind: string) {
  return KIND_COLORS[kind] || { bg: "#6b7280", fg: "white" };
}

export default function TokenStream({
  step,
  allSteps,
  currentIndex,
}: TokenStreamProps): React.ReactElement | null {
  const linStep = allSteps.find((s) => s.phase === "linearization");
  if (!linStep || allSteps.indexOf(linStep) > currentIndex) return null;

  const tokens: Array<{
    kind: string;
    tag: string;
    attr_name: string;
    value: string;
    position_key: string;
  }> = linStep.data.tokens || [];

  return (
    <div style={{ marginTop: "1rem" }}>
      <h4>Token Stream (Page 1)</h4>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "3px",
          padding: "8px",
          background: "var(--ifm-color-emphasis-100)",
          borderRadius: "6px",
          overflowX: "auto",
        }}
      >
        {tokens.map((tok, i) => {
          const { bg, fg } = kindStyle(tok.kind);
          let label = "";
          if (tok.kind === "open") label = `<${tok.tag}>`;
          else if (tok.kind === "close") label = `</${tok.tag}>`;
          else if (tok.kind === "attr")
            label = `@${tok.attr_name}=${tok.value ? `"${tok.value.slice(0, 15)}"` : '""'}`;
          else if (tok.kind === "text" || tok.kind === "tail")
            label = tok.value
              ? `"${tok.value.slice(0, 20)}${tok.value.length > 20 ? "..." : ""}"`
              : '""';

          return (
            <span
              key={i}
              title={`${tok.kind} | ${tok.position_key} | ${tok.value}`}
              style={{
                padding: "2px 6px",
                borderRadius: "3px",
                fontFamily: "monospace",
                fontSize: "0.75rem",
                background: bg,
                color: fg,
                whiteSpace: "nowrap",
                cursor: "default",
              }}
            >
              {label}
            </span>
          );
        })}
      </div>

      {/* Legend */}
      <div
        style={{
          display: "flex",
          gap: "12px",
          marginTop: "6px",
          fontSize: "0.75rem",
        }}
      >
        {Object.entries(KIND_COLORS).map(([kind, { bg }]) => (
          <span key={kind} style={{ display: "flex", alignItems: "center", gap: 3 }}>
            <span
              style={{
                display: "inline-block",
                width: 10,
                height: 10,
                borderRadius: 2,
                background: bg,
              }}
            />
            {kind}
          </span>
        ))}
      </div>
    </div>
  );
}
