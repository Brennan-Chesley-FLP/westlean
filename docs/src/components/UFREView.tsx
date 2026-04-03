import React from "react";
import type { TraceStep } from "../hooks/usePyodide";

interface UFREViewProps {
  step: TraceStep;
  allSteps: TraceStep[];
  currentIndex: number;
}

interface UFREElement {
  type: "literal" | "var" | "optional" | "repeat";
  token?: { kind: string; tag: string; attr_name: string; value: string; position_key: string };
  name?: string;
  kind?: string;
  position_key?: string;
  always_has_value?: boolean;
  elements?: UFREElement[];
  var_name?: string;
}

function UFREToken({ elem }: { elem: UFREElement }): React.ReactElement {
  if (elem.type === "literal") {
    const tok = elem.token!;
    let label = "";
    if (tok.kind === "open") label = `<${tok.tag}>`;
    else if (tok.kind === "close") label = `</${tok.tag}>`;
    else if (tok.kind === "attr") label = `@${tok.attr_name}`;
    else label = tok.value ? `"${tok.value.slice(0, 15)}"` : '""';

    return (
      <span
        title={`Literal: ${tok.kind} | ${tok.position_key}`}
        style={{
          padding: "3px 7px",
          borderRadius: "3px",
          fontFamily: "monospace",
          fontSize: "0.78rem",
          background: "rgba(56, 168, 86, 0.15)",
          color: "#2d7a3e",
          border: "1px solid rgba(56,168,86,0.3)",
          whiteSpace: "nowrap",
        }}
      >
        {label}
      </span>
    );
  }

  if (elem.type === "var") {
    return (
      <span
        title={`Var: ${elem.name} | ${elem.kind} | ${elem.position_key}`}
        style={{
          padding: "3px 7px",
          borderRadius: "3px",
          fontFamily: "monospace",
          fontSize: "0.78rem",
          background: "rgba(232, 123, 56, 0.15)",
          color: "#b5621e",
          border: "1px dashed rgba(232,123,56,0.5)",
          whiteSpace: "nowrap",
        }}
      >
        {elem.name}
      </span>
    );
  }

  if (elem.type === "optional") {
    return (
      <span
        style={{
          display: "inline-flex",
          gap: "2px",
          padding: "2px 4px",
          borderRadius: "4px",
          background: "rgba(139, 92, 246, 0.08)",
          border: "1px dotted rgba(139,92,246,0.4)",
          flexWrap: "wrap",
        }}
        title="Optional block"
      >
        <span style={{ fontSize: "0.65rem", opacity: 0.5, marginRight: 2 }}>
          opt
        </span>
        {(elem.elements || []).map((child, i) => (
          <UFREToken key={i} elem={child} />
        ))}
      </span>
    );
  }

  if (elem.type === "repeat") {
    return (
      <span
        style={{
          display: "inline-flex",
          gap: "2px",
          padding: "2px 4px",
          borderRadius: "4px",
          background: "rgba(59, 130, 246, 0.08)",
          border: "1px dashed rgba(59,130,246,0.4)",
          flexWrap: "wrap",
        }}
        title={`Repeat: ${elem.var_name || "iterator"}`}
      >
        <span style={{ fontSize: "0.65rem", opacity: 0.5, marginRight: 2 }}>
          {elem.var_name || "rep"}+
        </span>
        {(elem.elements || []).map((child, i) => (
          <UFREToken key={i} elem={child} />
        ))}
      </span>
    );
  }

  return <span>?</span>;
}

export default function UFREView({
  step,
  allSteps,
  currentIndex,
}: UFREViewProps): React.ReactElement | null {
  // Find latest UFRE state from acme_fold or result steps
  let ufre: UFREElement[] | null = null;
  let label = "";

  for (let i = currentIndex; i >= 0; i--) {
    const s = allSteps[i];
    if (s.phase === "repeat_detection" && s.data.ufre_after) {
      ufre = s.data.ufre_after;
      label = `After repeat detection (page ${s.data.page_index})`;
      break;
    }
    if (s.phase === "acme_fold" && s.data.ufre_after) {
      ufre = s.data.ufre_after;
      label = `After fold with page ${s.data.page_index + 1}`;
      break;
    }
    if (s.phase === "result" && s.data.ufre) {
      ufre = s.data.ufre;
      label = "Final UFRE";
      break;
    }
    if (s.phase === "ufre_init") {
      // Initial state — no ufre data in trace, just show stats
      label = `Initial: ${s.data.literal_count} literals`;
      break;
    }
  }

  // Only show after ufre_init phase (RoadRunner) or result with ufre data (ExAlg)
  const initStep = allSteps.find((s) => s.phase === "ufre_init");
  const resultWithUfre = allSteps.find((s) => s.phase === "result" && s.data.ufre);
  const showAfter = initStep || resultWithUfre;
  if (!showAfter || allSteps.indexOf(showAfter) > currentIndex) return null;

  if (!ufre) {
    return (
      <div style={{ marginTop: "1rem" }}>
        <h4>UFRE Grammar</h4>
        <div
          style={{
            padding: "8px 12px",
            background: "var(--ifm-color-emphasis-100)",
            borderRadius: "4px",
            fontSize: "0.85rem",
          }}
        >
          {label}
        </div>
      </div>
    );
  }

  // Count element types
  const counts = { literal: 0, var: 0, optional: 0, repeat: 0 };
  function countElements(elems: UFREElement[]) {
    for (const e of elems) {
      if (e.type in counts) counts[e.type as keyof typeof counts]++;
      if ((e.type === "optional" || e.type === "repeat") && e.elements) countElements(e.elements);
    }
  }
  countElements(ufre);

  return (
    <div style={{ marginTop: "1rem" }}>
      <h4>UFRE Grammar</h4>
      <div
        style={{
          fontSize: "0.8rem",
          opacity: 0.7,
          marginBottom: "6px",
        }}
      >
        {label} — {counts.literal} literals, {counts.var} vars
        {counts.optional > 0 && `, ${counts.optional} optional`}
        {counts.repeat > 0 && `, ${counts.repeat} repeat`}
      </div>
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
        {ufre.map((elem, i) => (
          <UFREToken key={i} elem={elem} />
        ))}
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
        <span>
          <span
            style={{
              display: "inline-block",
              width: 10,
              height: 10,
              borderRadius: 2,
              background: "rgba(56, 168, 86, 0.3)",
              border: "1px solid rgba(56,168,86,0.5)",
              marginRight: 3,
              verticalAlign: "middle",
            }}
          />
          Literal (fixed)
        </span>
        <span>
          <span
            style={{
              display: "inline-block",
              width: 10,
              height: 10,
              borderRadius: 2,
              background: "rgba(232, 123, 56, 0.3)",
              border: "1px dashed rgba(232,123,56,0.5)",
              marginRight: 3,
              verticalAlign: "middle",
            }}
          />
          Var (data)
        </span>
        <span>
          <span
            style={{
              display: "inline-block",
              width: 10,
              height: 10,
              borderRadius: 2,
              background: "rgba(139, 92, 246, 0.15)",
              border: "1px dotted rgba(139,92,246,0.4)",
              marginRight: 3,
              verticalAlign: "middle",
            }}
          />
          Optional
        </span>
        <span>
          <span
            style={{
              display: "inline-block",
              width: 10,
              height: 10,
              borderRadius: 2,
              background: "rgba(59, 130, 246, 0.15)",
              border: "1px dashed rgba(59,130,246,0.4)",
              marginRight: 3,
              verticalAlign: "middle",
            }}
          />
          Repeat
        </span>
      </div>
    </div>
  );
}
