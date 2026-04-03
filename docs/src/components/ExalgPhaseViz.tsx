import React from "react";
import type { TraceStep } from "../hooks/usePyodide";

interface Props {
  step: TraceStep;
  allSteps: TraceStep[];
  currentIndex: number;
}

/* ------------------------------------------------------------------ */
/*  Tokenization: show token stream for page 1 with colored badges    */
/* ------------------------------------------------------------------ */

function TokenizationViz({ step }: { step: TraceStep }) {
  const { page_count, tokens_per_page, sample_tokens } = step.data;
  if (!sample_tokens) return null;

  const kindColor: Record<string, string> = {
    open: "#2563eb",
    close: "#2563eb",
    text: "#16a34a",
    tail: "#16a34a",
    attr: "#9333ea",
  };

  return (
    <div style={{ marginTop: "0.75rem" }}>
      <div style={{ fontSize: "0.82rem", marginBottom: "0.5rem", opacity: 0.7 }}>
        {page_count} pages — {(tokens_per_page || []).join(", ")} tokens each
      </div>
      <div style={{ fontSize: "0.78rem", fontWeight: 600, marginBottom: 4 }}>
        Page 1 token stream:
      </div>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "3px",
          padding: "8px",
          background: "var(--ifm-color-emphasis-100)",
          borderRadius: "6px",
          maxHeight: "220px",
          overflowY: "auto",
        }}
      >
        {sample_tokens.map((tok: any, i: number) => {
          const color = kindColor[tok.kind] || "#666";
          let label = "";
          if (tok.kind === "open") label = `<${tok.tag}>`;
          else if (tok.kind === "close") label = `</${tok.tag}>`;
          else if (tok.kind === "attr") label = `@${tok.attr_name}="${tok.value}"`;
          else if (tok.kind === "text" && tok.value) label = `"${tok.value.slice(0, 20)}"`;
          else if (tok.kind === "text") label = '""';
          else if (tok.kind === "tail" && tok.value) label = `tail:"${tok.value.slice(0, 15)}"`;
          else label = `tail:""`;

          return (
            <span
              key={i}
              title={`${tok.kind} | context: ${tok.context} | pos: ${tok.position_key}`}
              style={{
                padding: "2px 6px",
                borderRadius: "3px",
                fontFamily: "monospace",
                fontSize: "0.72rem",
                background: `${color}18`,
                color,
                border: `1px solid ${color}40`,
                whiteSpace: "nowrap",
              }}
            >
              {label}
            </span>
          );
        })}
      </div>
      <div style={{ display: "flex", gap: "12px", marginTop: "4px", fontSize: "0.7rem", opacity: 0.6 }}>
        <span><span style={{ color: "#2563eb" }}>■</span> structural</span>
        <span><span style={{ color: "#16a34a" }}>■</span> text/tail</span>
        <span><span style={{ color: "#9333ea" }}>■</span> attribute</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  FindEq: occurrence vectors + classification                       */
/* ------------------------------------------------------------------ */

function FindEqViz({ step }: { step: TraceStep }) {
  const {
    total_structural_keys,
    template_constants,
    loop_markers,
    optional_markers,
    sample_vectors,
  } = step.data;

  const vectors: Record<string, number[]> = sample_vectors || {};
  const entries = Object.entries(vectors).sort(([, a], [, b]) => {
    const aVaries = new Set(a as number[]).size > 1;
    const bVaries = new Set(b as number[]).size > 1;
    if (aVaries !== bVaries) return aVaries ? 1 : -1;
    return 0;
  });

  return (
    <div style={{ marginTop: "0.75rem" }}>
      {/* Summary bar */}
      <div style={{ display: "flex", gap: "8px", marginBottom: "0.5rem", fontSize: "0.82rem" }}>
        <span style={{ padding: "2px 8px", borderRadius: 4, background: "rgba(56,168,86,0.15)", color: "#2d7a3e", fontWeight: 600 }}>
          {template_constants} template
        </span>
        <span style={{ padding: "2px 8px", borderRadius: 4, background: "rgba(59,130,246,0.15)", color: "#1d4ed8", fontWeight: 600 }}>
          {loop_markers} loop
        </span>
        {optional_markers > 0 && (
          <span style={{ padding: "2px 8px", borderRadius: 4, background: "rgba(139,92,246,0.15)", color: "#7c3aed", fontWeight: 600 }}>
            {optional_markers} optional
          </span>
        )}
        <span style={{ opacity: 0.5 }}>({total_structural_keys} total keys)</span>
      </div>

      {/* Vector table */}
      <div style={{ maxHeight: "280px", overflowY: "auto", borderRadius: 6, border: "1px solid var(--ifm-color-emphasis-200)" }}>
        <table style={{ width: "100%", fontSize: "0.78rem", borderCollapse: "collapse", margin: 0 }}>
          <thead>
            <tr style={{ background: "var(--ifm-color-emphasis-100)", position: "sticky", top: 0 }}>
              <th style={{ padding: "4px 8px", textAlign: "left" }}>Structural Key</th>
              <th style={{ padding: "4px 8px", textAlign: "left" }}>Vector</th>
              <th style={{ padding: "4px 8px", textAlign: "left" }}>Class</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([key, vec]) => {
              const vecArr = vec as number[];
              const isFixed = new Set(vecArr).size === 1;
              const hasZero = vecArr.some((v) => v === 0);
              const cls = hasZero ? "optional" : isFixed ? "template" : "loop";
              const clsColor = cls === "template" ? "#2d7a3e" : cls === "loop" ? "#1d4ed8" : "#7c3aed";
              return (
                <tr key={key} style={{ borderTop: "1px solid var(--ifm-color-emphasis-200)" }}>
                  <td style={{ padding: "3px 8px", fontFamily: "monospace", fontSize: "0.74rem" }}>{key}</td>
                  <td style={{ padding: "3px 8px", fontFamily: "monospace" }}>({vecArr.join(", ")})</td>
                  <td style={{ padding: "3px 8px", color: clsColor, fontWeight: 600 }}>{cls}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  DiffFormat: context refinement                                    */
/* ------------------------------------------------------------------ */

function DiffFormatViz({ step }: { step: TraceStep }) {
  const { contexts_refined, changed_contexts, template_constants_before, template_constants_after, loop_markers_before, loop_markers_after } = step.data;

  if (!contexts_refined) {
    return (
      <div style={{ marginTop: "0.75rem", padding: "12px", background: "var(--ifm-color-emphasis-100)", borderRadius: 6, fontSize: "0.85rem" }}>
        <strong>No context refinement needed.</strong> All fixed-count elements have unique contexts — no sibling disambiguation required.
        <div style={{ marginTop: 4, opacity: 0.6, fontSize: "0.78rem" }}>
          Template constants: {template_constants_before} | Loop markers: {loop_markers_before}
        </div>
      </div>
    );
  }

  return (
    <div style={{ marginTop: "0.75rem" }}>
      <div style={{ padding: "12px", background: "rgba(59,130,246,0.06)", border: "1px solid rgba(59,130,246,0.2)", borderRadius: 6, fontSize: "0.85rem" }}>
        <strong>Context refinement applied.</strong> Fixed-count siblings were indexed to disambiguate descendant tokens.
        <div style={{ marginTop: 8, fontSize: "0.78rem" }}>
          Template constants: {template_constants_before} → {template_constants_after} |
          Loop markers: {loop_markers_before} → {loop_markers_after}
        </div>
        {(changed_contexts || []).length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontWeight: 600, fontSize: "0.78rem", marginBottom: 4 }}>Changed contexts:</div>
            <ul style={{ margin: 0, paddingLeft: 16, fontSize: "0.75rem", fontFamily: "monospace" }}>
              {changed_contexts.map((c: string, i: number) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  HandInv: demotion of nested skeleton tokens                       */
/* ------------------------------------------------------------------ */

function HandInvViz({ step }: { step: TraceStep }) {
  const { demoted_count, demoted_keys, template_constants_before, template_constants_after } = step.data;

  if (demoted_count === 0) {
    return (
      <div style={{ marginTop: "0.75rem", padding: "12px", background: "var(--ifm-color-emphasis-100)", borderRadius: 6, fontSize: "0.85rem" }}>
        <strong>No demotions needed.</strong> All template-constant tokens are at the correct nesting level — none are inside loop-body elements.
        <div style={{ marginTop: 4, opacity: 0.6, fontSize: "0.78rem" }}>
          Template constants: {template_constants_before} (unchanged)
        </div>
      </div>
    );
  }

  return (
    <div style={{ marginTop: "0.75rem" }}>
      <div style={{ padding: "12px", background: "rgba(234,88,12,0.06)", border: "1px solid rgba(234,88,12,0.2)", borderRadius: 6, fontSize: "0.85rem" }}>
        <strong>Demoted {demoted_count} token(s)</strong> from skeleton — they were nested inside loop-body elements.
        <div style={{ marginTop: 4, opacity: 0.6, fontSize: "0.78rem" }}>
          Template constants: {template_constants_before} → {template_constants_after}
        </div>
        {(demoted_keys || []).length > 0 && (
          <ul style={{ margin: "8px 0 0 0", paddingLeft: 16, fontSize: "0.78rem", fontFamily: "monospace" }}>
            {demoted_keys.map((k: string, i: number) => (
              <li key={i} style={{ color: "#ea580c" }}>{k}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  DiffEq: promotion decisions                                      */
/* ------------------------------------------------------------------ */

function DiffEqViz({ step }: { step: TraceStep }) {
  const { promoted_count, promoted_keys, promotion_decisions, skeleton_size_final } = step.data;
  const decisions: any[] = promotion_decisions || [];

  return (
    <div style={{ marginTop: "0.75rem" }}>
      {promoted_count > 0 ? (
        <div style={{ padding: "12px", background: "rgba(16,185,129,0.06)", border: "1px solid rgba(16,185,129,0.2)", borderRadius: 6, fontSize: "0.85rem", marginBottom: 8 }}>
          <strong>Promoted {promoted_count} token(s)</strong> — first loop instance has fixed value across all pages.
          <ul style={{ margin: "8px 0 0 0", paddingLeft: 16, fontSize: "0.78rem", fontFamily: "monospace" }}>
            {(promoted_keys || []).map((k: string, i: number) => (
              <li key={i} style={{ color: "#059669" }}>{k}</li>
            ))}
          </ul>
        </div>
      ) : (
        <div style={{ padding: "12px", background: "var(--ifm-color-emphasis-100)", borderRadius: 6, fontSize: "0.85rem", marginBottom: 8 }}>
          <strong>No promotions.</strong> No loop-key group qualifies for first-instance promotion.
        </div>
      )}

      {decisions.length > 0 && (
        <div style={{ maxHeight: "200px", overflowY: "auto", borderRadius: 6, border: "1px solid var(--ifm-color-emphasis-200)" }}>
          <table style={{ width: "100%", fontSize: "0.75rem", borderCollapse: "collapse", margin: 0 }}>
            <thead>
              <tr style={{ background: "var(--ifm-color-emphasis-100)", position: "sticky", top: 0 }}>
                <th style={{ padding: "4px 8px", textAlign: "left" }}>Context</th>
                <th style={{ padding: "4px 8px", textAlign: "left" }}>Decision</th>
                <th style={{ padding: "4px 8px", textAlign: "left" }}>Reason</th>
              </tr>
            </thead>
            <tbody>
              {decisions.map((d: any, i: number) => {
                const color = d.decision === "promoted" ? "#059669" : d.decision.startsWith("skip") ? "#9ca3af" : "#666";
                return (
                  <tr key={i} style={{ borderTop: "1px solid var(--ifm-color-emphasis-200)" }}>
                    <td style={{ padding: "3px 8px", fontFamily: "monospace", fontSize: "0.72rem" }}>{d.context}</td>
                    <td style={{ padding: "3px 8px", color, fontWeight: 600 }}>{d.decision}</td>
                    <td style={{ padding: "3px 8px", fontSize: "0.72rem" }}>{d.reason}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div style={{ marginTop: 6, fontSize: "0.78rem", opacity: 0.6 }}>
        Final skeleton size: {skeleton_size_final} structural keys
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Skeleton: skeleton tokens + gap regions                           */
/* ------------------------------------------------------------------ */

function SkeletonViz({ step }: { step: TraceStep }) {
  const { skeleton_tokens, gap_regions, template_constants, promoted_first_instances } = step.data;

  return (
    <div style={{ marginTop: "0.75rem" }}>
      <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
        <div style={{ flex: "1 1 200px", padding: "12px", background: "rgba(56,168,86,0.06)", border: "1px solid rgba(56,168,86,0.2)", borderRadius: 6, textAlign: "center" }}>
          <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#2d7a3e" }}>{skeleton_tokens}</div>
          <div style={{ fontSize: "0.78rem", opacity: 0.7 }}>skeleton tokens</div>
          <div style={{ fontSize: "0.7rem", opacity: 0.5, marginTop: 2 }}>
            {template_constants} constants{promoted_first_instances > 0 ? ` + ${promoted_first_instances} promoted` : ""}
          </div>
        </div>
        <div style={{ flex: "1 1 200px", padding: "12px", background: "rgba(232,123,56,0.06)", border: "1px solid rgba(232,123,56,0.2)", borderRadius: 6, textAlign: "center" }}>
          <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#b5621e" }}>{gap_regions}</div>
          <div style={{ fontSize: "0.78rem", opacity: 0.7 }}>gap regions</div>
          <div style={{ fontSize: "0.7rem", opacity: 0.5, marginTop: 2 }}>
            analyzed for loops, optionals, variables
          </div>
        </div>
      </div>
      <div style={{ marginTop: 8, fontSize: "0.8rem", opacity: 0.6 }}>
        The skeleton forms the fixed frame of the template. Gaps between skeleton tokens contain variable content (loops, optionals, or data values) that differ across pages.
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main dispatcher                                                   */
/* ------------------------------------------------------------------ */

export default function ExalgPhaseViz({ step }: Props): React.ReactElement | null {
  switch (step.phase) {
    case "tokenization":
      return <TokenizationViz step={step} />;
    case "equivalence_classes":
      return <FindEqViz step={step} />;
    case "diffformat":
      return <DiffFormatViz step={step} />;
    case "handinv":
      return <HandInvViz step={step} />;
    case "diffeq":
      return <DiffEqViz step={step} />;
    case "skeleton":
      return <SkeletonViz step={step} />;
    default:
      return null;
  }
}
