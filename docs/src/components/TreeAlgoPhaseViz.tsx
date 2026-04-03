/**
 * Shared phase visualizations for tree-based algorithms (FiVaTech, k-Testable).
 *
 * Renders phases that the generic SequenceAlignment and SlotInspector
 * components don't cover: structure_check/structure_analysis, gap_analysis,
 * classification, generalization.
 */
import React from "react";
import type { TraceStep } from "../hooks/usePyodide";

interface Props {
  step: TraceStep;
  allSteps: TraceStep[];
  currentIndex: number;
}

/* ------------------------------------------------------------------ */
/*  structure_check / structure_analysis                               */
/* ------------------------------------------------------------------ */

function StructureViz({ step }: { step: TraceStep }) {
  const d = step.data;
  const tag = d.tag || d.root_tags?.[0] || "?";
  const pageCount = d.page_count || 0;
  const allMatch = d.all_match ?? d.root_tag_match;
  const attrMatch = d.attr_names_match;
  const childrenPerPage = d.children_per_page;
  const k = d.k;

  return (
    <div style={{ marginTop: "0.75rem" }}>
      <div
        style={{
          display: "flex",
          gap: 12,
          flexWrap: "wrap",
          alignItems: "center",
          marginBottom: 8,
        }}
      >
        <span
          style={{
            padding: "4px 12px",
            borderRadius: 6,
            background: allMatch ? "rgba(56,168,86,0.12)" : "rgba(239,68,68,0.12)",
            color: allMatch ? "#2d7a3e" : "#dc2626",
            fontWeight: 600,
            fontSize: "0.85rem",
          }}
        >
          {allMatch ? "Compatible" : "Incompatible"}
        </span>
        <span style={{ fontFamily: "monospace", fontSize: "0.9rem" }}>
          &lt;{tag}&gt;
        </span>
        <span style={{ opacity: 0.6, fontSize: "0.82rem" }}>
          {pageCount} pages
        </span>
        {k !== undefined && (
          <span style={{ opacity: 0.6, fontSize: "0.82rem" }}>k={k}</span>
        )}
      </div>

      {attrMatch !== undefined && (
        <div style={{ fontSize: "0.8rem", opacity: 0.7, marginBottom: 4 }}>
          Attribute names: {attrMatch ? "match across all pages" : "differ across pages"}
        </div>
      )}

      {childrenPerPage && (
        <div style={{ fontSize: "0.8rem", opacity: 0.7 }}>
          Children per page: {childrenPerPage.join(", ")}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  gap_analysis (FiVaTech)                                           */
/* ------------------------------------------------------------------ */

function GapAnalysisViz({ step }: { step: TraceStep }) {
  const { backbone, repeating_regions, optional_regions, has_repeating, has_optional } = step.data;

  return (
    <div style={{ marginTop: "0.75rem" }}>
      {/* Backbone */}
      {backbone && backbone.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontWeight: 600, fontSize: "0.8rem", marginBottom: 4 }}>
            Backbone ({backbone.length} elements):
          </div>
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
            {backbone.map((tag: string, i: number) => (
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

      {/* Repeating regions */}
      {has_repeating && (repeating_regions || []).length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontWeight: 600, fontSize: "0.8rem", marginBottom: 4 }}>
            Repeating Regions:
          </div>
          {repeating_regions.map((r: any, i: number) => (
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
              <span style={{ marginLeft: 8, opacity: 0.6, fontSize: "0.78rem" }}>
                counts: [{(r.counts || []).join(", ")}]
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Optional regions */}
      {has_optional && (optional_regions || []).length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontWeight: 600, fontSize: "0.8rem", marginBottom: 4 }}>
            Optional Regions:
          </div>
          {optional_regions.map((o: any, i: number) => (
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
              <span style={{ marginLeft: 8, opacity: 0.6, fontSize: "0.78rem" }}>
                present: [{(o.present_in || []).map((p: boolean) => p ? "yes" : "no").join(", ")}]
              </span>
            </div>
          ))}
        </div>
      )}

      {!has_repeating && !has_optional && (
        <div style={{ padding: 12, background: "var(--ifm-color-emphasis-100)", borderRadius: 6, fontSize: "0.85rem" }}>
          No repeating or optional regions found — all children are part of the fixed backbone.
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  classification (FiVaTech)                                         */
/* ------------------------------------------------------------------ */

function ClassificationViz({ step }: { step: TraceStep }) {
  const { fixed_count, variable_count, positions } = step.data;
  const posEntries: any[] = positions || [];

  return (
    <div style={{ marginTop: "0.75rem" }}>
      <div style={{ display: "flex", gap: 8, marginBottom: 8, fontSize: "0.82rem" }}>
        <span
          style={{
            padding: "2px 8px",
            borderRadius: 4,
            background: "rgba(56,168,86,0.15)",
            color: "#2d7a3e",
            fontWeight: 600,
          }}
        >
          {fixed_count} fixed
        </span>
        <span
          style={{
            padding: "2px 8px",
            borderRadius: 4,
            background: "rgba(232,123,56,0.15)",
            color: "#b5621e",
            fontWeight: 600,
          }}
        >
          {variable_count} variable
        </span>
      </div>

      {posEntries.length > 0 && (
        <div
          style={{
            maxHeight: 220,
            overflowY: "auto",
            borderRadius: 6,
            border: "1px solid var(--ifm-color-emphasis-200)",
          }}
        >
          <table style={{ width: "100%", fontSize: "0.78rem", borderCollapse: "collapse", margin: 0 }}>
            <thead>
              <tr style={{ background: "var(--ifm-color-emphasis-100)", position: "sticky", top: 0 }}>
                <th style={{ padding: "4px 8px", textAlign: "left" }}>Position</th>
                <th style={{ padding: "4px 8px", textAlign: "left" }}>Type</th>
                <th style={{ padding: "4px 8px", textAlign: "left" }}>Value / Var</th>
              </tr>
            </thead>
            <tbody>
              {posEntries.map((p: any, i: number) => (
                <tr key={i} style={{ borderTop: "1px solid var(--ifm-color-emphasis-200)" }}>
                  <td style={{ padding: "3px 8px", fontFamily: "monospace", fontSize: "0.74rem" }}>
                    {p.path || p.position}
                  </td>
                  <td
                    style={{
                      padding: "3px 8px",
                      color: p.fixed ? "#2d7a3e" : "#b5621e",
                      fontWeight: 600,
                    }}
                  >
                    {p.fixed ? "fixed" : "variable"}
                  </td>
                  <td style={{ padding: "3px 8px", fontSize: "0.74rem" }}>
                    {p.fixed ? (p.value ? `"${p.value}"` : '""') : p.var_name || "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  generalization (k-Testable)                                       */
/* ------------------------------------------------------------------ */

function GeneralizationViz({ step }: { step: TraceStep }) {
  const {
    repeating_regions_total,
    optional_regions_total,
    fixed_text_slots,
    variable_text_slots,
    fixed_attr_slots,
    variable_attr_slots,
    generalization_notes,
  } = step.data;

  const stats = [
    { label: "Repeating regions", value: repeating_regions_total, color: "#1d4ed8", bg: "rgba(59,130,246,0.08)" },
    { label: "Optional regions", value: optional_regions_total, color: "#7c3aed", bg: "rgba(139,92,246,0.08)" },
    { label: "Fixed text slots", value: fixed_text_slots, color: "#2d7a3e", bg: "rgba(56,168,86,0.08)" },
    { label: "Variable text slots", value: variable_text_slots, color: "#b5621e", bg: "rgba(232,123,56,0.08)" },
    { label: "Fixed attr slots", value: fixed_attr_slots, color: "#2d7a3e", bg: "rgba(56,168,86,0.08)" },
    { label: "Variable attr slots", value: variable_attr_slots, color: "#b5621e", bg: "rgba(232,123,56,0.08)" },
  ];

  return (
    <div style={{ marginTop: "0.75rem" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
          gap: 8,
          marginBottom: 8,
        }}
      >
        {stats.map((s, i) => (
          <div
            key={i}
            style={{
              padding: "8px 12px",
              borderRadius: 6,
              background: s.bg,
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: "1.2rem", fontWeight: 700, color: s.color }}>{s.value ?? 0}</div>
            <div style={{ fontSize: "0.72rem", opacity: 0.7 }}>{s.label}</div>
          </div>
        ))}
      </div>

      {(generalization_notes || []).length > 0 && (
        <div style={{ fontSize: "0.8rem", opacity: 0.7 }}>
          {generalization_notes.map((note: string, i: number) => (
            <div key={i}>• {note}</div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main dispatcher                                                   */
/* ------------------------------------------------------------------ */

export default function TreeAlgoPhaseViz({ step }: Props): React.ReactElement | null {
  switch (step.phase) {
    case "structure_check":
    case "structure_analysis":
      return <StructureViz step={step} />;
    case "gap_analysis":
      return <GapAnalysisViz step={step} />;
    case "classification":
      return <ClassificationViz step={step} />;
    case "generalization":
      return <GeneralizationViz step={step} />;
    default:
      return null;
  }
}
