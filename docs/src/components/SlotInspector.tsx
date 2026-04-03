import React from "react";
import type { TraceStep } from "../hooks/usePyodide";

interface SlotModel {
  is_fixed: boolean;
  value: string;
}

interface TplNodeModel {
  tag: string;
  attr_names: string[];
  text: SlotModel;
  tail: SlotModel;
  attrs: Record<string, SlotModel>;
  children: TplNodeModel[] | null;
  children_var: string | null;
  text_always_present: boolean;
  tail_always_present: boolean;
}

interface SlotInspectorProps {
  step: TraceStep;
  allSteps: TraceStep[];
  currentIndex: number;
}

/** Flatten a TplNode tree into a list of rows for the table. */
function flattenNodes(
  node: TplNodeModel,
  path: string = ""
): Array<{ path: string; tag: string; node: TplNodeModel }> {
  const rows: Array<{ path: string; tag: string; node: TplNodeModel }> = [];
  rows.push({ path: path || "(root)", tag: node.tag, node });
  if (node.children) {
    for (let i = 0; i < node.children.length; i++) {
      const childPath = path ? `${path}/${i}` : String(i);
      rows.push(...flattenNodes(node.children[i], childPath));
    }
  }
  return rows;
}

function SlotBadge({ slot }: { slot: SlotModel }): React.ReactElement {
  return (
    <span
      style={{
        padding: "2px 6px",
        borderRadius: "4px",
        fontSize: "0.8rem",
        fontFamily: "monospace",
        background: slot.is_fixed
          ? "rgba(56, 168, 86, 0.15)"
          : "rgba(232, 123, 56, 0.15)",
        color: slot.is_fixed ? "#2d7a3e" : "#b5621e",
        border: `1px solid ${slot.is_fixed ? "rgba(56,168,86,0.3)" : "rgba(232,123,56,0.3)"}`,
      }}
    >
      {slot.is_fixed ? `"${slot.value}"` : slot.value}
    </span>
  );
}

export default function SlotInspector({
  step,
  allSteps,
  currentIndex,
}: SlotInspectorProps): React.ReactElement | null {
  // Find the latest template state from initial_template, pairwise_fold, or recursive_merge
  let templateData: TplNodeModel | null = null;

  for (let i = currentIndex; i >= 0; i--) {
    const s = allSteps[i];
    if (s.phase === "initial_template" && s.data.template) {
      templateData = s.data.template;
      break;
    }
    if (s.phase === "pairwise_fold" && s.data.template_after) {
      templateData = s.data.template_after;
      break;
    }
    if (s.phase === "recursive_merge" && s.data.pattern_tree) {
      templateData = s.data.pattern_tree;
      break;
    }
    if (s.phase === "result" && s.data.root) {
      templateData = s.data.root;
      break;
    }
  }

  if (!templateData) return null;

  const rows = flattenNodes(templateData);

  return (
    <div style={{ overflowX: "auto", marginTop: "1rem" }}>
      <h4>Template Slots</h4>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: "0.82rem",
        }}
      >
        <thead>
          <tr>
            {["Path", "Tag", "Text", "Tail", "Attributes", "Children"].map(
              (h) => (
                <th
                  key={h}
                  style={{
                    textAlign: "left",
                    padding: "6px 8px",
                    borderBottom: "2px solid var(--ifm-color-emphasis-300)",
                    whiteSpace: "nowrap",
                  }}
                >
                  {h}
                </th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ path, tag, node }) => (
            <tr key={path}>
              <td
                style={{
                  padding: "4px 8px",
                  fontFamily: "monospace",
                  borderBottom: "1px solid var(--ifm-color-emphasis-200)",
                  whiteSpace: "nowrap",
                }}
              >
                {path}
              </td>
              <td
                style={{
                  padding: "4px 8px",
                  fontWeight: 600,
                  borderBottom: "1px solid var(--ifm-color-emphasis-200)",
                }}
              >
                &lt;{tag}&gt;
              </td>
              <td
                style={{
                  padding: "4px 8px",
                  borderBottom: "1px solid var(--ifm-color-emphasis-200)",
                }}
              >
                {node.text.value || node.text_always_present ? (
                  <SlotBadge slot={node.text} />
                ) : (
                  <span style={{ opacity: 0.3 }}>--</span>
                )}
              </td>
              <td
                style={{
                  padding: "4px 8px",
                  borderBottom: "1px solid var(--ifm-color-emphasis-200)",
                }}
              >
                {node.tail.value || node.tail_always_present ? (
                  <SlotBadge slot={node.tail} />
                ) : (
                  <span style={{ opacity: 0.3 }}>--</span>
                )}
              </td>
              <td
                style={{
                  padding: "4px 8px",
                  borderBottom: "1px solid var(--ifm-color-emphasis-200)",
                }}
              >
                {Object.entries(node.attrs).length > 0 ? (
                  <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                    {Object.entries(node.attrs).map(([name, slot]) => (
                      <span key={name}>
                        <span style={{ opacity: 0.6 }}>{name}=</span>
                        <SlotBadge slot={slot} />
                      </span>
                    ))}
                  </div>
                ) : (
                  <span style={{ opacity: 0.3 }}>--</span>
                )}
              </td>
              <td
                style={{
                  padding: "4px 8px",
                  borderBottom: "1px solid var(--ifm-color-emphasis-200)",
                }}
              >
                {node.children === null ? (
                  <span
                    style={{
                      padding: "2px 6px",
                      borderRadius: "4px",
                      fontSize: "0.8rem",
                      background: "rgba(232, 123, 56, 0.15)",
                      color: "#b5621e",
                      border: "1px solid rgba(232,123,56,0.3)",
                    }}
                  >
                    {node.children_var || "variable"}
                  </span>
                ) : (
                  <span style={{ opacity: 0.5 }}>
                    {node.children.length} child
                    {node.children.length !== 1 ? "ren" : ""}
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
