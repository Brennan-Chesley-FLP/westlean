import React, { useMemo } from "react";
import { hierarchy, tree as d3tree } from "d3-hierarchy";
import { linkVertical } from "d3-shape";
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

interface Mismatch {
  path: string;
  field: string;
  template_value: string;
  page_value: string;
}

interface NewVar {
  path: string;
  field: string;
  var_name: string;
}

interface TreeNode {
  tag: string;
  path: string;
  textSlot?: SlotModel;
  textValue?: string;
  attrs: Record<string, SlotModel>;
  childrenVar: string | null;
  children: TreeNode[];
}

function tplToTree(node: TplNodeModel, path: string = ""): TreeNode {
  return {
    tag: node.tag,
    path,
    textSlot: node.text,
    attrs: node.attrs,
    childrenVar: node.children_var,
    children: node.children
      ? node.children.map((c, i) =>
          tplToTree(c, path ? `${path}/${i}` : String(i))
        )
      : [],
  };
}

const NODE_W = 90;
const NODE_H = 56;
const MARGIN = { top: 24, right: 16, bottom: 16, left: 16 };

function SingleTree({
  root,
  width,
  label,
  highlightPaths,
  highlightColor,
  annotations,
}: {
  root: TreeNode;
  width: number;
  label: string;
  highlightPaths: Set<string>;
  highlightColor: string;
  annotations: Map<string, string[]>;
}): React.ReactElement {
  const hier = hierarchy(root);
  const treeHeight = (hier.height + 1) * 80;
  const treeWidth = Math.max(hier.descendants().length * 100, 200);

  const layout = d3tree<TreeNode>().size([
    treeWidth - MARGIN.left - MARGIN.right,
    treeHeight - MARGIN.top - MARGIN.bottom,
  ]);
  const laid = layout(hier);
  const nodes = laid.descendants();
  const links = laid.links();
  const linkGen = linkVertical<any, any>()
    .x((d: any) => d.x)
    .y((d: any) => d.y);

  const svgW = treeWidth;
  const svgH = treeHeight + 40;

  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div
        style={{
          textAlign: "center",
          fontWeight: 600,
          fontSize: "0.85rem",
          marginBottom: 4,
          color: "var(--ifm-color-emphasis-700)",
        }}
      >
        {label}
      </div>
      <div style={{ overflowX: "auto" }}>
        <svg width={svgW} height={svgH} style={{ display: "block", margin: "0 auto" }}>
          <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
            {links.map((link, i) => (
              <path
                key={i}
                d={linkGen(link) || ""}
                fill="none"
                stroke="var(--ifm-color-emphasis-300)"
                strokeWidth={1.5}
              />
            ))}
            {nodes.map((node) => {
              const p = node.data.path;
              const isHighlighted = highlightPaths.has(p);
              const nodeAnnotations = annotations.get(p) || [];
              const slot = node.data.textSlot;
              const isVar = slot && !slot.is_fixed;

              let fillColor = "#6b7280"; // gray
              if (isHighlighted) fillColor = highlightColor;
              else if (isVar) fillColor = "#e87b38"; // variable orange
              else if (slot?.is_fixed && slot.value) fillColor = "#38a856"; // fixed green

              return (
                <g key={p} transform={`translate(${node.x}, ${node.y})`}>
                  <rect
                    x={-NODE_W / 2}
                    y={-NODE_H / 2}
                    width={NODE_W}
                    height={NODE_H}
                    rx={6}
                    fill={fillColor}
                    opacity={0.9}
                    stroke={isHighlighted ? highlightColor : "none"}
                    strokeWidth={isHighlighted ? 2 : 0}
                  />
                  <text
                    textAnchor="middle"
                    dy="-0.3em"
                    fill="white"
                    fontSize="12"
                    fontWeight="600"
                    fontFamily="monospace"
                  >
                    &lt;{node.data.tag}&gt;
                  </text>
                  {/* Show text value or var name */}
                  {slot && (slot.value || isVar) && (
                    <text
                      textAnchor="middle"
                      dy="1em"
                      fill="rgba(255,255,255,0.85)"
                      fontSize="9"
                      fontFamily="monospace"
                    >
                      {isVar
                        ? slot.value
                        : slot.value.length > 14
                          ? slot.value.slice(0, 12) + "…"
                          : slot.value}
                    </text>
                  )}
                  {/* Children var badge */}
                  {node.data.childrenVar && (
                    <text
                      textAnchor="middle"
                      dy="1em"
                      fill="rgba(255,255,255,0.85)"
                      fontSize="9"
                      fontFamily="monospace"
                    >
                      children→{node.data.childrenVar}
                    </text>
                  )}
                  {/* Annotations below node */}
                  {nodeAnnotations.map((ann, ai) => (
                    <text
                      key={ai}
                      textAnchor="middle"
                      dy={NODE_H / 2 + 12 + ai * 11}
                      fill="var(--ifm-color-emphasis-700)"
                      fontSize="9"
                      fontFamily="monospace"
                      fontWeight="600"
                    >
                      {ann}
                    </text>
                  ))}
                </g>
              );
            })}
          </g>
        </svg>
      </div>
    </div>
  );
}

interface AntiUnifTreeViewProps {
  step: TraceStep;
  allSteps: TraceStep[];
  currentIndex: number;
}

export default function AntiUnifTreeView({
  step,
  allSteps,
  currentIndex,
}: AntiUnifTreeViewProps): React.ReactElement | null {
  // Determine which template tree(s) to display based on current phase
  const phase = step.phase;

  if (phase === "initial_template") {
    const tpl: TplNodeModel = step.data.template;
    if (!tpl) return null;
    const tree = tplToTree(tpl);
    return (
      <div style={{ marginTop: "1rem" }}>
        <h4>Template (from page 1)</h4>
        <SingleTree
          root={tree}
          width={400}
          label="Initial Template (all fixed)"
          highlightPaths={new Set()}
          highlightColor=""
          annotations={new Map()}
        />
      </div>
    );
  }

  if (phase === "compare") {
    const tplBefore: TplNodeModel = step.data.template_before;
    const incomingPage: TplNodeModel = step.data.incoming_page;
    const mismatches: Mismatch[] = step.data.mismatches || [];
    if (!tplBefore || !incomingPage) return null;

    const treeBefore = tplToTree(tplBefore);
    const treePage = tplToTree(incomingPage);

    // Highlight nodes that have mismatches
    const mismatchPaths = new Set(mismatches.map((m) => m.path === "(root)" ? "" : m.path));

    // Annotations for mismatches
    const tplAnnotations = new Map<string, string[]>();
    const pageAnnotations = new Map<string, string[]>();
    for (const m of mismatches) {
      const key = m.path === "(root)" ? "" : m.path;
      const tplAnns = tplAnnotations.get(key) || [];
      tplAnns.push(`${m.field}: "${m.template_value.slice(0, 20)}"`);
      tplAnnotations.set(key, tplAnns);
      const pageAnns = pageAnnotations.get(key) || [];
      pageAnns.push(`${m.field}: "${m.page_value.slice(0, 20)}"`);
      pageAnnotations.set(key, pageAnns);
    }

    return (
      <div style={{ marginTop: "1rem" }}>
        <h4>
          Compare: template vs page {step.data.page_index + 1}
          {mismatches.length > 0 && (
            <span
              style={{
                marginLeft: 8,
                fontSize: "0.8rem",
                fontWeight: 400,
                color: "#d94a4a",
              }}
            >
              ({mismatches.length} mismatch{mismatches.length !== 1 ? "es" : ""})
            </span>
          )}
          {mismatches.length === 0 && (
            <span
              style={{
                marginLeft: 8,
                fontSize: "0.8rem",
                fontWeight: 400,
                color: "var(--ifm-color-success)",
              }}
            >
              (no new mismatches)
            </span>
          )}
        </h4>
        <div style={{ display: "flex", gap: "16px", alignItems: "start" }}>
          <SingleTree
            root={treeBefore}
            width={400}
            label="Current Template"
            highlightPaths={mismatchPaths}
            highlightColor="#d94a4a"
            annotations={tplAnnotations}
          />
          <SingleTree
            root={treePage}
            width={400}
            label={`Page ${step.data.page_index + 1}`}
            highlightPaths={mismatchPaths}
            highlightColor="#d94a4a"
            annotations={pageAnnotations}
          />
        </div>
      </div>
    );
  }

  if (phase === "fold") {
    const tplAfter: TplNodeModel = step.data.template_after;
    const newVars: NewVar[] = step.data.new_vars || [];
    if (!tplAfter) {
      return (
        <div style={{ marginTop: "1rem" }}>
          <h4>Fold failed — structure incompatible</h4>
        </div>
      );
    }

    const tree = tplToTree(tplAfter);
    const varPaths = new Set(newVars.map((v) => v.path === "(root)" ? "" : v.path));

    const annotations = new Map<string, string[]>();
    for (const v of newVars) {
      const key = v.path === "(root)" ? "" : v.path;
      const anns = annotations.get(key) || [];
      anns.push(`${v.field} → ${v.var_name}`);
      annotations.set(key, anns);
    }

    return (
      <div style={{ marginTop: "1rem" }}>
        <h4>
          Fold: merged with page {step.data.page_index + 1}
          {newVars.length > 0 && (
            <span
              style={{
                marginLeft: 8,
                fontSize: "0.8rem",
                fontWeight: 400,
                color: "#e87b38",
              }}
            >
              ({newVars.length} new variable{newVars.length !== 1 ? "s" : ""})
            </span>
          )}
        </h4>
        <SingleTree
          root={tree}
          width={600}
          label="Merged Template"
          highlightPaths={varPaths}
          highlightColor="#e87b38"
          annotations={annotations}
        />
      </div>
    );
  }

  if (phase === "region_detection") {
    const regions: Array<{
      path: string;
      backbone_tags: string[];
      repeating_regions: Array<{ after_backbone_pos: number; tag: string; var_name: string }>;
      optional_regions: Array<{ after_backbone_pos: number; tags: string[] }>;
      collapsed_to_variable?: string;
    }> = step.data.regions || [];

    return (
      <div style={{ marginTop: "1rem" }}>
        <h4>LCS-Based Child Alignment (Rigid UFOG)</h4>
        {regions.map((region, ri) => (
          <div
            key={ri}
            style={{
              padding: "12px",
              background: "var(--ifm-color-emphasis-100)",
              borderRadius: "6px",
              marginBottom: "8px",
              fontFamily: "monospace",
              fontSize: "0.85rem",
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: "4px" }}>
              Node: {region.path}
            </div>
            {region.collapsed_to_variable ? (
              <div style={{ color: "#b5621e" }}>
                Children collapsed to hedge variable: {region.collapsed_to_variable}
              </div>
            ) : (
              <>
                <div>
                  <span style={{ opacity: 0.7 }}>LCS backbone: </span>
                  {region.backbone_tags.length > 0
                    ? region.backbone_tags.map((t, i) => (
                        <span
                          key={i}
                          style={{
                            padding: "2px 6px",
                            borderRadius: "4px",
                            background: "rgba(56, 168, 86, 0.15)",
                            color: "#2d7a3e",
                            marginRight: "4px",
                            border: "1px solid rgba(56,168,86,0.3)",
                          }}
                        >
                          &lt;{t}&gt;
                        </span>
                      ))
                    : "(empty)"}
                </div>
                {region.repeating_regions.length > 0 && (
                  <div style={{ marginTop: "4px" }}>
                    <span style={{ opacity: 0.7 }}>Repeating (hedge vars): </span>
                    {region.repeating_regions.map((rr, i) => (
                      <span
                        key={i}
                        style={{
                          padding: "2px 6px",
                          borderRadius: "4px",
                          background: "rgba(232, 123, 56, 0.15)",
                          color: "#b5621e",
                          marginRight: "4px",
                          border: "1px solid rgba(232,123,56,0.3)",
                        }}
                      >
                        &lt;{rr.tag}&gt;* = {rr.var_name}
                      </span>
                    ))}
                  </div>
                )}
                {region.optional_regions.length > 0 && (
                  <div style={{ marginTop: "4px" }}>
                    <span style={{ opacity: 0.7 }}>Optional: </span>
                    {region.optional_regions.map((or_, i) => (
                      <span
                        key={i}
                        style={{
                          padding: "2px 6px",
                          borderRadius: "4px",
                          background: "rgba(100, 100, 220, 0.15)",
                          color: "#4a4aa0",
                          marginRight: "4px",
                          border: "1px solid rgba(100,100,220,0.3)",
                        }}
                      >
                        {or_.tags.map((t) => `<${t}>`).join(", ")}?
                      </span>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        ))}
      </div>
    );
  }

  if (phase === "result") {
    const root: TplNodeModel = step.data.root;
    if (!root) return null;
    const tree = tplToTree(root);
    return (
      <div style={{ marginTop: "1rem" }}>
        <h4>Final Template</h4>
        <SingleTree
          root={tree}
          width={600}
          label="Result"
          highlightPaths={new Set()}
          highlightColor=""
          annotations={new Map()}
        />
      </div>
    );
  }

  return null;
}
