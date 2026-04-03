import React, { useMemo } from "react";
import { hierarchy, tree as d3tree } from "d3-hierarchy";
import { linkVertical } from "d3-shape";
import type { TraceStep } from "../hooks/usePyodide";

interface TagStruct {
  tag: string;
  attrs: string[];
}

interface TreeNode {
  name: string;
  path: string;
  attrs: string[];
  children: TreeNode[];
}

interface DOMTreeVizProps {
  step: TraceStep;
  allSteps: TraceStep[];
  currentIndex: number;
}

/** Convert the flat path→{tag, attrs} map into a hierarchy. */
function buildTree(
  struct: Record<string, TagStruct>
): TreeNode | null {
  const root = struct[""];
  if (!root) return null;

  const node: TreeNode = {
    name: root.tag,
    path: "",
    attrs: root.attrs,
    children: [],
  };

  // Group paths by depth and parent
  const paths = Object.keys(struct)
    .filter((p) => p !== "")
    .sort((a, b) => {
      // Sort by depth first, then lexicographically within depth
      const da = a.split("/").length;
      const db = b.split("/").length;
      if (da !== db) return da - db;
      return a.localeCompare(b);
    });

  // Build a lookup from path to TreeNode
  const lookup: Record<string, TreeNode> = { "": node };

  for (const path of paths) {
    const info = struct[path];
    const child: TreeNode = {
      name: info.tag,
      path,
      attrs: info.attrs,
      children: [],
    };
    lookup[path] = child;

    // Find parent: "0/1/2" → "0/1", "0" → ""
    const parts = path.split("/");
    parts.pop();
    const parentPath = parts.join("/");
    const parent = lookup[parentPath] || lookup[""];
    parent.children.push(child);
  }

  return node;
}

const NODE_WIDTH = 80;
const NODE_HEIGHT = 36;
const MARGIN = { top: 20, right: 20, bottom: 20, left: 20 };

export default function DOMTreeViz({
  step,
  allSteps,
  currentIndex,
}: DOMTreeVizProps): React.ReactElement | null {
  // Get structure from the structure_validation step (always first)
  const structStep = allSteps.find((s) => s.phase === "structure_validation");
  if (!structStep) return null;

  const struct: Record<string, TagStruct> =
    structStep.data.structures?.["0"] || {};

  // Get classification data if available
  const classStep = allSteps.find((s) => s.phase === "classification");
  const hasClassification =
    classStep && allSteps.indexOf(classStep) <= currentIndex;
  const fixed: Record<string, string> = hasClassification
    ? classStep!.data.fixed || {}
    : {};
  const variant: Record<string, string> = hasClassification
    ? classStep!.data.variant || {}
    : {};

  // Get frequency data if available
  const freqStep = allSteps.find((s) => s.phase === "frequency_analysis");
  const hasFrequency = freqStep && allSteps.indexOf(freqStep) <= currentIndex;
  const perKey: Record<string, { all_same: boolean }> = hasFrequency
    ? freqStep!.data.per_key || {}
    : {};

  // Get position map data if available
  const mapStep = allSteps.find((s) => s.phase === "position_mapping");
  const hasPositions = mapStep && allSteps.indexOf(mapStep) <= currentIndex;
  const positionMaps: Record<string, string>[] = hasPositions
    ? mapStep!.data.position_maps || []
    : [];

  const treeData = useMemo(() => buildTree(struct), [struct]);
  if (!treeData) return null;

  const root = hierarchy(treeData);
  const nodeCount = root.descendants().length;
  const treeHeight = (root.height + 1) * 80;
  const treeWidth = Math.max(nodeCount * 90, 300);

  const layout = d3tree<TreeNode>().size([
    treeWidth - MARGIN.left - MARGIN.right,
    treeHeight - MARGIN.top - MARGIN.bottom,
  ]);

  const laidOut = layout(root);
  const nodes = laidOut.descendants();
  const links = laidOut.links();

  const svgWidth = treeWidth;
  const svgHeight = treeHeight;

  const linkGen = linkVertical<any, any>()
    .x((d: any) => d.x)
    .y((d: any) => d.y);

  /** Determine the color for a node based on current phase */
  function nodeColor(path: string): string {
    if (hasClassification) {
      // Check if any position under this node is fixed or variable
      const fixedKeys = Object.keys(fixed).filter(
        (k) => k.startsWith(path + "/") || k.startsWith(path + "/@") ||
               (path === "" && !k.includes("/"))
      );
      const varKeys = Object.keys(variant).filter(
        (k) => k.startsWith(path + "/") || k.startsWith(path + "/@") ||
               (path === "" && !k.includes("/"))
      );
      if (varKeys.length > 0 && fixedKeys.length > 0) return "#e8a838"; // mixed
      if (varKeys.length > 0) return "#e87b38"; // variable
      if (fixedKeys.length > 0) return "#38a856"; // fixed
    }
    if (hasFrequency) {
      const relevantKeys = Object.keys(perKey).filter(
        (k) => k.startsWith(path + "/") || k.startsWith(path + "/@") ||
               (path === "" && !k.includes("/"))
      );
      const allSame = relevantKeys.length > 0 &&
        relevantKeys.every((k) => perKey[k]?.all_same);
      if (relevantKeys.length > 0) {
        return allSame ? "rgba(56, 168, 86, 0.6)" : "rgba(232, 123, 56, 0.6)";
      }
    }
    return "#6b7280"; // gray default
  }

  /** Get position annotations for a node */
  function nodeAnnotations(path: string): string[] {
    if (!hasClassification) return [];
    const annotations: string[] = [];
    for (const [key, val] of Object.entries(fixed)) {
      if (
        key.startsWith(path + "/") || key.startsWith(path + "/@") ||
        (path === "" && !key.includes("/"))
      ) {
        const shortKey = path === "" ? key : key.slice(path.length + 1);
        annotations.push(`${shortKey} = "${val}"`);
      }
    }
    for (const [key, varName] of Object.entries(variant)) {
      if (
        key.startsWith(path + "/") || key.startsWith(path + "/@") ||
        (path === "" && !key.includes("/"))
      ) {
        const shortKey = path === "" ? key : key.slice(path.length + 1);
        annotations.push(`${shortKey} → ${varName}`);
      }
    }
    return annotations;
  }

  return (
    <div style={{ marginTop: "1rem" }}>
      <h4>DOM Tree Structure</h4>
      <div style={{ overflowX: "auto" }}>
        <svg
          width={svgWidth}
          height={svgHeight + 60}
          style={{ display: "block", margin: "0 auto" }}
        >
          <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
            {/* Links */}
            {links.map((link, i) => (
              <path
                key={i}
                d={linkGen(link) || ""}
                fill="none"
                stroke="var(--ifm-color-emphasis-400)"
                strokeWidth={1.5}
              />
            ))}

            {/* Nodes */}
            {nodes.map((node) => {
              const color = nodeColor(node.data.path);
              const annotations = nodeAnnotations(node.data.path);
              const attrStr =
                node.data.attrs.length > 0
                  ? ` [${node.data.attrs.join(", ")}]`
                  : "";

              return (
                <g key={node.data.path} transform={`translate(${node.x}, ${node.y})`}>
                  {/* Node background */}
                  <rect
                    x={-NODE_WIDTH / 2}
                    y={-NODE_HEIGHT / 2}
                    width={NODE_WIDTH}
                    height={NODE_HEIGHT}
                    rx={6}
                    fill={color}
                    opacity={0.9}
                  />

                  {/* Tag name */}
                  <text
                    textAnchor="middle"
                    dy="0.35em"
                    fill="white"
                    fontSize="13"
                    fontWeight="600"
                    fontFamily="monospace"
                  >
                    &lt;{node.data.name}&gt;
                  </text>

                  {/* Attributes badge */}
                  {node.data.attrs.length > 0 && (
                    <text
                      textAnchor="middle"
                      dy={NODE_HEIGHT / 2 + 12}
                      fill="var(--ifm-color-emphasis-600)"
                      fontSize="10"
                      fontFamily="monospace"
                    >
                      {attrStr}
                    </text>
                  )}

                  {/* Classification annotations */}
                  {annotations.map((ann, i) => (
                    <text
                      key={i}
                      textAnchor="middle"
                      dy={NODE_HEIGHT / 2 + 24 + i * 12}
                      fill="var(--ifm-color-emphasis-700)"
                      fontSize="9"
                      fontFamily="monospace"
                    >
                      {ann.length > 25 ? ann.slice(0, 22) + "..." : ann}
                    </text>
                  ))}
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      {/* Legend */}
      {(hasFrequency || hasClassification) && (
        <div
          style={{
            display: "flex",
            gap: "16px",
            justifyContent: "center",
            fontSize: "0.8rem",
            marginTop: "8px",
          }}
        >
          <span>
            <span
              style={{
                display: "inline-block",
                width: 12,
                height: 12,
                borderRadius: 3,
                background: hasClassification ? "#38a856" : "rgba(56, 168, 86, 0.6)",
                marginRight: 4,
                verticalAlign: "middle",
              }}
            />
            Fixed
          </span>
          <span>
            <span
              style={{
                display: "inline-block",
                width: 12,
                height: 12,
                borderRadius: 3,
                background: hasClassification ? "#e87b38" : "rgba(232, 123, 56, 0.6)",
                marginRight: 4,
                verticalAlign: "middle",
              }}
            />
            Variable
          </span>
          {hasClassification && (
            <span>
              <span
                style={{
                  display: "inline-block",
                  width: 12,
                  height: 12,
                  borderRadius: 3,
                  background: "#e8a838",
                  marginRight: 4,
                  verticalAlign: "middle",
                }}
              />
              Mixed
            </span>
          )}
        </div>
      )}
    </div>
  );
}
