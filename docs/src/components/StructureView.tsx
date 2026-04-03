import React from "react";
import type { TraceStep } from "../hooks/usePyodide";

interface StructureViewProps {
  step: TraceStep;
}

interface TagStruct {
  tag: string;
  attrs: string[];
}

export default function StructureView({
  step,
}: StructureViewProps): React.ReactElement | null {
  if (step.phase !== "structure_validation") return null;

  const { structures, all_match, page_count } = step.data;

  // Show the first page's structure as a tree
  const struct: Record<string, TagStruct> = structures?.["0"] || {};
  const paths = Object.keys(struct).sort();

  return (
    <div style={{ marginTop: "1rem" }}>
      <h4>
        Tag Structure{" "}
        {all_match ? (
          <span style={{ color: "var(--ifm-color-success)", fontWeight: 400 }}>
            ({page_count} pages match)
          </span>
        ) : (
          <span style={{ color: "var(--ifm-color-danger)", fontWeight: 400 }}>
            (structures differ!)
          </span>
        )}
      </h4>
      <div
        style={{
          fontFamily: "monospace",
          fontSize: "0.85rem",
          lineHeight: "1.6",
          padding: "12px",
          background: "var(--ifm-color-emphasis-100)",
          borderRadius: "4px",
        }}
      >
        {paths.map((path) => {
          const { tag, attrs } = struct[path];
          const depth = path === "" ? 0 : path.split("/").length;
          const indent = "  ".repeat(depth);
          const attrStr =
            attrs.length > 0
              ? " " + attrs.map((a: string) => `${a}="..."`).join(" ")
              : "";
          return (
            <div key={path}>
              {indent}&lt;{tag}
              {attrStr}&gt;
              <span style={{ opacity: 0.4, marginLeft: "8px" }}>
                {path || "(root)"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
