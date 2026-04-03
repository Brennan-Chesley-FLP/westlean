import React, { useState, useCallback } from "react";
import { usePyodide } from "../hooks/usePyodide";
import type { TraceStep } from "../hooks/usePyodide";
import TracePlayer from "./TracePlayer";

export type VizRenderFn = (
  currentStep: TraceStep,
  stepIndex: number,
  allSteps: TraceStep[]
) => React.ReactNode;

interface AlgorithmDemoProps {
  algorithm: string;
  defaultPages: string[];
  /** Custom render function for algorithm-specific visualizations */
  renderViz?: VizRenderFn;
  /** Custom phase labels for the trace player */
  phaseLabels?: Record<string, string>;
}

export default function AlgorithmDemo({
  algorithm,
  defaultPages,
  renderViz,
  phaseLabels,
}: AlgorithmDemoProps): React.ReactElement {
  const { ready, loading, status, error, runAlgorithm } = usePyodide();
  const [pages, setPages] = useState<string[]>(defaultPages);
  const [traceSteps, setTraceSteps] = useState<TraceStep[] | null>(null);
  const [running, setRunning] = useState(false);

  const handleRun = useCallback(async () => {
    setRunning(true);
    setTraceSteps(null);
    try {
      const steps = await runAlgorithm(algorithm, pages);
      setTraceSteps(steps);
    } finally {
      setRunning(false);
    }
  }, [algorithm, pages, runAlgorithm]);

  const handlePageChange = useCallback((index: number, value: string) => {
    setPages((prev) => {
      const next = [...prev];
      next[index] = value;
      return next;
    });
  }, []);

  const addPage = useCallback(() => {
    setPages((prev) => [...prev, "<div></div>"]);
  }, []);

  const removePage = useCallback((index: number) => {
    setPages((prev) => prev.filter((_, i) => i !== index));
  }, []);

  return (
    <div
      style={{
        border: "1px solid var(--ifm-color-emphasis-300)",
        borderRadius: "8px",
        padding: "16px",
        marginTop: "1rem",
      }}
    >
      {/* Input area */}
      <h3 style={{ marginTop: 0 }}>Input HTML Pages</h3>
      {pages.map((page, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            gap: "8px",
            marginBottom: "8px",
            alignItems: "start",
          }}
        >
          <label
            style={{
              fontWeight: 600,
              minWidth: "60px",
              paddingTop: "6px",
              fontSize: "0.85rem",
            }}
          >
            Page {i + 1}
          </label>
          <textarea
            value={page}
            onChange={(e) => handlePageChange(i, e.target.value)}
            rows={2}
            style={{
              flex: 1,
              fontFamily: "monospace",
              fontSize: "0.85rem",
              padding: "6px 8px",
              borderRadius: "4px",
              border: "1px solid var(--ifm-color-emphasis-300)",
              resize: "vertical",
            }}
          />
          {pages.length > 2 && (
            <button
              onClick={() => removePage(i)}
              style={{ padding: "4px 8px" }}
              title="Remove page"
            >
              &times;
            </button>
          )}
        </div>
      ))}

      <div style={{ display: "flex", gap: "8px", marginBottom: "1rem" }}>
        <button onClick={addPage}>+ Add Page</button>
        <button
          onClick={handleRun}
          disabled={!ready || running || pages.length < 2}
          style={{
            background: ready
              ? "var(--ifm-color-primary)"
              : "var(--ifm-color-emphasis-300)",
            color: ready ? "white" : "var(--ifm-color-emphasis-600)",
            border: "none",
            padding: "6px 16px",
            borderRadius: "4px",
            cursor: ready ? "pointer" : "not-allowed",
            fontWeight: 600,
          }}
        >
          {running ? "Running..." : ready ? "Run Algorithm" : "Loading..."}
        </button>
      </div>

      {/* Status */}
      {loading && (
        <div
          style={{
            padding: "12px",
            background: "var(--ifm-color-emphasis-100)",
            borderRadius: "4px",
            marginBottom: "1rem",
          }}
        >
          {status}
        </div>
      )}

      {error && (
        <div
          style={{
            padding: "12px",
            background: "rgba(220, 50, 50, 0.1)",
            border: "1px solid var(--ifm-color-danger)",
            borderRadius: "4px",
            marginBottom: "1rem",
            color: "var(--ifm-color-danger)",
          }}
        >
          Error: {error}
        </div>
      )}

      {/* Trace visualization */}
      {traceSteps && traceSteps.length > 0 && (
        <TracePlayer steps={traceSteps} phaseLabels={phaseLabels}>
          {(currentStep, stepIndex) =>
            renderViz
              ? renderViz(currentStep, stepIndex, traceSteps)
              : null
          }
        </TracePlayer>
      )}
    </div>
  );
}
