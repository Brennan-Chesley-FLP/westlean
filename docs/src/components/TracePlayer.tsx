import React, { useState, useEffect, useRef, useCallback } from "react";
import type { TraceStep } from "../hooks/usePyodide";

interface TracePlayerProps {
  steps: TraceStep[];
  phaseLabels?: Record<string, string>;
  children: (currentStep: TraceStep, stepIndex: number) => React.ReactNode;
}

const DEFAULT_PHASE_LABELS: Record<string, string> = {
  // ExAlg
  tokenization: "Tokenize",
  equivalence_classes: "FindEq",
  diffformat: "DiffFormat",
  handinv: "HandInv",
  diffeq: "DiffEq",
  skeleton: "Skeleton",
  result: "Result",
  // Anti-Unification
  initial_template: "Initial",
  compare: "Compare",
  region_detection: "LCS Regions",
  fold: "Fold",
  // FiVaTech
  structure_check: "Structure",
  child_sequences: "Sequences",
  backbone_computation: "Backbone",
  alignment: "Alignment",
  gap_analysis: "Gaps",
  recursive_merge: "Merge",
  // RoadRunner
  linearization: "Linearize",
  ufre_init: "UFRE Init",
  acme_fold: "ACME Fold",
  repeat_detection: "Repeat Detect",
  // k-Testable Tree Automata
  structure_analysis: "Structure",
  template_tree: "Template",
  generalization: "Generalize",
  // k-Testable (legacy phases)
  pattern_extraction: "Patterns",
  pattern_union: "Union",
  root_patterns: "Roots",
};

export default function TracePlayer({
  steps,
  phaseLabels,
  children,
}: TracePlayerProps): React.ReactElement {
  const labels = { ...DEFAULT_PHASE_LABELS, ...phaseLabels };
  const [currentIndex, setCurrentIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const maxIndex = steps.length - 1;
  const currentStep = steps[currentIndex];

  useEffect(() => {
    setCurrentIndex(0);
    setPlaying(false);
  }, [steps]);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setCurrentIndex((prev) => {
          if (prev >= maxIndex) {
            setPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 1500);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, maxIndex]);

  const stepBack = useCallback(
    () => setCurrentIndex((i) => Math.max(0, i - 1)),
    []
  );
  const stepForward = useCallback(
    () => setCurrentIndex((i) => Math.min(maxIndex, i + 1)),
    []
  );

  if (!currentStep) return <div>No trace data</div>;

  return (
    <div style={{ marginTop: "1rem" }}>
      {/* Phase timeline */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "4px",
          marginBottom: "1rem",
          flexWrap: "wrap",
        }}
      >
        {steps.map((step, i) => (
          <button
            key={i}
            onClick={() => setCurrentIndex(i)}
            style={{
              padding: "4px 12px",
              borderRadius: "16px",
              border:
                i === currentIndex
                  ? "2px solid var(--ifm-color-primary)"
                  : "1px solid var(--ifm-color-emphasis-300)",
              background:
                i === currentIndex
                  ? "var(--ifm-color-primary)"
                  : i < currentIndex
                    ? "var(--ifm-color-emphasis-200)"
                    : "transparent",
              color:
                i === currentIndex
                  ? "white"
                  : "var(--ifm-font-color-base)",
              cursor: "pointer",
              fontSize: "0.85rem",
              fontWeight: i === currentIndex ? 600 : 400,
            }}
          >
            {labels[step.phase] || step.phase}
          </button>
        ))}
      </div>

      {/* Controls */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          marginBottom: "1rem",
        }}
      >
        <button onClick={stepBack} disabled={currentIndex === 0}>
          &larr; Back
        </button>
        <button onClick={() => setPlaying(!playing)}>
          {playing ? "Pause" : "Play"}
        </button>
        <button onClick={stepForward} disabled={currentIndex === maxIndex}>
          Forward &rarr;
        </button>
        <span style={{ marginLeft: "8px", color: "var(--ifm-color-emphasis-600)" }}>
          Step {currentIndex + 1} of {steps.length}
        </span>
      </div>

      {/* Description */}
      <div
        style={{
          padding: "8px 12px",
          background: "var(--ifm-color-emphasis-100)",
          borderRadius: "4px",
          marginBottom: "1rem",
          fontSize: "0.9rem",
        }}
      >
        <strong>{labels[currentStep.phase] || currentStep.phase}:</strong>{" "}
        {currentStep.description}
      </div>

      {/* Visualization content */}
      {children(currentStep, currentIndex)}
    </div>
  );
}
