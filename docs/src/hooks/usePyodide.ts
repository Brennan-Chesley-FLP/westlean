import { useState, useRef, useCallback, useEffect } from "react";
import useBaseUrl from "@docusaurus/useBaseUrl";

export interface TraceStep {
  algorithm: string;
  phase: string;
  step_index: number;
  description: string;
  data: Record<string, any>;
}

interface PyodideState {
  ready: boolean;
  loading: boolean;
  status: string;
  error: string | null;
  runAlgorithm: (
    algorithm: string,
    pagesHtml: string[]
  ) => Promise<TraceStep[]>;
}

export function usePyodide(): PyodideState {
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const resolveRef = useRef<((steps: TraceStep[]) => void) | null>(null);
  const rejectRef = useRef<((error: Error) => void) | null>(null);
  const workerUrl = useBaseUrl("/pyodideWorker.js");

  useEffect(() => {
    // Load worker from static/ (bypasses webpack bundling)
    const worker = new Worker(workerUrl);
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent) => {
      const { type, message, steps } = event.data;

      switch (type) {
        case "status":
          setStatus(message);
          break;
        case "ready":
          setReady(true);
          setLoading(false);
          setStatus("Ready");
          break;
        case "trace":
          resolveRef.current?.(steps);
          resolveRef.current = null;
          rejectRef.current = null;
          break;
        case "error":
          setError(message);
          rejectRef.current?.(new Error(message));
          resolveRef.current = null;
          rejectRef.current = null;
          break;
      }
    };

    setLoading(true);
    worker.postMessage({ type: "init" });

    return () => {
      worker.terminate();
    };
  }, [workerUrl]);

  const runAlgorithm = useCallback(
    (algorithm: string, pagesHtml: string[]): Promise<TraceStep[]> => {
      return new Promise((resolve, reject) => {
        if (!workerRef.current || !ready) {
          reject(new Error("Pyodide not ready"));
          return;
        }
        resolveRef.current = resolve;
        rejectRef.current = reject;
        setError(null);
        workerRef.current.postMessage({
          type: "run",
          algorithm,
          pagesHtml,
        });
      });
    },
    [ready]
  );

  return { ready, loading, status, error, runAlgorithm };
}
