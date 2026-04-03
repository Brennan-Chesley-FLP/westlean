/**
 * Web Worker that runs Python algorithms via Pyodide.
 * Served from static/ to bypass webpack bundling.
 */

let pyodide = null;

async function initPyodide() {
  self.postMessage({ type: "status", message: "Loading Pyodide..." });

  importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js");

  pyodide = await loadPyodide();

  self.postMessage({ type: "status", message: "Loading lxml..." });
  await pyodide.loadPackage("lxml");

  self.postMessage({ type: "status", message: "Loading pydantic..." });
  await pyodide.loadPackage("pydantic");

  self.postMessage({ type: "status", message: "Loading micropip..." });
  await pyodide.loadPackage("micropip");

  self.postMessage({ type: "status", message: "Installing westlean..." });
  const micropip = pyodide.pyimport("micropip");
  const wheelUrl = self.location.origin + "/westlean-0.1.0-py3-none-any.whl";
  await micropip.install(wheelUrl);

  self.postMessage({ type: "ready" });
}

async function runAlgorithm(algorithm, pagesHtml) {
  if (!pyodide) {
    throw new Error("Pyodide not initialized");
  }

  const infererMap = {
    exalg:
      "from westlean.algorithms.tracing_exalg import TracingExalgInferer as Inferer",
    anti_unification:
      "from westlean.algorithms.tracing_anti_unification import TracingAntiUnificationInferer as Inferer",
    fivatech:
      "from westlean.algorithms.tracing_fivatech import TracingFiVaTechInferer as Inferer",
    roadrunner:
      "from westlean.algorithms.tracing_roadrunner import TracingRoadRunnerInferer as Inferer",
    k_testable:
      "from westlean.algorithms.tracing_tree_automata import TracingKTestableInferer as Inferer",
  };

  const importLine = infererMap[algorithm];
  if (!importLine) {
    throw new Error("Unknown algorithm: " + algorithm);
  }

  // Pass data via Pyodide's JS→Python bridge to avoid string escaping issues
  pyodide.globals.set("_pages_html", pyodide.toPy(pagesHtml));

  const result = await pyodide.runPythonAsync(`
import json
from lxml.html import fragment_fromstring
from westlean.tracer import tracing
${importLine}

pages = [fragment_fromstring(html) for html in _pages_html]

with tracing() as _tracer:
    _inferer = Inferer()
    _template = _inferer.infer(pages)

json.dumps(_tracer.to_json())
`);

  return JSON.parse(result);
}

self.onmessage = async function (event) {
  const { type, algorithm, pagesHtml } = event.data;

  try {
    if (type === "init") {
      await initPyodide();
    } else if (type === "run") {
      const steps = await runAlgorithm(algorithm, pagesHtml);
      self.postMessage({ type: "trace", steps });
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
