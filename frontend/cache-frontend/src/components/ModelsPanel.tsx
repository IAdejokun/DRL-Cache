// frontend/src/components/ModelsPanel.tsx
import { Fragment, useEffect, useState } from "react";
import {
  models as fetchModels,
  trainModel,
  evaluateModel,
  getModelReport,
  getActiveModel,
  promoteModel,
  resetModelRegistry,
  type ModelEntry,
  type EvalRow,
  type EvaluateResponse,
} from "../lib/api";

export default function ModelsPanel() {
  const [list, setList] = useState<ModelEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [trace, setTrace] = useState("/simulator/data/run_1.csv");
  const [epochs, setEpochs] = useState(20);
  const [status, setStatus] = useState<string | null>(null);
  const [expandedModelId, setExpandedModelId] = useState<string | null>(null);
  const [reportData, setReportData] = useState<Record<string, EvalRow[]>>({});
  const [active, setActive] = useState<{ path: string | null; loaded: boolean } | null>(null);


  const load = async () => {
  setLoading(true);
  try {
    const [data, activeInfo] = await Promise.all([fetchModels(), getActiveModel()]);
    setList(data);
    setActive(activeInfo);
  } catch (e) {
    console.error(e);
    setStatus(String(e));
  } finally {
    setLoading(false);
  }
};

  const onPromote = async (id: string) => {
  setStatus("promoting...");
  try {
    await promoteModel({ model_id: id });
    const a = await getActiveModel();
    setActive(a);
    setStatus("model promoted and reloaded");
  } catch (e) {
    setStatus("promote error: " + String(e));
  }
};


  useEffect(() => {
    load();
  }, []);

  const onTrain = async () => {
    setStatus("starting training...");
    try {
      const res = await trainModel({ trace, epochs });
      setStatus(`training started: out=${res.out_name}`);
      setTimeout(load, 1200);
    } catch (e) {
      setStatus("train error: " + String(e));
    }
  };

  const onEvaluate = async (model: ModelEntry) => {
    setStatus("evaluating...");
    try {
      // Send model path explicitly (backend also supports model_id)
      const payload = { path: model.path, trace, plot: false };
      const res: EvaluateResponse = await evaluateModel(payload);
      setStatus(`eval done; csv=${res.eval_log}`);

      // Immediately pull the parsed report for this model
      try {
        const r = await getModelReport(model.id);
        if (Array.isArray(r.rows)) {
          setReportData((prev) => ({ ...prev, [model.id]: r.rows }));
          setExpandedModelId(model.id);
        } else {
          setReportData((prev) => ({ ...prev, [model.id]: [] }));
          setExpandedModelId(model.id);
        }
      } catch (err) {
        setStatus("Report fetch error: " + String(err));
      }

      setTimeout(load, 800);
    } catch (e) {
      setStatus("eval error: " + String(e));
    }
  };

  const onViewReport = async (modelId: string) => {
    if (expandedModelId === modelId) {
      setExpandedModelId(null);
      return;
    }
    setStatus("fetching report...");
    try {
      const res = await getModelReport(modelId);
      if (Array.isArray(res.rows)) {
        setReportData((prev) => ({ ...prev, [modelId]: res.rows }));
      } else {
        setReportData((prev) => ({ ...prev, [modelId]: [] }));
      }
      setExpandedModelId(modelId);
      setStatus(null);
    } catch (e) {
      setStatus("report error: " + String(e));
    }
  };

  const renderReportBlock = (modelId: string) => {
  const rows = reportData[modelId];

  // Dark-friendly container so text always has contrast
  return (
    <tr key={`${modelId}-report`}>
      <td colSpan={4} style={{ background: "#1f1f1f", padding: 10 }}>
        {rows === undefined ? (
          <div style={{ padding: 8, color: "#ddd", fontStyle: "italic" }}>
            No report loaded.
          </div>
        ) : rows.length === 0 ? (
          <div style={{ padding: 8, color: "#ddd", fontStyle: "italic" }}>
            Report has no rows yet. If you just evaluated, click “View Report” again in a moment.
          </div>
        ) : (
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              background: "#222",      // dark background
              color: "#fff",           // visible text on dark
              border: "1px solid #444",
              borderRadius: 6,
              overflow: "hidden",
            }}
          >
            <thead>
              <tr>
                {["Policy", "Total", "Hits", "Hit Ratio %", "Avg Latency (ms)", "Stale %"].map((h) => (
                  <th
                    key={h}
                    style={{
                      textAlign: h === "Policy" ? "left" : "right",
                      padding: 8,
                      borderBottom: "1px solid #333",
                      fontWeight: 600,
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={`${modelId}-r-${i}`}>
                  <td style={{ padding: 8, borderBottom: "1px solid #333", textAlign: "left" }}>
                    {row.policy}
                  </td>
                  <td style={{ padding: 8, borderBottom: "1px solid #333", textAlign: "right" }}>
                    {row.total}
                  </td>
                  <td style={{ padding: 8, borderBottom: "1px solid #333", textAlign: "right" }}>
                    {row.hits}
                  </td>
                  <td style={{ padding: 8, borderBottom: "1px solid #333", textAlign: "right" }}>
                    {row.hit_ratio_pct}
                  </td>
                  <td style={{ padding: 8, borderBottom: "1px solid #333", textAlign: "right" }}>
                    {row.avg_latency_ms}
                  </td>
                  <td style={{ padding: 8, borderBottom: "1px solid #333", textAlign: "right" }}>
                    {row.stale_pct}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </td>
    </tr>
  );
};


  return (
    <div style={{ padding: 12 }}>
      <h3>Model Registry & Training</h3>
      <div style={{ marginBottom: 8 }}>
        <label>Trace file: </label>
        <input value={trace} onChange={(e) => setTrace(e.target.value)} style={{ width: 360 }} />
      </div>
      <div style={{ marginBottom: 8 }}>
        <label>Epochs: </label>
        <input type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} style={{ width: 100 }} />
      </div>
      <button onClick={onTrain}>Start Training</button>{" "}
      <button
        onClick={async () => {
          setStatus("evaluating TTL only...");
          try {
            // If you want a TTL-only button, point to your /api/models/evaluate with a dummy path or separate TTL endpoint.
            // For now we just call evaluate with trace and an empty path (backend will ignore model specifics).
            const res: EvaluateResponse = await evaluateModel({ trace, plot: false, path: "" });
            setStatus(`ttl-only eval done; csv=${res.eval_log}`);
          } catch (e) {
            setStatus("eval error: " + String(e));
          }
        }}
      >
        Evaluate TTL only
      </button>{" "}
      <button
        onClick={async () => {
          try {
            await resetModelRegistry();
            setReportData({});
            setExpandedModelId(null);
            setStatus("Registry cleared");
            await load();
          } catch (e) {
            setStatus("Reset error: " + String(e));
          }
        }}
      >
        Clear Registry
      </button>

      <div style={{ marginTop: 12 }}>
        <strong>Status:</strong> {status ?? "idle"}
      </div>

    <div style={{ marginTop: 12 }}>
  <strong>Active model:</strong>{" "}
  {active?.path ? <span title={active.path}>{active.loaded ? "Loaded" : "Not loaded"} — {active.path}</span> : "None"}
  </div>

      
      <h4 style={{ marginTop: 16 }}>Available Models</h4>
      {loading ? (
        <div>Loading...</div>
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left", padding: 6 }}>id</th>
              <th style={{ textAlign: "left", padding: 6 }}>created</th>
              <th style={{ textAlign: "left", padding: 6 }}>path</th>
              <th style={{ textAlign: "left", padding: 6 }}>actions</th>
            </tr>
          </thead>
          <tbody>
            {list.map((m) => (
              <Fragment key={m.id}>
                <tr>
                  <td style={{ padding: 6 }}>{m.id}</td>
                  <td style={{ padding: 6 }}>{new Date(m.created_ts).toLocaleString()}</td>
                  <td style={{ padding: 6, fontSize: 12, wordBreak: "break-all" }}>{m.path}</td>
                  <td style={{ padding: 6 }}>
                    <button onClick={() => onEvaluate(m)}>Evaluate</button>{" "}
                    <button onClick={() => onViewReport(m.id)}>
                      {expandedModelId === m.id ? "Hide Report" : "View Report"}
                    </button>
                    <button onClick={() => onPromote(m.id)}>Promote</button>
                  </td>
                </tr>
                {expandedModelId === m.id && renderReportBlock(m.id)}
              </Fragment>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
