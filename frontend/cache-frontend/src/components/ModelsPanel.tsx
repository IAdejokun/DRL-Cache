import { Fragment, useEffect, useState } from "react";
import {
  models as fetchModels,
  trainModel,
  evaluateModel,
  getModelReport,
  resetModelRegistry,
  activeModel,
  promoteModel,
  type ModelEntry,
  type EvalRow,
  type ActiveModel,
} from "../lib/api";

/* type EvaluatePayload = Parameters<typeof evaluateModel>[0]; */

const tableShellStyle: React.CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  background: "#111",
  color: "#fff",
  border: "1px solid #333",
  borderRadius: 8,
  overflow: "hidden",
};
const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "10px 12px",
  background: "#222",
  color: "#fff",
  borderBottom: "1px solid #333",
  fontWeight: 600,
};
const tdStyle: React.CSSProperties = {
  padding: "10px 12px",
  color: "#fff",
  borderBottom: "1px solid #333",
  verticalAlign: "top",
  wordBreak: "break-word",
};
const actionsTd: React.CSSProperties = { ...tdStyle, whiteSpace: "nowrap" };
const badge: React.CSSProperties = {
  display: "inline-block",
  marginLeft: 8,
  padding: "2px 6px",
  background: "#2e7d32",
  color: "#fff",
  borderRadius: 6,
  fontSize: 12,
  fontWeight: 600,
};

export default function ModelsPanel() {
  const [list, setList] = useState<ModelEntry[]>([]);
  const [loading, setLoading] = useState(false);

  // Training / evaluation params
  const [trace, setTrace] = useState("/simulator/data/run_1.csv");
  const [epochs, setEpochs] = useState(20);

  // NEW: evaluator knobs (used by upgraded backend/evaluator)
  const [maxBytes, setMaxBytes] = useState<number>(1_000_000_000);
  const [ttlSeconds, setTtlSeconds] = useState<number>(300);
  const [hitMs, setHitMs] = useState<number>(20);

  const [status, setStatus] = useState<string | null>(null);
  const [expandedModelId, setExpandedModelId] = useState<string | null>(null);
  const [reportData, setReportData] = useState<Record<string, EvalRow[]>>({});
  const [active, setActive] = useState<ActiveModel | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const [models, act] = await Promise.all([fetchModels(), activeModel().catch(() => null)]);
      setList(models);
      setActive(act ?? null);
    } catch (e) {
      console.error(e);
      setStatus(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const onTrain = async () => {
    setStatus("starting training...");
    try {
      const res = await trainModel({ trace, epochs });
      setStatus(`training started: out=${res.out_name ?? "(no name)"}`);
      setTimeout(load, 1500);
    } catch (e) {
      setStatus("train error: " + String(e));
    }
  };

  const onEvaluate = async (id?: string) => {
    setStatus("evaluating...");
    try {
      // NOTE: Backend must accept these extra fields to influence evaluator.
      // If your /api/models/evaluate endpoint doesn’t read them yet, it will simply ignore them.
      const payload: {
        trace?: string;
        plot?: boolean;
        model_id?: string;
        max_bytes?: number;
        ttl_s?: number;
        hit_ms?: number;
      } = {
        trace,
        plot: false,
        max_bytes: maxBytes,
        ttl_s: ttlSeconds,
        hit_ms: hitMs,
      };
      if (id) payload.model_id = id;

      await evaluateModel(payload);
      setStatus("eval done");

      if (id) {
        const rep = await getModelReport(id);
        if (rep && Array.isArray(rep.rows)) {
          setReportData((prev) => ({ ...prev, [id]: rep.rows as EvalRow[] }));
          setExpandedModelId(id);
        }
      }
      setTimeout(load, 1000);
    } catch (e) {
      setStatus("eval error: " + String(e));
    }
  };

  const onPromote = async (id: string, path: string) => {
    setStatus("promoting model...");
    try {
      await promoteModel({ model_id: id });
      setStatus("model promoted");
      setTimeout(load, 500);
    } catch (e) {
      try {
        // fallback by explicit path (backend expects model_path key)
        await promoteModel({ model_path: path });
        setStatus("model promoted (by path)");
        setTimeout(load, 500);
      } catch (e2) {
        setStatus("promote error: " + String(e) + "; " + String(e2));
      }
    }
  };

  const onViewReport = async (modelId: string) => {
    if (expandedModelId === modelId) {
      setExpandedModelId(null);
      return;
    }
    if (reportData[modelId]) {
      setExpandedModelId(modelId);
      return;
    }
    setStatus("fetching report...");
    try {
      const res = await getModelReport(modelId);
      if (res && Array.isArray(res.rows)) {
        setReportData((prev) => ({ ...prev, [modelId]: res.rows as EvalRow[] }));
      }
      setExpandedModelId(modelId);
      setStatus(null);
    } catch (e) {
      setStatus("report error: " + String(e));
    }
  };

  const isActive = (m: ModelEntry) => {
    if (!active?.path) return false;
    const left = (active.path || "").replace(/\\/g, "/").toLowerCase();
    const right = (m.path || "").replace(/\\/g, "/").toLowerCase();
    return left === right || left.endsWith("/model.pt");
  };

  const header = (
    <>
      <h3 style={{ color: "#fff" }}>Model Registry & Training</h3>

      {/* Training inputs */}
      <div style={{ marginBottom: 8, color: "#fff" }}>
        <label>Trace file: </label>
        <input value={trace} onChange={(e) => setTrace(e.target.value)} style={{ width: 360 }} />
      </div>
      <div style={{ marginBottom: 8, color: "#fff" }}>
        <label>Epochs: </label>
        <input
          type="number"
          value={epochs}
          onChange={(e) => setEpochs(Number(e.target.value))}
          style={{ width: 100 }}
        />
      </div>

      {/* NEW: Evaluator knobs */}
      <div style={{ marginBottom: 8, color: "#fff", display: "flex", gap: 16, flexWrap: "wrap" }}>
        <div>
          <label>Max Bytes: </label>
          <input
            type="number"
            value={maxBytes}
            onChange={(e) => setMaxBytes(Number(e.target.value))}
            style={{ width: 160 }}
            min={0}
          />
        </div>
        <div>
          <label>TTL (s): </label>
          <input
            type="number"
            value={ttlSeconds}
            onChange={(e) => setTtlSeconds(Number(e.target.value))}
            style={{ width: 120 }}
            min={0}
          />
        </div>
        <div>
          <label>Hit Latency (ms): </label>
          <input
            type="number"
            value={hitMs}
            onChange={(e) => setHitMs(Number(e.target.value))}
            style={{ width: 140 }}
            min={0}
          />
        </div>
      </div>

      <button onClick={onTrain}>Start Training</button>{" "}
      <button onClick={() => onEvaluate(undefined)}>Evaluate TTL only</button>{" "}
      <button
        onClick={async () => {
          try {
            await resetModelRegistry();
            setStatus("Registry cleared");
            await load();
          } catch (e) {
            setStatus("Reset error: " + String(e));
          }
        }}
      >
        Clear Registry
      </button>

      <div style={{ marginTop: 12, color: "#fff" }}>
        <strong>Status:</strong> {status ?? "idle"}
      </div>
      <h4 style={{ marginTop: 16, color: "#fff" }}>Available Models</h4>
    </>
  );

  return (
    <div style={{ padding: 12 }}>
      {header}

      {loading ? (
        <div style={{ color: "#fff" }}>Loading...</div>
      ) : (
        <table style={tableShellStyle}>
          <thead>
            <tr>
              <th style={thStyle}>id</th>
              <th style={thStyle}>created</th>
              <th style={thStyle}>path</th>
              <th style={thStyle}>actions</th>
            </tr>
          </thead>

          <tbody>
            {list.map((m) => (
              <Fragment key={m.id}>
                <tr>
                  <td style={tdStyle}>
                    {m.id}
                    {isActive(m) && <span style={badge}>ACTIVE</span>}
                  </td>
                  <td style={tdStyle}>{new Date(m.created_ts).toLocaleString()}</td>
                  <td style={{ ...tdStyle, fontSize: 12 }}>{m.path}</td>
                  <td style={actionsTd}>
                    <button onClick={() => onEvaluate(m.id)}>Evaluate</button>{" "}
                    <button onClick={() => onPromote(m.id, m.path)}>Promote</button>{" "}
                    <button onClick={() => onViewReport(m.id)}>
                      {expandedModelId === m.id ? "Hide Report" : "View Report"}
                    </button>
                  </td>
                </tr>

                {expandedModelId === m.id && (
                  <tr>
                    <td colSpan={4} style={{ ...tdStyle, background: "#111" }}>
                      <div
                        style={{
                          background: "#0d0d0d",
                          padding: 10,
                          borderRadius: 8,
                          border: "1px solid #333",
                        }}
                      >
                        {/* Show the evaluator parameters used for clarity */}
                        <div style={{ color: "#bbb", marginBottom: 8, fontSize: 13 }}>
                          <strong>Eval params:</strong> max_bytes={maxBytes.toLocaleString()} · ttl_s={ttlSeconds} · hit_ms={hitMs}
                        </div>

                        {reportData[m.id] ? (
                          <table
                            style={{
                              width: "100%",
                              borderCollapse: "collapse",
                              background: "#1a1a1a",
                              color: "#fff",
                              border: "1px solid #333",
                              borderRadius: 6,
                              overflow: "hidden",
                            }}
                          >
                            <thead>
                              <tr>
                                <th style={{ ...thStyle, background: "#262626" }}>Policy</th>
                                <th style={{ ...thStyle, background: "#262626" }}>Total</th>
                                <th style={{ ...thStyle, background: "#262626" }}>Hits</th>
                                <th style={{ ...thStyle, background: "#262626" }}>Hit Ratio %</th>
                                <th style={{ ...thStyle, background: "#262626" }}>Latency</th>
                                <th style={{ ...thStyle, background: "#262626" }}>Stale %</th>
                                {/* NEW: always show the three bandwidth columns if present in any row */}
                                {reportData[m.id].some(r => typeof r.bytes_from_cache !== "undefined") && (
                                  <>
                                    <th style={{ ...thStyle, background: "#262626" }}>Bytes Cached</th>
                                    <th style={{ ...thStyle, background: "#262626" }}>Bytes Origin</th>
                                    <th style={{ ...thStyle, background: "#262626" }}>BW Savings %</th>
                                  </>
                                )}
                              </tr>
                            </thead>
                            <tbody>
                              {reportData[m.id].map((row, i) => (
                                <tr key={i}>
                                  <td style={tdStyle}>{row.policy}</td>
                                  <td style={tdStyle}>{row.total}</td>
                                  <td style={tdStyle}>{row.hits}</td>
                                  <td style={tdStyle}>{row.hit_ratio_pct}</td>
                                  <td style={tdStyle}>{row.avg_latency_ms}</td>
                                  <td style={tdStyle}>{row.stale_pct}</td>
                                  {reportData[m.id].some(r => typeof r.bytes_from_cache !== "undefined") && (
                                    <>
                                      <td style={tdStyle}>
                                        {typeof row.bytes_from_cache === "number" ? row.bytes_from_cache.toLocaleString() : ""}
                                      </td>
                                      <td style={tdStyle}>
                                        {typeof row.bytes_from_origin === "number" ? row.bytes_from_origin.toLocaleString() : ""}
                                      </td>
                                      <td style={tdStyle}>
                                        {typeof row.bandwidth_savings_pct === "number" ? row.bandwidth_savings_pct : ""}
                                      </td>
                                    </>
                                  )}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        ) : (
                          <div style={{ color: "#bbb" }}>
                            No report found yet. Click <em>Evaluate</em> first, or try again in a moment.
                          </div>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
