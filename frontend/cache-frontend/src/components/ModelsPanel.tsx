// frontend/src/components/ModelsPanel.tsx
import React, { useEffect, useState } from "react";
import { models as fetchModels, trainModel, evaluateModel } from "../lib/api";
import type { ModelEntry, TrainResponse, EvaluateResponse } from "../lib/api";

export default function ModelsPanel(): React.ReactElement {
  const [list, setList] = useState<ModelEntry[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [trace, setTrace] = useState<string>("/simulator/data/run_1.csv");
  const [epochs, setEpochs] = useState<number>(20);
  const [status, setStatus] = useState<string | null>(null);
  const [actionBusy, setActionBusy] = useState<boolean>(false);

  const load = async () => {
    setLoading(true);
    try {
      const data = await fetchModels();
      setList(data ?? []);
    } catch (err) {
      console.error(err);
      setStatus(String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const onTrain = async () => {
    setStatus("starting training...");
    setActionBusy(true);
    try {
      const res: TrainResponse = await trainModel({ trace, epochs });
      const outName = res.out_name ?? "(unknown)";
      setStatus(`training started: out=${outName}`);
      // refresh list after a short delay to allow registry to be written
      setTimeout(load, 1500);
    } catch (e) {
      console.error(e);
      setStatus("train error: " + String(e));
    } finally {
      setActionBusy(false);
    }
  };

  type EvaluatePayload = {
    model_id?: string;
    trace: string;
    plot: boolean;
  };

  const onEvaluate = async (id?: string) => {
    setStatus("evaluating...");
    setActionBusy(true);
    try {
      const payload: EvaluatePayload = { trace, plot: false };
      if (id) payload.model_id = id;
      const res: EvaluateResponse = await evaluateModel(payload);
      setStatus(`eval done; out=${res.out}`);
    } catch (e) {
      console.error(e);
      setStatus("eval error: " + String(e));
    } finally {
      setActionBusy(false);
    }
  };

  return (
    <div style={{ padding: 12 }}>
      <h3>Model Registry & Training</h3>

      <div style={{ marginBottom: 8 }}>
        <label style={{ marginRight: 8 }}>Trace file:</label>
        <input
          value={trace}
          onChange={(e) => setTrace(e.target.value)}
          style={{ width: 360 }}
        />
      </div>

      <div style={{ marginBottom: 8 }}>
        <label style={{ marginRight: 8 }}>Epochs:</label>
        <input
          type="number"
          value={epochs}
          onChange={(e) => setEpochs(Number(e.target.value))}
          style={{ width: 100 }}
          min={1}
        />
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <button onClick={onTrain} disabled={actionBusy}>
          {actionBusy ? "Working..." : "Start Training"}
        </button>
        <button onClick={() => onEvaluate(undefined)} disabled={actionBusy}>
          Evaluate TTL only
        </button>
        <button onClick={load} disabled={loading}>
          Refresh list
        </button>
      </div>

      <div style={{ marginTop: 12 }}>
        <strong>Status:</strong> {status ?? "idle"}
      </div>

      <h4 style={{ marginTop: 16 }}>Available Models</h4>
      {loading ? (
        <div>Loading...</div>
      ) : list.length === 0 ? (
        <div style={{ padding: 12, color: "#777" }}>No models registered yet.</div>
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
              <tr key={m.id}>
                <td style={{ padding: 6, verticalAlign: "top", fontFamily: "monospace" }}>{m.id}</td>
                <td style={{ padding: 6 }}>
                  {m.created_ts ? new Date(m.created_ts).toLocaleString() : "—"}
                </td>
                <td style={{ padding: 6, fontSize: 12, wordBreak: "break-all" }}>{m.path}</td>
                <td style={{ padding: 6 }}>
                  <button onClick={() => onEvaluate(m.id)} disabled={actionBusy}>
                    Evaluate
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
