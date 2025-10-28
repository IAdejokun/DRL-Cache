// frontend/src/components/Experiments.tsx
import { useEffect, useMemo, useState } from "react";
import {
  startRun,
  runs as fetchRuns,
  historyByRun,
  type RunRow,
  type HistoryRow,
  type StartRunPayload,
} from "../lib/api";

const label: React.CSSProperties = { marginRight: 8 };
const input: React.CSSProperties = { width: 90, marginRight: 14 };
const select: React.CSSProperties = { marginRight: 14 };

export default function Experiments() {
  // form state
  const [workload, setWorkload] = useState<"zipf" | "uniform">("zipf");
  const [minutes, setMinutes] = useState<number>(10);
  const [rps, setRps] = useState<number>(60);
  const [rate, setRate] = useState<number>(60);

  // NEW knobs
  const [objects, setObjects] = useState<number>(5000);
  const [seed, setSeed] = useState<number | "">("");

  // runs / selection / history
  const [list, setList] = useState<RunRow[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [runHistory, setRunHistory] = useState<HistoryRow[]>([]);
  const [status, setStatus] = useState<string>("idle");

  const selectedRun = useMemo(
    () => list.find((r) => r.id === selectedRunId) || null,
    [list, selectedRunId]
  );

  // load recent runs
  const loadRuns = async () => {
    try {
      const rs = await fetchRuns();
      setList(rs);
    } catch (e) {
      setStatus("runs error: " + String(e));
    }
  };

  useEffect(() => {
    loadRuns();
    const t = setInterval(loadRuns, 4000);
    return () => clearInterval(t);
  }, []);

  // per-run history loader
  const loadHistory = async (rid: number) => {
    try {
      const h = await historyByRun(rid, 300);
      setRunHistory(h);
    } catch (e) {
      console.error("Failed to load history:", e);
      setRunHistory([]);
    }
  };

  // poll history while run is running
  useEffect(() => {
    if (!selectedRunId) return;
    loadHistory(selectedRunId);

    // poll if still running
    const t = setInterval(() => {
      const r = list.find((x) => x.id === selectedRunId);
      if (!r) return;
      if (r.status === "running") {
        loadHistory(selectedRunId);
      } else {
        // fetch once after stop as well, then end polling
        loadHistory(selectedRunId);
        clearInterval(t);
      }
    }, 2000);

    return () => clearInterval(t);
  }, [selectedRunId, list]); // re-evaluate when status/list changes

  // start a new run
  const onRun = async () => {
    setStatus("starting...");
    try {
      const payload: StartRunPayload = {
        workload,
        minutes,
        rps,
        rate,
        objects,            // NEW
        seed: seed === "" ? undefined : Number(seed), // NEW (optional)
      };
      const out = await startRun(payload);
      setStatus(`run ${out.run_id} started`);
      await loadRuns();
      setSelectedRunId(out.run_id);
    } catch (e) {
      setStatus("start error: " + String(e));
    }
  };

  return (
    <div style={{ padding: 12 }}>
      <h3>Experiment Runner</h3>

      {/* Controls */}
      <div style={{ marginBottom: 10 }}>
        <span style={label}>Workload</span>
        <select
          style={select}
          value={workload}
          onChange={(e) => setWorkload(e.target.value as "zipf" | "uniform")}
        >
          <option value="zipf">zipf</option>
           <option value="flash">flash (hot burst)</option>
           <option value="writeheavy">writeheavy</option>
        </select>

        <span style={label}>Minutes</span>
        <input
          type="number"
          style={input}
          value={minutes}
          min={1}
          onChange={(e) => setMinutes(Number(e.target.value))}
        />

        <span style={label}>RPS</span>
        <input
          type="number"
          style={input}
          value={rps}
          min={1}
          onChange={(e) => setRps(Number(e.target.value))}
        />

        <span style={label}>Rate</span>
        <input
          type="number"
          style={input}
          value={rate}
          min={1}
          onChange={(e) => setRate(Number(e.target.value))}
        />

        {/* NEW: Objects & Seed */}
        <span style={label}>Objects</span>
        <input
          type="number"
          style={input}
          value={objects}
          min={100}
          onChange={(e) => setObjects(Number(e.target.value))}
          title="Number of unique objects in the trace"
        />

        <span style={label}>Seed</span>
        <input
          type="number"
          style={{ width: 110, marginRight: 14 }}
          value={seed}
          onChange={(e) => setSeed(e.target.value === "" ? "" : Number(e.target.value))}
          placeholder="optional"
          title="Use any integer for reproducible traces"
        />

        <button onClick={onRun}>Run</button>
      </div>

      <div style={{ marginTop: 8 }}>
        <strong>Status:</strong> {status}
      </div>

      {/* Runs table */}
      <h4 style={{ marginTop: 18 }}>Recent Runs</h4>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", padding: 6 }}>ID</th>
            <th style={{ textAlign: "left", padding: 6 }}>Started</th>
            <th style={{ textAlign: "left", padding: 6 }}>Workload</th>
            <th style={{ textAlign: "left", padding: 6 }}>mins</th>
            <th style={{ textAlign: "left", padding: 6 }}>rps</th>
            <th style={{ textAlign: "left", padding: 6 }}>rate</th>
            <th style={{ textAlign: "left", padding: 6 }}>status</th>
          </tr>
        </thead>
        <tbody>
          {list.map((r) => {
            const isSel = r.id === selectedRunId;
            return (
              <tr
                key={r.id}
                onClick={() => setSelectedRunId(r.id)}
                style={{
                  cursor: "pointer",
                  background: isSel ? "#1f2937" : "transparent",
                }}
                title="Click to view per-run history"
              >
                <td style={{ padding: 6 }}>{r.id}</td>
                <td style={{ padding: 6 }}>{new Date(r.started_ts).toLocaleString()}</td>
                <td style={{ padding: 6 }}>{r.workload}</td>
                <td style={{ padding: 6 }}>{r.minutes}</td>
                <td style={{ padding: 6 }}>{r.rps}</td>
                <td style={{ padding: 6 }}>{r.rate}</td>
                <td style={{ padding: 6 }}>{r.status}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {/* Per-run live history */}
      <div style={{ marginTop: 18 }}>
        <h4>
          Run History {selectedRunId ? `#${selectedRunId}` : ""}
          {selectedRun ? ` — ${selectedRun.status}` : ""}
        </h4>
        {selectedRunId ? (
          runHistory.length ? (
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", padding: 6 }}>Time</th>
                  <th style={{ textAlign: "left", padding: 6 }}>Hit Ratio %</th>
                  <th style={{ textAlign: "left", padding: 6 }}>Avg Latency (ms)</th>
                  <th style={{ textAlign: "left", padding: 6 }}>Staleness %</th>
                </tr>
              </thead>
              <tbody>
                {runHistory.map((h, i) => (
                  <tr key={i}>
                    <td style={{ padding: 6 }}>{new Date(h.ts).toLocaleTimeString()}</td>
                    <td style={{ padding: 6 }}>{h.hit_ratio_pct.toFixed(2)}</td>
                    <td style={{ padding: 6 }}>{h.avg_latency_ms.toFixed(2)}</td>
                    <td style={{ padding: 6 }}>{h.staleness_pct.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div>No samples yet…</div>
          )
        ) : (
          <div>Select a run above to view its live history.</div>
        )}
      </div>
    </div>
  );
}
