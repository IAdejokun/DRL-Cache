import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { RunRow } from "../lib/api";
import {startRun} from "../lib/api";

export default function Experiments() {
  const [workload, setWorkload] = useState("zipf");
  const [minutes, setMinutes] = useState(2);
  const [rps, setRps] = useState(5);
  const [rate, setRate] = useState(10);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const refresh = async () => setRuns(await api.runs());
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 1500);
    return () => clearInterval(t);
  }, []);

  const start = async () => {
    await startRun({ workload, minutes, rps, rate });
    await refresh();
  };

  return (
    <div>
      <h4>Start a Replay</h4>
      <div
        style={{
          display: "flex",
          gap: 8,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <label>
          Workload
          <select
            value={workload}
            onChange={(e) => setWorkload(e.target.value)}
            style={{ marginLeft: 6 }}
          >
            <option value="zipf">zipf</option>
            <option value="flash">flash</option>
            <option value="writeheavy">writeheavy</option>
          </select>
        </label>
        <label>
          Minutes{" "}
          <input
            type="number"
            value={minutes}
            onChange={(e) => setMinutes(+e.target.value)}
            style={{ width: 70, marginLeft: 6 }}
          />
        </label>
        <label>
          RPS{" "}
          <input
            type="number"
            value={rps}
            onChange={(e) => setRps(+e.target.value)}
            style={{ width: 70, marginLeft: 6 }}
          />
        </label>
        <label>
          Rate{" "}
          <input
            type="number"
            value={rate}
            onChange={(e) => setRate(+e.target.value)}
            style={{ width: 70, marginLeft: 6 }}
          />
        </label>
        <button onClick={start} style={{ padding: "6px 12px" }}>
          Run
        </button>
      </div>

      <h4 style={{ marginTop: 16 }}>Recent Runs</h4>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th
              style={{
                textAlign: "left",
                padding: 8,
                borderBottom: "1px solid #eee",
              }}
            >
              ID
            </th>
            <th
              style={{
                textAlign: "left",
                padding: 8,
                borderBottom: "1px solid #eee",
              }}
            >
              Started
            </th>
            <th
              style={{
                textAlign: "left",
                padding: 8,
                borderBottom: "1px solid #eee",
              }}
            >
              Workload
            </th>
            <th
              style={{
                textAlign: "right",
                padding: 8,
                borderBottom: "1px solid #eee",
              }}
            >
              mins
            </th>
            <th
              style={{
                textAlign: "right",
                padding: 8,
                borderBottom: "1px solid #eee",
              }}
            >
              rps
            </th>
            <th
              style={{
                textAlign: "right",
                padding: 8,
                borderBottom: "1px solid #eee",
              }}
            >
              rate
            </th>
            <th
              style={{
                textAlign: "left",
                padding: 8,
                borderBottom: "1px solid #eee",
              }}
            >
              status
            </th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id}>
              <td style={{ padding: 8, borderBottom: "1px solid #f2f2f2" }}>
                {r.id}
              </td>
              <td style={{ padding: 8, borderBottom: "1px solid #f2f2f2" }}>
                {new Date(r.started_ts).toLocaleString()}
              </td>
              <td style={{ padding: 8, borderBottom: "1px solid #f2f2f2" }}>
                {r.workload}
              </td>
              <td
                style={{
                  padding: 8,
                  textAlign: "right",
                  borderBottom: "1px solid #f2f2f2",
                }}
              >
                {r.minutes}
              </td>
              <td
                style={{
                  padding: 8,
                  textAlign: "right",
                  borderBottom: "1px solid #f2f2f2",
                }}
              >
                {r.rps}
              </td>
              <td
                style={{
                  padding: 8,
                  textAlign: "right",
                  borderBottom: "1px solid #f2f2f2",
                }}
              >
                {r.rate}
              </td>
              <td style={{ padding: 8, borderBottom: "1px solid #f2f2f2" }}>
                {r.status}
              </td>
            </tr>
          ))}
          {runs.length === 0 && (
            <tr>
              <td colSpan={7} style={{ padding: 12, color: "#777" }}>
                No runs yet
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
