import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import type { Stats } from "../lib/api";
import { StatCard } from "../components/Cards";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

type HistoryRowLocal = {
  ts: string;
  hit_ratio_pct: number;
  avg_latency_ms: number;
  staleness_pct: number;
};

type Row = {
  t: number;
  hit_ratio_pct: number;
  avg_latency_ms: number;
  staleness_pct: number;
};

export default function Overview() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [series, setSeries] = useState<Row[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  // Function to trigger backend simulation
  const handleSimulate = async (mode: string) => {
    try {
      setLoading(true);
      setMessage(`Starting ${mode.toUpperCase()} simulation...`);
      const res = await api.simulate(mode);
      setMessage(`Simulation (${mode}) started: ${res.status || "OK"}`);
    } catch (err) {
      console.error(err);
      setMessage(`Failed to start ${mode} simulation.`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const pull = async () => {
      try {
        const [s, h] = await Promise.all([
          api.stats(),
          api.history(120) as Promise<HistoryRowLocal[]>,
        ]);
        setStats(s);
        setSeries(
          h.map((r) => ({
            t: new Date(r.ts).getTime(),
            hit_ratio_pct: r.hit_ratio_pct,
            avg_latency_ms: r.avg_latency_ms,
            staleness_pct: r.staleness_pct,
          }))
        );
      } catch {
        // ignore transient errors
      }
    };
    pull();
    const id = window.setInterval(pull, 1000);
    return () => window.clearInterval(id);
  }, []);

  const latest = stats ?? {
    hit_ratio_pct: 0,
    avg_latency_ms: 0,
    staleness_pct: 0,
  };

  const chartData = useMemo(
    () =>
      series.map((r) => ({
        time: new Date(r.t).toLocaleTimeString(),
        ...r,
      })),
    [series]
  );

  return (
    <div>
      {/* Simulation Controls */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <button disabled={loading} onClick={() => handleSimulate("ttl")}>
          Run TTL Mode
        </button>
        <button disabled={loading} onClick={() => handleSimulate("drl")}>
          Run DRL Mode
        </button>
        <button disabled={loading} onClick={() => handleSimulate("hybrid")}>
          Run Hybrid Mode
        </button>
      </div>
      {message && <div style={{ marginBottom: 16, color: "#555" }}>{message}</div>}

      <div
        style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 16 }}
      >
        <StatCard
          title="Hit Ratio"
          value={latest.hit_ratio_pct.toFixed(2)}
          suffix="%"
        />
        <StatCard
          title="Avg Latency"
          value={latest.avg_latency_ms.toFixed(2)}
          suffix=" ms"
        />
        <StatCard
          title="Staleness"
          value={latest.staleness_pct.toFixed(2)}
          suffix="%"
        />
      </div>

      <div
        style={{
          padding: 12,
          border: "1px solid #eee",
          borderRadius: 12,
          background: "#fff",
        }}
      >
        <h4 style={{ margin: "6px 0 12px" }}>Live Metrics</h4>
        {chartData.length === 0 ? (
          <div style={{ padding: 12, color: "#777" }}>
            No data yet — start a simulation to see metrics.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip
                formatter={(v: number, name: string) => {
                  if (name.includes("Latency"))
                    return [`${v.toFixed(2)} ms`, name];
                  return [`${v.toFixed(2)} %`, name];
                }}
              />
              <Legend />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="hit_ratio_pct"
                name="Hit Ratio (%)"
                dot={false}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="avg_latency_ms"
                name="Avg Latency (ms)"
                dot={false}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="staleness_pct"
                name="Staleness (%)"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
