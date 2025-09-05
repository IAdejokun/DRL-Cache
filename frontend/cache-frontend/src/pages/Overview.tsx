import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import type { Stats } from "../lib/api";
import { StatCard } from "../components/Cards";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

type Row = {
  t: number;
  hit_ratio_pct: number;
  avg_latency_ms: number;
  staleness_pct: number;
};

export default function Overview() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [series, setSeries] = useState<Row[]>([]);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    const pull = async () => {
      try {
        const s = await api.stats();
        setStats(s);
        setSeries((prev) => [...prev.slice(-120), { t: Date.now(), ...s }]); // keep last ~2min @1Hz
      } catch {
        // ignore one-off errors
      }
    };
    pull();
    timer.current = window.setInterval(pull, 1000); // 1Hz
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, []);

  const latest = stats ?? {
    hit_ratio_pct: 0,
    avg_latency_ms: 0,
    staleness_pct: 0,
  };

  const chartData = useMemo(
    () =>
      series.map((r) => ({ time: new Date(r.t).toLocaleTimeString(), ...r })),
    [series]
  );

  return (
    <div>
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
        <LineChart width={980} height={320} data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip />
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
      </div>
    </div>
  );
}
