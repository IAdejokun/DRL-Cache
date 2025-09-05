import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { CacheItem, CacheStats } from "../lib/api";

export default function Cache() {
  const [items, setItems] = useState<CacheItem[]>([]);
  const [cs, setCs] = useState<CacheStats | null>(null);

  const pull = async () => {
    try {
      const [i, s] = await Promise.all([api.cache(), api.cacheStats()]);
      setItems(i);
      setCs(s);
    } catch (error) {
      console.error('Failed to fetch cache data:', error);
    }
  };

  useEffect(() => {
    pull();
    const id = setInterval(pull, 2000);
    return () => clearInterval(id);
  }, []);

  return (
    <div>
      <h4>Cache Capacity</h4>
      <div style={{ marginBottom: 8 }}>
        {cs ? (
          <div style={{ maxWidth: 600 }}>
            <div style={{ fontSize: 13, color: "#555", marginBottom: 4 }}>
              {cs.items} items • {cs.bytes_used.toLocaleString()} /{" "}
              {cs.max_bytes.toLocaleString()} bytes ({cs.pct_full}%)
            </div>
            <div
              style={{
                height: 12,
                width: "100%",
                background: "#eee",
                borderRadius: 8,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${Math.min(cs.pct_full, 100)}%`,
                  background: "#9cf",
                }}
              />
            </div>
          </div>
        ) : (
          "Loading..."
        )}
      </div>

      <h4 style={{ marginTop: 16 }}>Cache Items</h4>
      <div
        style={{ overflow: "auto", border: "1px solid #eee", borderRadius: 12 }}
      >
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead style={{ background: "#fafafa" }}>
            <tr>
              <th
                style={{
                  textAlign: "left",
                  padding: 8,
                  borderBottom: "1px solid #eee",
                }}
              >
                object_id
              </th>
              <th
                style={{
                  textAlign: "right",
                  padding: 8,
                  borderBottom: "1px solid #eee",
                }}
              >
                size_bytes
              </th>
              <th
                style={{
                  textAlign: "right",
                  padding: 8,
                  borderBottom: "1px solid #eee",
                }}
              >
                ttl_s
              </th>
              <th
                style={{
                  textAlign: "left",
                  padding: 8,
                  borderBottom: "1px solid #eee",
                }}
              >
                last_updated_ts
              </th>
            </tr>
          </thead>
          <tbody>
            {items.map((r) => (
              <tr key={r.object_id}>
                <td style={{ padding: 8, borderBottom: "1px solid #f2f2f2" }}>
                  {r.object_id}
                </td>
                <td
                  style={{
                    padding: 8,
                    textAlign: "right",
                    borderBottom: "1px solid #f2f2f2",
                  }}
                >
                  {r.size_bytes.toLocaleString()}
                </td>
                <td
                  style={{
                    padding: 8,
                    textAlign: "right",
                    borderBottom: "1px solid #f2f2f2",
                  }}
                >
                  {r.ttl_s}
                </td>
                <td style={{ padding: 8, borderBottom: "1px solid #f2f2f2" }}>
                  {new Date(r.last_updated_ts).toLocaleString()}
                </td>
              </tr>
            ))}
            {items.length === 0 && (
              <tr>
                <td colSpan={4} style={{ padding: 12, color: "#777" }}>
                  Empty cache
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
