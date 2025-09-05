export function StatCard({
  title,
  value,
  suffix,
}: {
  title: string;
  value: string | number;
  suffix?: string;
}) {
  return (
    <div
      style={{
        flex: 1,
        minWidth: 220,
        padding: 16,
        border: "1px solid #eee",
        borderRadius: 12,
        boxShadow: "0 1px 3px #00000010",
        background: "#fff",
      }}
    >
      <div style={{ fontSize: 12, color: "#777", marginBottom: 6 }}>
        {title}
      </div>
      <div style={{ fontSize: 24, fontWeight: 600 }}>
        {value}
        {suffix}
      </div>
    </div>
  );
}
