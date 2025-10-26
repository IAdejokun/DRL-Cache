import { Link, Outlet, useLocation } from "react-router-dom";

export default function App() {
  const loc = useLocation();
  const link = (to: string, label: string) => (
    <Link
      to={to}
      style={{
        padding: "8px 12px",
        borderRadius: 8,
        textDecoration: "none",
        background:
          loc.pathname === to || (to === "/" && loc.pathname === "/")
            ? "#eef"
            : "#f7f7f7",
        border: "1px solid #ddd",
        marginRight: 8,
        color: "#222",
      }}
    >
      {label}
    </Link>
  );

  return (
    <div style={{ maxWidth: 1100, margin: "24px auto", padding: 16 }}>
      <h2 style={{ marginBottom: 12 }}>DRL Cache Dashboard</h2>
      <div style={{ marginBottom: 16 }}>
        {link("/", "Overview")}
        {link("/cache", "Cache")}
        {link("/experiments", "Experiments")}
        {link("/models", "Models")} 
      </div>
      <Outlet />
    </div>
  );
}
