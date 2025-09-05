export default function Experiments() {
  return (
    <div>
      <h4>Experiments</h4>
      <p>Use the simulator to drive the API, then watch Overview/Cache:</p>
      <pre style={{ background: "#f7f7f7", padding: 12, borderRadius: 8 }}>
        {`# (from simulator/)
python replay.py --file data\\zipf_2m_5rps.csv --base http://127.0.0.1:8000 --rate 10

# then open Overview (live charts) and Cache (capacity bar)`}
      </pre>
      <p>Coming soon: in-UI triggers, saved runs, A/B overlays.</p>
    </div>
  );
}
