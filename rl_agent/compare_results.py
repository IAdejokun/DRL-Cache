# rl_agent/compare_results.py
"""
Run TTL vs DRL evaluation and plot comparison.
This is a small helper script that calls functions in evaluate.py
"""

import argparse
from rl_agent.evaluate import evaluate_ttl_from_trace, evaluate_drl_from_trace, save_report_csv, plot_comparison

def main(args):
    ttl_metrics = evaluate_ttl_from_trace(args.trace, max_bytes=args.max_bytes, ttl_s=args.ttl_s, hit_ms=args.hit_ms)
    drl_metrics = None
    if args.model and args.model_exists:
        drl_metrics = evaluate_drl_from_trace(args.trace, args.model, max_bytes=args.max_bytes, ttl_s=args.ttl_s, hit_ms=args.hit_ms)

    # Save CSV
    out_csv = args.out or "logs/eval_report.csv"
    save_report_csv(out_csv, ttl_metrics, drl_metrics)
    print("Saved evaluation CSV to", out_csv)

    # Plot (show & optionally save)
    plot_path = args.plot_save if args.plot_save else None
    plot_comparison(ttl_metrics, drl_metrics, save_path=plot_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--model", default="rl_agent/model.pt")
    p.add_argument("--out", default="logs/eval_report.csv")
    p.add_argument("--plot-save", default=None)
    p.add_argument("--max-bytes", type=int, default=50_000_000)
    p.add_argument("--ttl-s", type=int, default=300)
    p.add_argument("--hit-ms", type=int, default=20)
    args = p.parse_args()

    # quick model existence flag to avoid error
    args.model_exists = args.model and __import__("os").path.exists(args.model)

    main(args)
