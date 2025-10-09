# rl_agent/evaluate.py
"""
Evaluate DRL agent vs TTL baseline on a trace and optionally plot results.

Provides:
- run_policy_on_trace(...)  -> lowest-level runner for a single trace run
- evaluate_ttl_from_trace(...) -> returns metrics dict for TTL
- evaluate_drl_from_trace(...) -> returns metrics dict for DRL (loads model)
- save_report_csv(...) -> write CSV with ttl + drl rows
- plot_comparison(...) -> simple matplotlib bar chart (TTL vs DRL)
- CLI entry for quick runs: python rl_agent/evaluate.py --trace ...
"""

import argparse
import os
import csv
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

from rl_agent.trace_env import TraceEnv
from rl_agent.dqn import DQNAgent


def run_policy_on_trace(policy: str, trace_path: str, max_bytes: int, ttl_s: int, hit_ms: int, agent: Optional[DQNAgent] = None) -> Dict[str, Any]:
    """
    Run a single pass of `policy` over the CSV trace.
    policy: "ttl" or "drl"
    Returns metrics dict with keys: total, hits, hit_ratio_pct, avg_latency_ms, stale_pct
    """
    env = TraceEnv(trace_path, max_bytes=max_bytes, ttl_s=ttl_s, hit_latency_ms=hit_ms)
    # env.reset() may return (obs, meta) or obs — handle both
    reset_ret = env.reset()
    obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret

    done = False
    total = 0
    hits = 0
    stale_count = 0
    total_latency = 0.0

    while not done:
        if policy == "ttl":
            # TTL baseline: refresh on writes (peek row)
            row = env.df.iloc[env.index]
            was_write = bool(row["was_write"])
            action = 1 if was_write else 0
        elif policy == "drl":
            if agent is None:
                raise ValueError("DRL policy requested but no agent provided.")
            action = agent.act(obs, epsilon=0.0)
        else:
            action = 0

        next_obs, reward, done, info = env.step(action)

        total += 1
        total_latency += info.get("served_latency_ms", 0.0)
        # served_from_cache + not stale counts as a good hit
        if info.get("served_from_cache", False) and not info.get("stale", False):
            hits += 1
        if info.get("served_from_cache", False) and info.get("stale", False):
            stale_count += 1

        if next_obs is None:
            break
        obs = next_obs

    hit_ratio = 100.0 * hits / total if total else 0.0
    avg_latency = total_latency / total if total else 0.0
    stale_pct = 100.0 * stale_count / total if total else 0.0

    return {
        "total": total,
        "hits": hits,
        "hit_ratio_pct": hit_ratio,
        "avg_latency_ms": avg_latency,
        "stale_pct": stale_pct,
    }


def evaluate_ttl_from_trace(trace_path: str, max_bytes: int = 50_000_000, ttl_s: int = 300, hit_ms: int = 20) -> Dict[str, Any]:
    """Run TTL baseline on given trace and return metrics."""
    return run_policy_on_trace("ttl", trace_path, max_bytes, ttl_s, hit_ms, agent=None)


def evaluate_drl_from_trace(trace_path: str, model_path: str, max_bytes: int = 50_000_000, ttl_s: int = 300, hit_ms: int = 20) -> Optional[Dict[str, Any]]:
    """Load model_path and run DRL evaluation. Returns None if model missing."""
    if not model_path or not os.path.exists(model_path):
        print(f"No DRL model found at {model_path} — skipping DRL evaluation.")
        return None
    agent = DQNAgent.load(model_path, state_dim=5)
    print(f"Loaded DRL agent from {model_path}")
    return run_policy_on_trace("drl", trace_path, max_bytes, ttl_s, hit_ms, agent=agent)


def save_report_csv(out_csv: str, ttl_metrics: Dict[str, Any], drl_metrics: Optional[Dict[str, Any]] = None):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "total", "hits", "hit_ratio_pct", "avg_latency_ms", "stale_pct"])
        writer.writerow(["ttl",
                         ttl_metrics["total"],
                         ttl_metrics["hits"],
                         round(ttl_metrics["hit_ratio_pct"], 4),
                         round(ttl_metrics["avg_latency_ms"], 4),
                         round(ttl_metrics["stale_pct"], 4)])
        if drl_metrics:
            writer.writerow(["drl",
                             drl_metrics["total"],
                             drl_metrics["hits"],
                             round(drl_metrics["hit_ratio_pct"], 4),
                             round(drl_metrics["avg_latency_ms"], 4),
                             round(drl_metrics["stale_pct"], 4)])


def plot_comparison(ttl_metrics: Dict[str, Any], drl_metrics: Optional[Dict[str, Any]] = None, title: str = "DRL vs TTL Caching Performance", save_path: Optional[str] = None):
    """
    Plot a side-by-side bar chart comparing TTL and DRL for:
      - Hit Ratio (%)
      - Avg Latency (ms)
      - Staleness (%)
    """
    metrics = ["Hit Ratio (%)", "Avg Latency (ms)", "Staleness (%)"]
    ttl_values = [
        ttl_metrics["hit_ratio_pct"],
        ttl_metrics["avg_latency_ms"],
        ttl_metrics["stale_pct"],
    ]
    drl_values = None
    if drl_metrics:
        drl_values = [
            drl_metrics["hit_ratio_pct"],
            drl_metrics["avg_latency_ms"],
            drl_metrics["stale_pct"],
        ]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], ttl_values, width, label="TTL Baseline")
    if drl_values:
        ax.bar([i + width / 2 for i in x], drl_values, width, label="DRL Agent")

    # labels above bars
    def _label_bars(xs, vals):
        for xi, v in zip(xs, vals):
            ax.text(xi, v + (0.02 * max(vals + [1])), f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    _label_bars([i - width / 2 for i in x], ttl_values)
    if drl_values:
        _label_bars([i + width / 2 for i in x], drl_values)

    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()


def evaluate_cli(args):
    ttl_metrics = evaluate_ttl_from_trace(args.trace, max_bytes=args.max_bytes, ttl_s=args.ttl_s, hit_ms=args.hit_ms)
    drl_metrics = None
    if args.model and os.path.exists(args.model):
        drl_metrics = evaluate_drl_from_trace(args.trace, args.model, max_bytes=args.max_bytes, ttl_s=args.ttl_s, hit_ms=args.hit_ms)

    save_report_csv(args.out, ttl_metrics, drl_metrics)
    print("Evaluation report written to", args.out)
    print("TTL:", ttl_metrics)
    if drl_metrics:
        print("DRL:", drl_metrics)

    if args.plot:
        plot_save = args.plot_save if args.plot_save else None
        plot_comparison(ttl_metrics, drl_metrics, save_path=plot_save)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, help="Path to simulator CSV trace")
    p.add_argument("--model", default="rl_agent/model.pt", help="Path to DRL model")
    p.add_argument("--out", default="logs/eval_report.csv", help="CSV output path")
    p.add_argument("--max-bytes", type=int, default=50_000_000)
    p.add_argument("--ttl-s", type=int, default=300)
    p.add_argument("--hit-ms", type=int, default=20)
    p.add_argument("--plot", action="store_true", help="Show comparison plot after evaluation")
    p.add_argument("--plot-save", default=None, help="If set, save comparison plot to this path")
    args = p.parse_args()
    evaluate_cli(args)
