# simulator/make_csv.py
import argparse, csv, random, string
from datetime import datetime, timedelta, timezone

def rand_client(num_clients: int) -> str:
    return f"u{random.randint(1, num_clients)}"

def rand_size(min_kb=5, max_kb=500) -> int:
    return random.randint(min_kb*1024, max_kb*1024)

def rand_origin_latency(min_ms=150, max_ms=350) -> int:
    return random.randint(min_ms, max_ms)

def iso_utc(dt) -> str:
    # "YYYY-MM-DDTHH:MM:SSZ"
    return dt.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def choose_zipf_object(num_objects: int, s: float = 1.07) -> str:
    # weights ~ 1/(rank^s)
    ranks = list(range(1, num_objects+1))
    weights = [1.0/(r**s) for r in ranks]
    return f"item{random.choices(ranks, weights=weights, k=1)[0]}"

def write_csv(rows, outfile: str):
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","client_id","object_id","object_size_bytes","origin_latency_ms","was_write"])
        w.writerows(rows)

def gen_zipf(minutes: int, rps: int, num_objects: int, num_clients: int):
    """Skewed popularity: few objects dominate."""
    start = datetime.now(timezone.utc)
    total = minutes * 60 * rps
    rows = []
    for i in range(total):
        ts = iso_utc(start + timedelta(seconds=i / rps))
        obj = choose_zipf_object(num_objects)
        rows.append([ts, rand_client(num_clients), obj, rand_size(), rand_origin_latency(), "false"])
    return rows

def gen_flash(minutes: int, rps: int, num_objects: int, num_clients: int, spike_pct: float = 0.5):
    """
    Flash crowd: middle window spikes one hot object.
    spike_pct = fraction of the run that is 'hot' (0.5 => middle 50%).
    """
    start = datetime.now(timezone.utc)
    total = minutes * 60 * rps
    mid_start = int(total * (0.5 - spike_pct/2))
    mid_end   = int(total * (0.5 + spike_pct/2))
    hot_obj = f"item{random.randint(1, max(5, num_objects//10))}"  # early object becomes hot

    rows = []
    for i in range(total):
        ts = iso_utc(start + timedelta(seconds=i / rps))
        if mid_start <= i < mid_end:
            obj = hot_obj if random.random() < 0.8 else f"item{random.randint(1, num_objects)}"
        else:
            obj = f"item{random.randint(1, num_objects)}"
        rows.append([ts, rand_client(num_clients), obj, rand_size(), rand_origin_latency(), "false"])
    return rows

def gen_write_heavy(minutes: int, rps: int, num_objects: int, num_clients: int, write_every: int = 5):
    """
    Write-heavy: every Nth request is a write (refresh).
    """
    start = datetime.now(timezone.utc)
    total = minutes * 60 * rps
    rows = []
    for i in range(total):
        ts = iso_utc(start + timedelta(seconds=i / rps))
        obj = f"item{random.randint(1, num_objects)}"
        is_write = (i % write_every == 0)  # ~20% writes if N=5
        rows.append([ts, rand_client(num_clients), obj, rand_size(), rand_origin_latency(), "true" if is_write else "false"])
    return rows

def main():
    p = argparse.ArgumentParser(description="Generate synthetic request CSVs.")
    p.add_argument("--workload", choices=["zipf","flash","writeheavy"], required=True)
    p.add_argument("--minutes", type=int, default=2, help="duration in minutes")
    p.add_argument("--rps", type=int, default=5, help="requests per second")
    p.add_argument("--objects", type=int, default=200, help="number of distinct objects (item1..itemN)")
    p.add_argument("--clients", type=int, default=50, help="number of clients (u1..uN)")
    p.add_argument("--outfile", required=True)
    args = p.parse_args()

    if args.workload == "zipf":
        rows = gen_zipf(args.minutes, args.rps, args.objects, args.clients)
    elif args.workload == "flash":
        rows = gen_flash(args.minutes, args.rps, args.objects, args.clients)
    else:
        rows = gen_write_heavy(args.minutes, args.rps, args.objects, args.clients)

    write_csv(rows, args.outfile)
    print(f"Wrote {len(rows)} rows to {args.outfile}")

if __name__ == "__main__":
    main()
