# simulator/replay.py
import argparse, csv, time, requests

def replay(csv_file: str, base_url: str, rate: float | None, dry_run: bool):
    url = base_url.rstrip("/") + "/api/request"
    sent = ok = fail = 0

    def post_row(row):
        nonlocal ok, fail
        payload = {
            "ts": row["timestamp"],
            "client_id": row["client_id"],
            "object_id": row["object_id"],
            "object_size_bytes": int(row["object_size_bytes"]),
            "origin_latency_ms": int(row["origin_latency_ms"]),
            "was_write": row["was_write"].lower() == "true",
        }
        if dry_run:
            return True
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code == 200:
                ok += 1
                return True
            else:
                fail += 1
                print("POST failed:", r.status_code, r.text[:200])
                return False
        except Exception as e:
            fail += 1
            print("POST error:", e)
            return False

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sent += 1
            post_row(row)
            if rate and rate > 0:
                time.sleep(1.0 / rate)

    print(f"Done. Sent={sent} OK={ok} Fail={fail}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay CSV into FastAPI /api/request")
    ap.add_argument("--file", required=True, help="path to CSV")
    ap.add_argument("--base", default="http://127.0.0.1:8000", help="API base URL")
    ap.add_argument("--rate", type=float, default=10.0, help="requests per second; 0 to go as fast as possible")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.rate == 0:
        args.rate = None
    replay(args.file, args.base, args.rate, args.dry_run)
