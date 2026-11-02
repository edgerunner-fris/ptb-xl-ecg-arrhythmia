"""Download PTB-XL v1.0.3 (100 Hz subset) from the official PhysioNet S3 mirror.

PhysioNet's HTTPS download endpoint is throttled to ~5 KB/s from this network,
while the public `physionet-open` S3 bucket mirrors the same files with ETags
(MD5) for verification. We fetch records100/ plus the two metadata CSVs only.

Usage:  python src/download_ptbxl.py [--workers 32] [--verify-md5]
"""
import argparse
import hashlib
import sys
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

S3 = "https://physionet-open.s3.amazonaws.com"
PREFIX = "ptb-xl/1.0.3/"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "ptb-xl" / "1.0.3"
LOG = ROOT / "results" / "m0_download.log"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

_print_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _print_lock:
        print(line, flush=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def list_keys(prefix: str):
    token = None
    while True:
        url = f"{S3}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s3:Contents", NS):
            yield (
                c.find("s3:Key", NS).text,
                int(c.find("s3:Size", NS).text),
                c.find("s3:ETag", NS).text.strip('"'),
            )
        if root.find("s3:IsTruncated", NS).text != "true":
            break
        token = root.find("s3:NextContinuationToken", NS).text


def fetch_one(key: str, size: int, etag: str, verify_md5: bool) -> str:
    rel = key[len(PREFIX):]
    dest = OUT / rel
    if dest.exists() and dest.stat().st_size == size:
        if not verify_md5:
            return "skip"
        if hashlib.md5(dest.read_bytes()).hexdigest() == etag:
            return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{S3}/{urllib.parse.quote(key)}"
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                data = r.read()
            if len(data) != size:
                raise IOError(f"size {len(data)} != {size}")
            if "-" not in etag and hashlib.md5(data).hexdigest() != etag:
                raise IOError("md5 mismatch")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return "ok"
        except Exception as e:  # noqa: BLE001 - network errors are expected
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{key}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--verify-md5", action="store_true")
    args = ap.parse_args()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"start: prefix={PREFIX} out={OUT} workers={args.workers}")

    items = []
    for key, size, etag in list_keys(PREFIX):
        if key.startswith(PREFIX + "records100/") and key.endswith((".hea", ".dat")):
            items.append((key, size, etag))
        elif key in (PREFIX + "ptbxl_database.csv", PREFIX + "scp_statements.csv"):
            items.append((key, size, etag))
    log(f"listed {len(items)} objects, total {sum(i[1] for i in items)/1e6:.1f} MB")

    done = {"ok": 0, "skip": 0, "err": 0}
    t0 = time.time()
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(fetch_one, k, s, e, args.verify_md5): k
            for k, s, e in items
        }
        for n, fut in enumerate(as_completed(futs), 1):
            key = futs[fut]
            try:
                done[fut.result()] += 1
            except Exception as e:  # noqa: BLE001
                done["err"] += 1
                errors.append(str(e))
                log(f"ERROR {key}: {e}")
            if n % 1000 == 0 or n == len(items):
                el = time.time() - t0
                rate = n / el
                eta = (len(items) - n) / rate if rate else 0
                log(
                    f"{n}/{len(items)}  ok={done['ok']} skip={done['skip']} "
                    f"err={done['err']}  {rate:.0f} files/s  elapsed={el/60:.1f}m "
                    f"eta={eta/60:.1f}m"
                )

    log(f"finished: {done} in {(time.time()-t0)/60:.1f} min")
    if errors:
        log(f"{len(errors)} errors; first 3: {errors[:3]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
