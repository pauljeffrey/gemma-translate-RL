"""Extract only nairaland.com pages from Common Crawl (index -> ranged WARC fetch).

Stages (resumable; stdlib only):
  index : page the CDX API per crawl -> <out>/index/<crawl>.jsonl (status 200, html)
  fetch : ranged GET of each unique URL's latest capture -> <out>/pages.jsonl.gz

Run the fetch stage on the vast instance, not the laptop.

  python nairaland_cc.py --out /data/nairaland_cc --stage index
  python nairaland_cc.py --out /data/nairaland_cc --stage fetch --workers 16
"""
import argparse
import gzip
import json
import os
import random
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

IDX = "https://index.commoncrawl.org"
DATA = "https://data.commoncrawl.org"
UA = "sabiyarn-research-crawler/1.0 (<email>)"
DOMAIN = "nairaland.com"


def get(url, headers=None, timeout=120, tries=8):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # 502/504/timeouts are routine on the index server
            last = e
            blocked = getattr(e, "code", None) in (403, 429, 503)  # throttled: back off hard
            time.sleep(min(600 if blocked else 60, 2 ** i * (10 if blocked else 1)) + random.random())
    raise RuntimeError(f"giving up on {url}: {last}")


def crawls(only):
    allc = [c["id"] for c in json.loads(get(f"{IDX}/collinfo.json"))]
    return [c for c in allc if not only or c in only]


KEY_LO, KEY_HI = "com,nairaland)", "com,nairaland-"  # SURT keys for nairaland.com and its subdomains


def _is_html_ok(r):
    return r.get("status") == "200" and "html" in r.get("mime-detected", r.get("mime", ""))


def cluster_rows(c):
    """Index rows for DOMAIN straight from the raw CDX files (no flaky API):
    binary-search cluster.idx by key with ranged reads, then read only the matching ~3000-line blocks."""
    base = f"{DATA}/cc-index/collections/{c}/indexes"
    req = urllib.request.Request(f"{base}/cluster.idx", method="HEAD", headers={"User-Agent": UA})
    size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])

    def rd(a, n):
        return get(f"{base}/cluster.idx", {"Range": f"bytes={a}-{min(a + n, size) - 1}"})

    def key_at(pos):  # key of the first full line starting after pos
        chunk = rd(pos, 4096)
        nl = chunk.find(b"\n")
        return pos + nl + 1, chunk[nl + 1:].split(b" ", 1)[0].decode("utf-8", "replace")

    lo, hi = 0, size
    while hi - lo > 65536:  # invariant: line starting at lo has key < KEY_LO
        mid = (lo + hi) // 2
        start, k = key_at(mid)
        if k < KEY_LO:
            lo = start
        else:
            hi = mid
    blocks, prev, pos, done = [], None, lo, False
    while pos < size and not done:
        buf = rd(pos, 262144)
        lines = buf.split(b"\n")
        pos += len(buf)
        carry = lines.pop()  # possibly partial last line: re-read it next time
        pos -= len(carry)
        for ln in lines:
            if not ln:
                continue
            k = ln.split(b" ", 1)[0].decode("utf-8", "replace")
            if k >= KEY_HI:
                done = True
                break
            if k >= KEY_LO:
                if prev is not None:
                    blocks.append(prev)
                    prev = None
                blocks.append(ln)
            else:
                prev = ln
        if pos >= size:
            break
    rows = []
    for ln in blocks:
        _, fn, off, ln_, _seq = ln.decode().split("\t")
        raw = get(f"{base}/{fn}", {"Range": f"bytes={int(off)}-{int(off) + int(ln_) - 1}"})
        for l in gzip.decompress(raw).decode("utf-8", "replace").splitlines():
            key, ts, js = l.split(" ", 2)
            if key.startswith("com,nairaland)") or key.startswith("com,nairaland,"):
                r = json.loads(js)
                r.update(urlkey=key, timestamp=ts)
                if _is_html_ok(r):
                    rows.append(r)
    return rows


def index_stage(out, only):
    os.makedirs(f"{out}/index", exist_ok=True)
    for c in crawls(only):
        path = f"{out}/index/{c}.jsonl"
        if os.path.exists(path):
            continue
        try:
            rows = cluster_rows(c)
            with open(path + ".tmp", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            os.replace(path + ".tmp", path)
            print(f"{c}: {len(rows)} html captures (cluster.idx)", flush=True)
            continue
        except Exception as e:  # e.g. pre-2013 crawls have no cluster.idx: fall back to the API
            print(c, "cluster route failed, trying API:", str(e)[:120], flush=True)
        q = {"url": DOMAIN, "matchType": "domain", "output": "json"}
        base = f"{IDX}/{c}-index?"
        try:
            n = json.loads(get(base + urllib.parse.urlencode({**q, "showNumPages": "true"})))["pages"]
        except Exception as e:
            print(c, "index unavailable:", e)
            continue
        rows = []
        try:
            for p in range(n):
                for attempt in range(6):  # index pages are sometimes truncated: re-request
                    body = get(base + urllib.parse.urlencode({**q, "page": p})).decode("utf-8", "replace")
                    try:
                        page = [json.loads(l) for l in body.splitlines() if l.startswith("{")]
                        break
                    except json.JSONDecodeError:
                        time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"{c} page {p}: persistently malformed")
                rows += [r for r in page if r.get("status") == "200"
                         and "html" in r.get("mime-detected", r.get("mime", ""))]
        except RuntimeError as e:  # skip this crawl; a rerun retries it (no file written)
            print(c, "skipped:", e, flush=True)
            continue
        with open(path + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(path + ".tmp", path)
        print(f"{c}: {len(rows)} html captures", flush=True)


def parse_record(raw):
    rec = gzip.decompress(raw)
    if rec.startswith(b"WARC/"):
        _, _, rest = rec.partition(b"\r\n\r\n")  # WARC headers
    else:  # legacy ARC (2008-2009 crawls): one URL-record line, then the HTTP response
        _, _, rest = rec.partition(b"\n")
    head, _, body = rest.partition(b"\r\n\r\n")  # HTTP headers
    charset = "utf-8"
    for h in head.split(b"\r\n"):
        if h.lower().startswith(b"content-type") and b"charset=" in h.lower():
            charset = h.lower().split(b"charset=")[1].split(b";")[0].decode().strip() or "utf-8"
    try:
        return body.decode(charset, "replace")
    except LookupError:
        return body.decode("utf-8", "replace")


def fetch_stage(out, workers, only):
    latest = {}
    local = sorted(f[:-6] for f in os.listdir(f"{out}/index") if f.endswith(".jsonl"))
    for c in [x for x in local if not only or x in only]:  # no network: index server is flaky
        p = f"{out}/index/{c}.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            u = r["url"].split("#")[0]
            if u not in latest or r["timestamp"] > latest[u]["timestamp"]:
                latest[u] = r
    if not latest:
        raise SystemExit(f"no index rows under {out}/index; run --stage index first")
    sink = f"{out}/pages.jsonl.gz"
    done = set()
    if os.path.exists(sink):
        with gzip.open(sink, "rt") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["url"])
                except Exception:
                    pass
    todo = [r for u, r in latest.items() if u not in done]
    print(f"{len(latest)} unique urls, {len(done)} done, {len(todo)} to fetch", flush=True)
    lock = threading.Lock()
    sinkf = gzip.open(sink, "at")
    cnt = [0]

    def work(r):
        off, ln = int(r["offset"]), int(r["length"])
        try:
            raw = get(f"{DATA}/{r['filename']}", {"Range": f"bytes={off}-{off + ln - 1}"})
            html = parse_record(raw)
        except Exception as e:
            print("fail", r["url"], e, flush=True)
            return
        with lock:
            sinkf.write(json.dumps({"url": r["url"], "timestamp": r["timestamp"],
                                    "crawl": r["filename"].split("/")[1], "html": html}) + "\n")
            cnt[0] += 1
            if cnt[0] % 1000 == 0:
                sinkf.flush()
                print(cnt[0], "fetched", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))
    sinkf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["index", "fetch"], required=True)
    ap.add_argument("--crawl", action="append", help="restrict to these crawl ids")
    ap.add_argument("--workers", type=int, default=4)  # Common Crawl throttles (403) aggressive clients
    a = ap.parse_args()
    if a.stage == "index":
        index_stage(a.out, a.crawl)
    else:
        fetch_stage(a.out, a.workers, a.crawl)
