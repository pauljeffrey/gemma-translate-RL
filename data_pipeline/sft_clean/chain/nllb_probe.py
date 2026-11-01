#!/usr/bin/env python3
"""Quality probe of the OPUS NLLB (mined bitext) en<->African pairs. Streams the FIRST --scan lines of every pair through HTTP range requests (no full download),
samples --n pairs per language, applies the heuristics of the chain (ok(): lengths, ratio, letters, copy, URL) plus script and duplicate checks, writes the sample as a normal chain
input directory (parquet: english, target, iso, key), and later compares AfriCOMET-QE scores with the per-language CAL thresholds of the gate.

    python nllb_probe.py fetch  --out work/nllb_probe --n 3000 --scan 300000 [--langs af sw ...]
    <venv_comet> qe_score.py --src work/nllb_probe/pairs --out work/nllb_probe/scores --device cuda --half --batch 64 --chunk 20000 --workers 4
    python nllb_probe.py report --out work/nllb_probe --thresholds work/gate/thresholds.json

Limits: the first --scan lines are not a uniform sample of the whole file (OPUS may order by document or score); the report prints the QE trend over the scanned range.
"""
import argparse, io, json, os, random, re, statistics, sys, time, urllib.request, zipfile
import pyarrow as pa, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import key_of

LANG = {"af": "afr", "sw": "swh", "am": "amh", "so": "som", "rw": "kin", "sn": "sna", "xh": "xho", "tn": "tsn", "ig": "ibo", "zu": "zul", "ee": "ewe", "kab": "kab", "tw": "twi", "sg": "sag",
        "lg": "lug", "nso": "nso", "om": "orm", "ln": "lin", "luo": "luo", "lua": "lua", "ha": "hau", "mg": "mlg", "kg": "kon", "ki": "kik", "mos": "mos", "fon": "fon", "kam": "kam", "kbp": "kbp",
        "ny": "nya", "ff": "ful", "yo": "yor", "wo": "wol", "ti": "tir", "st": "sot", "kea": "kea", "nus": "nus", "rn": "run", "ts": "tso", "tum": "tum", "bem": "bem", "umb": "umb",
        "dyu": "dyu", "bm": "bam", "ss": "ssw", "kmb": "kmb", "tzm": "tzm"}
ETHIOPIC = {"amh", "tir"}
URL = re.compile(r"https?://|www\.", re.I)


class HttpRangeFile(io.RawIOBase):
    """read-only seekable file over HTTP Range requests with a 4 MB block cache (enough for zipfile)"""
    BLOCK = 4 << 20

    def __init__(self, url):
        self.url = url; self.pos = 0; self.cache = {}
        req = urllib.request.Request(url, method="HEAD"); r = urllib.request.urlopen(req, timeout=60); self.size = int(r.headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            if len(self.cache) > 8: self.cache.pop(next(iter(self.cache)))
            a = i * self.BLOCK; b = min(self.size, a + self.BLOCK) - 1
            for t in range(5):
                try:
                    req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"}); self.cache[i] = urllib.request.urlopen(req, timeout=120).read(); break
                except Exception:
                    time.sleep(2 + 3 * t)
            else:
                raise IOError("range request failed")
        return self.cache[i]

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0: return 0
        out = bytearray()
        while len(out) < n:
            i, o = divmod(self.pos + len(out), self.BLOCK); blk = self._block(i); out += blk[o:o + n - len(out)]
        buf[:n] = out; self.pos += n
        return n


def first_lines(url, member_suffix, scan):
    f = HttpRangeFile(url); z = zipfile.ZipFile(f)
    name = [n for n in z.namelist() if n.endswith(member_suffix)][0]
    out = []
    with z.open(name) as fh:
        for i, line in enumerate(io.TextIOWrapper(fh, encoding="utf-8", errors="replace")):
            if i >= scan: break
            out.append(line.rstrip("\n"))
    return out


def ok(en, tg):
    if not (3 <= len(en) <= 1500 and 3 <= len(tg) <= 2500): return False
    r = len(tg) / max(1, len(en))
    if not (0.25 <= r <= 4.5): return False
    if en.casefold() == tg.casefold() or URL.search(en) or URL.search(tg): return False
    if sum(c.isalpha() for c in en) / len(en) < 0.4 or not re.search(r"[A-Za-z]{2}", en): return False
    return True


def fetch(a):
    api = json.load(urllib.request.urlopen("https://opus.nlpl.eu/opusapi/?source=en&corpus=NLLB&preprocessing=moses&version=v1", timeout=120))["corpora"]
    pairs = {}
    for c in api:
        o = c["target"] if c["source"] == "en" else c["source"]
        if o in LANG: pairs[o] = c
    os.makedirs(os.path.join(a.out, "pairs"), exist_ok=True); stats = {}
    rng = random.Random(1234)
    for code in (a.langs or list(LANG)):
        c = pairs.get(code)
        if not c: print(code, "no pair"); continue
        iso = LANG[code]; t0 = time.time()
        try:
            en = first_lines(c["url"], ".en", a.scan); tg = first_lines(c["url"], "." + code, a.scan)
        except Exception as e:
            print(code, "FAILED", type(e).__name__, str(e)[:100]); continue
        n = min(len(en), len(tg)); idx = rng.sample(range(n), min(a.n, n))
        rows = [(i, en[i].strip(), tg[i].strip()) for i in sorted(idx)]
        st = {"pairs_in_corpus": c["alignment_pairs"], "scanned": n, "sampled": len(rows)}
        st["pass_ok"] = sum(ok(e, t) for _, e, t in rows) / max(1, len(rows))
        st["copy"] = sum(e.casefold() == t.casefold() for _, e, t in rows) / max(1, len(rows))
        ratios = [len(t) / max(1, len(e)) for _, e, t in rows]
        st["ratio_median"] = round(statistics.median(ratios), 2); st["ratio_out_0.5_2.5"] = sum(r < 0.5 or r > 2.5 for r in ratios) / max(1, len(ratios))
        st["dup_in_sample"] = 1 - len({(e.casefold(), t.casefold()) for _, e, t in rows}) / max(1, len(rows))
        if iso in ETHIOPIC:
            st["script_ok"] = sum(sum("ሀ" <= ch <= "፿" for ch in t) / max(1, len(t)) > 0.5 for _, e, t in rows) / max(1, len(rows))
        else:
            st["script_ok"] = sum(sum(ch.isalpha() and ch.isascii() or ch in " .,'-" or ord(ch) > 127 for ch in t) / max(1, len(t)) > 0.8 for _, e, t in rows) / max(1, len(rows))
        keep = [(i, e, t) for i, e, t in rows if ok(e, t)]
        tbl = pa.table({"english": [e for _, e, _ in keep], "target": [t for _, _, t in keep], "iso": [iso] * len(keep), "idx": [i for i, _, _ in keep],
                        "key": pa.array([key_of(e, iso, t) for _, e, t in keep], pa.uint64())})
        pq.write_table(tbl, os.path.join(a.out, "pairs", f"nllb_{iso}.parquet"))
        stats[code] = st; print(code, iso, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()}, f"{time.time() - t0:.0f}s", flush=True)
        json.dump(stats, open(os.path.join(a.out, "heuristics.json"), "w"), indent=1)


def report(a):
    import glob, numpy as np
    thr = json.load(open(a.thresholds))
    pooled = float(thr.get("pooled_offset", 0)); langs = thr["languages"]
    sc = {}
    for f in glob.glob(os.path.join(a.out, "scores", "*.parquet")):
        t = pq.read_table(f).to_pydict(); sc.update(dict(zip(t["key"], t["qe"])))
    H = json.load(open(os.path.join(a.out, "heuristics.json")))
    rows = []; ex = []
    for code, st in H.items():
        iso = LANG[code]; t = pq.read_table(os.path.join(a.out, "pairs", f"nllb_{iso}.parquet")).to_pydict()
        q = np.array([sc.get(k, np.nan) for k in t["key"]], float); idx = np.array(t["idx"]); m = ~np.isnan(q)
        if m.sum() == 0: continue
        L = langs.get(iso, {}); cal = L.get("thr_cal"); own = L.get("median")
        thirds = [float(np.nanmean(q[(idx >= lo) & (idx < hi)])) if ((idx >= lo) & (idx < hi) & m).any() else float("nan") for lo, hi in ((0, st["scanned"] / 3), (st["scanned"] / 3, 2 * st["scanned"] / 3), (2 * st["scanned"] / 3, 1e18))]
        rows.append({"code": code, "iso": iso, "corpus_pairs_M": round(st["pairs_in_corpus"] / 1e6, 2), "pass_ok": round(st["pass_ok"], 2), "script_ok": round(st["script_ok"], 2), "copy": round(st["copy"], 3),
                     "ratio_out": round(st["ratio_out_0.5_2.5"], 2), "dup": round(st["dup_in_sample"], 3), "n_scored": int(m.sum()), "qe_median": round(float(np.nanmedian(q)), 3),
                     "our_data_qe_median": round(own, 3) if own else None, "thr_cal": round(cal, 3) if cal else None,
                     "keep_at_our_gate": round(float(np.nanmean(q[m] >= cal)), 2) if cal else None, "trend_thirds": [round(x, 3) for x in thirds]})
        for i in range(min(3, len(t["english"]))): ex.append((iso, t["english"][i][:140], t["target"][i][:140], round(float(q[i]), 3)))
    rows.sort(key=lambda r: -r["corpus_pairs_M"])
    json.dump(rows, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(f"{'iso':4s} {'M pairs':>7s} {'ok':>5s} {'script':>6s} {'copy':>5s} {'dup':>5s} {'QE med':>6s} {'ours':>6s} {'thr':>6s} {'keep@gate':>9s}  trend(first/mid/last third of scanned range)")
    for r in rows:
        print(f"{r['iso']:4s} {r['corpus_pairs_M']:7.2f} {r['pass_ok']:5.2f} {r['script_ok']:6.2f} {r['copy']:5.3f} {r['dup']:5.3f} {r['qe_median']:6.3f} {str(r['our_data_qe_median']):>6s} {str(r['thr_cal']):>6s} {str(r['keep_at_our_gate']):>9s}  {r['trend_thirds']}")
    with open(os.path.join(a.out, "examples.tsv"), "w") as f:
        for e in ex: f.write("\t".join(map(str, e)) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--out", required=True); f.add_argument("--n", type=int, default=3000); f.add_argument("--scan", type=int, default=300000); f.add_argument("--langs", nargs="*")
    r = sub.add_parser("report"); r.add_argument("--out", required=True); r.add_argument("--thresholds", required=True)
    a = ap.parse_args(); {"fetch": fetch, "report": report}[a.cmd](a)
