#!/usr/bin/env python3
"""Benchmark contamination audit of the NEW SFT text of the 385M model (CPU only, streams the packed datasets from the bucket, nothing is stored but the results).

Reuses the rules of `gemma_mt.eval.contamination` (the audit scanner of contam_audit_07/10/12 and of `sft_benchmarks --contamination-dir`): per benchmark row
    exact      the normalised English sentence equals a sentence of a training window OR occurs inside it as a contiguous token run   (flag `contaminated`)
    sentence   ... only as a whole sentence of a window (the dev-filter rule; column n_sentence_rule)
    near       not exact, but >= 50% of its 8-token n-grams occur in the training English (flag `near_dup`)
    pair       exact AND the reference occurs in the TARGET side of a training window of the same language (flag `pair_seen`)
and adds the TARGET-side scan the first scanner did not have: the same machinery on a second index built from the REFERENCES (target-language sentences), scanned against the
non-English side of every window (`ref_any`: reference sentence / contiguous run anywhere in the other side of any window; `ref_same_iso`: in a window of the same language;
`ref_sentence_any`: whole-sentence rule only). `any8` = the row's English shares at least one 8-gram with the training English (the strict R3 rule of the intake filter).

Input: packed datasets (train/{tokens.u32.bin,sample_offset,sample_len,prompt_len,target_len,iso_id,direction}.npy + meta.json) in the object-storage bucket; the token file is read by
RANGED GETs (no big download), a unit = UNIT samples, the result of a unit is saved when it is complete (resumable).

    python contam_audit_14_385m_scan.py scan --work DIR --bench-root BENCH/test_sets --extra-root EXTRA --tree gemma_mt_tree --tokenizer tokenizer.json \\
        --dataset g0=<S3_PREFIX>/g0/packed_g0 --dataset s1=<S3_PREFIX>/shards/s1 ... --processes 6 [--limit-samples N]
    python contam_audit_14_385m_scan.py finalize --work DIR --bench-root ... --extra-root ... --tree ... --out-dir DIR/contamination
"""
import argparse, csv, glob, json, os, pickle, re, sys, time, unicodedata
from collections import Counter, defaultdict

if os.environ.get("PYTHONHASHSEED") != "0":                    # n-gram hashes are stored in the unit files: they must be comparable across runs
    os.environ["PYTHONHASHSEED"] = "0"; os.environ.setdefault("TOKENIZERS_PARALLELISM", "false"); os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.execv(sys.executable, [sys.executable] + sys.argv)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false"); os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np

UNIT = 1_000_000
CHUNK = 50_000
HIT_CAP = 200_000
EVAL_BENCHMARKS = ["flores_plus", "ntrex", "mafand", "smol_sent", "ibom_mt", "flores_moz", "glot_story", "salt"]
EXTRA_BENCHMARKS = ["flores_plus_dev", "afriscience_test", "afriscience_dev"]
BUCKET = os.environ['S3_BUCKET']
_SPEC = re.compile(r"<(?:bos|eos|pad|start_of_turn|end_of_turn)>")
_ROLE = re.compile(r"^\s*(?:user|model|system)\n")
_WRAP = re.compile(r"^Translate the text to [^:\n]{1,60}:\s+")
_TAG = re.compile(r"^\s*<(?:unused\d+|[a-z]{3})>\s*")
G = {}


# ------------------------------------------------------------------------------------------------ benchmark rows and the two indexes
def load_rows(a):
    sys.path.insert(0, a.tree)
    from gemma_mt.eval import benchmarks as B
    import pyarrow.parquet as pq
    from gemma_mt.languages import canonical_iso
    sets = {}
    for n in EVAL_BENCHMARKS:
        sets[n] = B.load_benchmark(n, path=os.path.join(a.bench_root, n))
    rows = B.load_flores_plus(split="dev", path=os.path.join(a.extra_root, "flores_plus"))
    for r in rows: r["benchmark"] = "flores_plus_dev"
    hp = os.path.join(a.extra_root, "devhalf_ids.json")
    if os.path.exists(hp):                                  # the D2 rule: the train half of the dev ids is exempt (20 languages), the held-out half is protected
        h = json.load(open(hp)); ho = {str(x) for x in h["heldout_ids"]}; tr = {str(x) for x in h["train_ids"]}
        sets["flores_plus_dev_heldout"] = [r for r in rows if r["id"] in ho]; sets["flores_plus_dev_trainhalf"] = [r for r in rows if r["id"] in tr]
        for r in rows:
            if r["id"] not in ho and r["id"] not in tr: sets.setdefault("flores_plus_dev_other", []).append(r)
    else:
        sets["flores_plus_dev"] = rows
    for sp in ("test", "dev"):
        out = []
        t = pq.read_table(os.path.join(a.extra_root, "afsci", f"{sp}-00000.parquet")).to_pylist()
        for r in t:
            iso = canonical_iso(r["target_lang"]) if r.get("source_lang") == "eng" else None
            if iso and r["source"] and r["target"]:
                out.append({"english": r["source"].strip(), "reference": r["target"].strip(), "iso": iso, "benchmark": f"afriscience_{sp}", "id": f"{r['paper_id']}#{r['sentence_id']}"})
        sets[f"afriscience_{sp}"] = out
    return sets


def build_indexes(sets):
    from gemma_mt.eval import contamination as C
    rows = [r for rs in sets.values() for r in rs]
    fwd = C.BenchIndex(rows)
    rev = C.BenchIndex([{"english": r["reference"], "reference": r["english"], "iso": r["iso"]} for r in rows])
    return fwd, rev


# ------------------------------------------------------------------------------------------------ scanning (rules of contamination.scan_windows, applied to BOTH sides of every sample)
# The packed samples are not reliably "English prompt -> target": mined and JW/Bible bitext contains flipped pairs (prompt "Translate the text to Yoruba:" followed by Yoruba text,
# target = English; seen in shard 1), so the benchmark English can sit on either side. Every sample therefore contributes two texts (prompt side, target side); each text is looked up
# in the English index AND in the reference index; a PAIR is a hit of an English sentence in one text together with its reference (same language) in the other text.
class Acc:
    def __init__(self):
        self.fwd_exact, self.fwd_sent, self.grams, self.pair = set(), set(), set(), set()
        self.rev_grams = set()                                                                   # reference 8-grams seen in the training text (near rule on the target side)      # fwd = benchmark English sentences
        self.rev_exact, self.rev_sent, self.rev_iso = set(), set(), set()                        # rev = benchmark reference (target-language) sentences; rev_iso: (id, iso of the sample)
        self.fwd_side = [set(), set()]                                                           # English ids found on the prompt side / the target side
        self.windows = 0; self.unique_texts = 0

    def merge(self, o):
        for k in ("fwd_exact", "fwd_sent", "grams", "pair", "rev_exact", "rev_sent", "rev_iso", "rev_grams"): getattr(self, k).update(getattr(o, k))
        for j in (0, 1): self.fwd_side[j] |= o.fwd_side[j]
        self.windows += o.windows; self.unique_texts += o.unique_texts


def scan_items(fwd, rev, items, tr_flags, hits_out):
    """items: list of (prompt side, target side, iso). Mirrors contamination.scan_windows (exact = sentence or contiguous run, sentence rule, 8-gram set of the near rule, pair)."""
    from gemma_mt.eval.contamination import normalise, _grams, _SENT_SPLIT
    n = fwd.n; acc = Acc(); cache = {}

    def analyse(text):
        ent = cache.get(text)
        if ent is not None: return ent
        acc.unique_texts += 1
        sents, toks = [], []
        for sent in _SENT_SPLIT.split(text):
            ns = normalise(sent)
            if ns: sents.append(ns); toks.extend(ns.split())
        res = []
        gs = _grams(toks, n) if len(toks) >= n else []
        joined = None
        for index in (fwd, rev):
            found, sh = set(), set()
            for ns in sents:
                i = index.ids.get(ns)
                if i is not None: found.add(i); sh.add(i)
            for g in gs:
                if g not in index.gram_to_strings: continue
                (acc.grams if index is fwd else acc.rev_grams).add(g)
                fg = index.first_gram.get(g)
                if fg:
                    if joined is None: joined = " " + " ".join(toks) + " "
                    for i in fg:
                        if i not in found and (" " + index.texts[i] + " ") in joined: found.add(i)
            res.append((tuple(sorted(found)), frozenset(sh)))
        ent = cache[text] = tuple(res)
        acc.fwd_exact.update(ent[0][0]); acc.fwd_sent.update(ent[0][1]); acc.rev_exact.update(ent[1][0]); acc.rev_sent.update(ent[1][1])
        return ent

    for idx, (p, t, iso) in enumerate(items):
        acc.windows += 1
        ents = (analyse(p), analyse(t))
        for side, ent in enumerate(ents):
            (fo, fs), (ro, rs_) = ent
            for i in fo:
                acc.fwd_side[side].add(i)
                if len(hits_out) < HIT_CAP: hits_out.append((idx, i, 0, 0 if i in fs else 1, side, tr_flags[idx]))
            for i in ro:
                acc.rev_iso.add((i, iso))
                if len(hits_out) < HIT_CAP: hits_out.append((idx, i, 1, 0 if i in rs_ else 1, side, tr_flags[idx]))
            if fo:
                other = t if side == 0 else p
                if other:
                    tnorm = " " + " ".join(normalise(x) for x in _SENT_SPLIT.split(other)) + " "
                    for i in fo:
                        for ref in fwd.refs[i].get(iso, ()):
                            if ref and (" " + ref + " ") in tnorm: acc.pair.add((i, iso))
    return acc


def _s3():
    sys.path.insert(0, "/workspace/s3lib")
    import boto3
    from botocore.config import Config
    if "s3" not in G:
        for l in open("/root/.bucket_env"):
            if "=" in l: k, v = l.strip().split("=", 1); os.environ[k] = v
        G["s3"] = boto3.client("s3", endpoint_url=os.environ['S3_ENDPOINT_URL'], aws_access_key_id=os.environ["AK"], aws_secret_access_key=os.environ["SK"], region_name=os.environ.get("S3_REGION", "us-east-1"),
                               config=Config(retries={"max_attempts": 8}, read_timeout=120, connect_timeout=30))
    return G["s3"]


def _get(key, lo, hi):
    for t in range(8):
        try:
            return _s3().get_object(Bucket=BUCKET, Key=key, Range=f"bytes={lo}-{hi - 1}")["Body"].read()
        except Exception:
            G.pop("s3", None); time.sleep(3 * (t + 1))
    raise RuntimeError(f"cannot read {key} {lo}-{hi}")


def fetch_range(key, lo, hi, piece=8 << 20, threads=4):
    """bytes lo..hi of an object: pieces of `piece` bytes on `threads` connections (the bucket gives about 0.5-2 MB/s per connection)."""
    from concurrent.futures import ThreadPoolExecutor
    cuts = list(range(lo, hi, piece)) + [hi]
    with ThreadPoolExecutor(threads) as ex:
        return b"".join(ex.map(lambda i: _get(key, cuts[i], cuts[i + 1]), range(len(cuts) - 1)))


def clean_prompt(p):
    p = _ROLE.sub("", _SPEC.sub("\n", p), count=1).lstrip()
    m = _WRAP.match(p)
    return (p[m.end():] if m else p), bool(m)


def decode_chunk(name, lo, hi):
    """(english side, other side, iso, is_translation) lists of the samples lo..hi of a dataset; chat markers, the translation wrapper and the language tag removed."""
    from tokenizers import Tokenizer
    if "tk" not in G: G["tk"] = Tokenizer.from_file(G["tokenizer"])
    tk = G["tk"]; A = G["arr"][name]; table = G["iso_table"][name]
    so = A["sample_offset"][lo:hi].astype(np.int64); sl = A["sample_len"][lo:hi].astype(np.int64); pl = A["prompt_len"][lo:hi].astype(np.int64)
    tl = A["target_len"][lo:hi].astype(np.int64); iso = A["iso_id"][lo:hi]; dr = A["direction"][lo:hi]
    b0, b1 = int(so.min()), int((so + sl).max())
    buf = np.frombuffer(fetch_range(G["prefix"][name] + "/train/tokens.u32.bin", b0 * 4, b1 * 4), dtype="<u4")
    prompts = [buf[s - b0: s - b0 + p].tolist() for s, p in zip(so, pl)]
    targets = [buf[s - b0 + p: s - b0 + p + t].tolist() for s, p, t in zip(so, pl, tl)]
    P = tk.decode_batch(prompts, skip_special_tokens=False); T = tk.decode_batch(targets, skip_special_tokens=False)
    en, ot, isos, tr = [], [], [], []
    for p, t, d, i in zip(P, T, dr, iso):
        p, is_tr = clean_prompt(p); t = _SPEC.sub(" ", _TAG.sub("", t, count=1)).strip()
        if d == 1: p, t = t, p
        en.append(p); ot.append(t); isos.append(table[i] if 0 <= i < len(table) else ""); tr.append(bool(is_tr))
    return en, ot, isos, tr, int(tl.sum() + pl.sum())


def chunk_job(job):
    name, lo, hi = job
    en, ot, isos, tr, ntok = decode_chunk(name, lo, hi)
    hits = []
    acc = scan_items(G["fwd"], G["rev"], list(zip(en, ot, isos)), tr, hits)
    acc.hits = [(lo + h[0],) + h[1:] for h in hits]; acc.n = hi - lo; acc.n_tr = int(sum(tr)); acc.tokens = ntok; acc.chars_en = sum(map(len, en)); acc.chars_ot = sum(map(len, ot))
    return name, lo, acc


def merge_unit(acc, part):
    if acc is None:
        acc = Acc(); acc.hits = []; acc.n = acc.n_tr = acc.tokens = acc.chars_en = acc.chars_ot = acc.chunks = 0
    acc.merge(part)
    if len(acc.hits) < HIT_CAP * 5: acc.hits.extend(part.hits)
    for k in ("n", "n_tr", "tokens", "chars_en", "chars_ot"): setattr(acc, k, getattr(acc, k) + getattr(part, k))
    acc.chunks += 1
    return acc


def cmd_scan(a):
    sys.path.insert(0, a.tree)
    from multiprocessing import get_context
    os.makedirs(os.path.join(a.work, "units"), exist_ok=True); os.makedirs(os.path.join(a.work, "arrays"), exist_ok=True)
    t0 = time.time(); sets = load_rows(a); fwd, rev = build_indexes(sets)
    print(f"[scan] rows {sum(len(v) for v in sets.values())} ({', '.join(f'{k} {len(v)}' for k, v in sets.items())}); fwd index {len(fwd.texts)} sentences / {len(fwd.gram_to_strings)} 8-grams; "
          f"rev index {len(rev.texts)} / {len(rev.gram_to_strings)}; {time.time() - t0:.0f}s", flush=True)
    names = [d.split("=")[0] for d in a.dataset]; G["prefix"] = dict(d.split("=", 1) for d in a.dataset); G["arr"] = {}; G["iso_table"] = {}; G["tokenizer"] = a.tokenizer
    for n in names:                                                     # the small arrays go to disk once (a few hundred MB in all), the token file is never stored
        d = os.path.join(a.work, "arrays", n); os.makedirs(d, exist_ok=True)
        for fn in ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "direction"):
            p = f"{d}/{fn}.npy"; key = f"{G['prefix'][n]}/train/{fn}.npy"
            size = _s3().head_object(Bucket=BUCKET, Key=key)["ContentLength"]
            if not os.path.exists(p) or os.path.getsize(p) != size:
                open(p + ".tmp", "wb").write(fetch_range(key, 0, size)); os.replace(p + ".tmp", p)
        mp = f"{d}/meta.json"
        if not os.path.exists(mp): open(mp, "wb").write(fetch_range(f"{G['prefix'][n]}/meta.json", 0, _s3().head_object(Bucket=BUCKET, Key=f"{G['prefix'][n]}/meta.json")["ContentLength"]))
        G["arr"][n] = {fn: np.load(f"{d}/{fn}.npy", mmap_mode="r") for fn in ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "direction")}
        G["iso_table"][n] = json.load(open(mp))["iso_table"]
    G["fwd"], G["rev"] = fwd, rev
    jobs, units = [], {}
    for n in names:
        tot = len(G["arr"][n]["sample_offset"])
        if a.limit_samples: tot = min(tot, a.limit_samples)
        for k, lo in enumerate(range(0, tot, UNIT)):
            uh = min(lo + UNIT, tot); up = os.path.join(a.work, "units", f"{n}_{k:03d}.pkl")
            if os.path.exists(up): continue
            units[(n, k)] = dict(path=up, left=-(-(uh - lo) // CHUNK), acc=None, t0=time.time(), lo=lo, hi=uh)
            for c in range(lo, uh, CHUNK): jobs.append((n, c, min(c + CHUNK, uh), k))
    print(f"[scan] {len(jobs)} chunk jobs in {len(units)} units open; {a.processes} processes", flush=True)
    if not jobs: return
    ctx = get_context("fork"); done = 0
    with ctx.Pool(a.processes) as pool:
        for name, lo, part in pool.imap_unordered(_job_wrap, jobs):
            k = lo // UNIT; u = units[(name, k)]
            u["acc"] = merge_unit(u["acc"], part); u["left"] -= 1; done += 1
            if u["left"] == 0:
                acc = u["acc"]; acc.unit = (name, k, u["lo"], u["hi"]); acc.seconds = time.time() - u["t0"]
                tmp = u["path"] + ".tmp"; pickle.dump(acc, open(tmp, "wb"), protocol=4); os.replace(tmp, u["path"]); u["acc"] = None
                print(f"[scan] unit {name}_{k:03d} done: {acc.n} samples ({acc.n_tr} translation), English exact {len(acc.fwd_exact)}, reference exact {len(acc.rev_exact)}, "
                      f"pairs {len(acc.pair)}, hits {len(acc.hits)}; {done}/{len(jobs)} chunks, {time.time() - t0:.0f}s total", flush=True)


def _job_wrap(job):
    n, lo, hi, k = job
    return chunk_job((n, lo, hi))


# ------------------------------------------------------------------------------------------------ finalize: flags, files, tables
def cmd_finalize(a):
    sys.path.insert(0, a.tree)
    from gemma_mt.eval import contamination as C
    sets = load_rows(a); fwd, rev = build_indexes(sets)
    files = sorted(glob.glob(os.path.join(a.work, "units", "*.pkl")))
    T = Acc(); stats = defaultdict(Counter); hits = []
    for f in files:
        u = pickle.load(open(f, "rb")); T.merge(u)
        name = u.unit[0]
        for k in ("n", "n_tr", "tokens"): stats[name][k] += getattr(u, k)
        hits.extend((name,) + h for h in u.hits)
    S = C.ScanResult(); S.exact, S.sentence, S.grams, S.pair, S.windows = T.fwd_exact, T.fwd_sent, T.grams, T.pair, T.windows
    print(f"[final] {len(files)} units; samples {T.windows}; English exact {len(T.fwd_exact)}/{len(fwd.texts)}; reference exact {len(T.rev_exact)}/{len(rev.texts)}; pairs {len(T.pair)}", flush=True)
    os.makedirs(os.path.join(a.out_dir, "extra"), exist_ok=True)
    rowsout = []; summary = {}
    ov_any = {}
    for i, s in enumerate(fwd.texts):
        gs = set(C._grams(s.split(), fwd.n)); ov_any[i] = (sum(1 for g in gs if g in T.grams) if gs else 0)
    rev_cnt, rev_ov = {}, {}
    for j, s_ in enumerate(rev.texts):
        gs = set(C._grams(s_.split(), rev.n)); k_ = sum(1 for g in gs if g in T.rev_grams) if gs else 0
        rev_cnt[j] = k_; rev_ov[j] = (k_ / len(gs)) if gs else 0.0
    for name, rows in sets.items():
        flags = C.flags_for_rows(rows, fwd, S)
        extra = {"training_text": "385M SFT text: " + ", ".join(sorted(stats)), "training_samples": T.windows, "scanner": "contam_audit_14_385m_scan.py (rules of gemma_mt.eval.contamination, both sides of every sample)"}
        C.write_flag_file(os.path.join(a.out_dir, f"contamination_{name}.json") if name in EVAL_BENCHMARKS else os.path.join(a.out_dir, "extra", f"contamination_{name}.json"), name, rows, flags, extra)
        per = defaultdict(Counter); tside = {"ref_any": {}, "ref_same_iso": {}, "ref_sentence_any": {}, "any8": {}, "ref_near": {}, "ref_any8": {}}
        for r in rows:
            k = C.row_key(r["iso"], r["id"]); iso = r["iso"]
            i = fwd.ids.get(C.normalise(r["english"])); j = rev.ids.get(C.normalise(r["reference"]))
            ra = j is not None and j in T.rev_exact; rs = j is not None and (j, iso) in T.rev_iso; rsent = j is not None and j in T.rev_sent
            a8 = i is not None and ov_any[i] > 0
            rn = j is not None and not ra and rev_ov[j] >= C.NEAR_THRESHOLD; r8 = j is not None and rev_cnt[j] > 0
            tside["ref_near"][k] = bool(rn); tside["ref_any8"][k] = bool(r8); d_ = per[iso]; d_["ref_near"] += int(rn); d_["ref_any8"] += int(r8)
            tside["ref_any"][k] = bool(ra); tside["ref_same_iso"][k] = bool(rs); tside["ref_sentence_any"][k] = bool(rsent); tside["any8"][k] = bool(a8)
            d = per[iso]; d["n"] += 1
            for fl in ("exact", "sentence", "near", "pair"): d[fl] += int(flags[fl][k])
            d["ref_any"] += int(ra); d["ref_same_iso"] += int(rs); d["ref_sentence_any"] += int(rsent); d["any8"] += int(a8)
            d["en_or_ref"] += int(flags["exact"][k] or ra); d["en_and_ref"] += int(flags["exact"][k] and ra)
            d["strict_union"] += int(flags["exact"][k] or flags["near"][k] or flags["pair"][k] or rs)
        with open(os.path.join(a.out_dir, "extra", f"targetside_{name}.json"), "w") as fh: json.dump(tside, fh, separators=(",", ":"))
        for iso, d in sorted(per.items()):
            rowsout.append(dict(benchmark=name, iso=iso, **{k: d[k] for k in ("n", "exact", "sentence", "near", "pair", "ref_any", "ref_same_iso", "ref_sentence_any", "ref_near", "any8", "ref_any8", "en_or_ref", "en_and_ref", "strict_union")}))
        tot = Counter()
        for d in per.values(): tot.update(d)
        summary[name] = dict(tot)
    with open(os.path.join(a.out_dir, "contamination_by_benchmark_language.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rowsout[0].keys())); w.writeheader(); w.writerows(rowsout)
    json.dump(dict(summary=summary, units=len(files), samples=T.windows, datasets={k: dict(v) for k, v in stats.items()}, fwd_index=[len(fwd.texts), len(fwd.gram_to_strings)],
                   rev_index=[len(rev.texts), len(rev.gram_to_strings)], english_found_prompt_side=len(T.fwd_side[0]), english_found_target_side=len(T.fwd_side[1])), open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    pickle.dump(dict(hits=hits, fwd_texts=fwd.texts, rev_texts=rev.texts), open(os.path.join(a.work, "hits.pkl"), "wb"), protocol=4)
    for name, d in summary.items():
        print(f"[final] {name}: n {d['n']} exact {d['exact']} sentence {d['sentence']} near {d['near']} pair {d['pair']} | ref_any {d['ref_any']} ref_same_iso {d['ref_same_iso']} ref_near {d['ref_near']} any8 {d['any8']} ref_any8 {d['ref_any8']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("scan", "finalize")); ap.add_argument("--work", required=True); ap.add_argument("--bench-root", required=True); ap.add_argument("--extra-root", required=True)
    ap.add_argument("--tree", required=True); ap.add_argument("--tokenizer"); ap.add_argument("--dataset", action="append", default=[], metavar="NAME=bucket/prefix")
    ap.add_argument("--processes", type=int, default=6); ap.add_argument("--limit-samples", type=int, default=0); ap.add_argument("--out-dir")
    a = ap.parse_args()
    {"scan": cmd_scan, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    main()
