#!/usr/bin/env python
"""TranslatePsy en->xx sample -> cheap filters -> protected-index scan (both sides) -> clean windows (english, target, iso) for gemma_mt.data.prepare_sft.

    python tp_prepare.py --in tp_sample/ --out tp_clean/ --index <protected_index.npz> --manifest <protected_manifest.json> --decon-dir <dir with decon.py + protected_index_loader.py>
                         [--limit-per-lang N] [--plant FLORES_ENGLISH_JSONL]

Order of the steps (every step's drop counts are written to <out>/prepare_report.json and <out>/per_language.csv):
  1. NFC, strip; drop empty; drop source == target; drop pairs outside the length / ratio window (R10 tp_build.py: 3..1500 chars source, 3..2500 target, ratio 0.25..4.5);
     drop rows with a URL, a control character, or an English-word share of the target >= 0.45 (code-mixing; R10 used the same threshold); drop 3-gram-repetitive targets (>= 0.30);
     drop exact duplicates (casefolded pair).
  2. Protected-index scan (the intake filter's machinery, `decon.Index.check_texts`: rules R1 exact sentence / sentence part, R2 contiguous (protected sentence of >= 8 tokens
     inside the text), R3 any shared word 8-gram; both normal forms (marks kept / stripped); protected sentences with < 3 tokens are never indexed, which is the "3-word rule")
     on the SOURCE and on the TARGET text separately. A row is dropped when either side has any hit. Hits are counted per protected set (FLORES+ dev / devtest, NTREX, MAFAND,
     SmolSent, SALT, GlotStoryBook, IbomMT, FLORES-MOZ, AfriScience-MT, ...) and per side. The FLORES+ dev train half is exempt (index exemption file), exactly as in the SFT build.
  3. Optional positive control (--plant): benchmark English sentences are scanned through the same code path; every one must be flagged.
GlotLID acceptance, AfriCOMET-QE and SSA-COMET gates are NOT applied here (GPU / fastText model not on the laptop): see the box commands in the report.
"""
import argparse, collections, csv, json, os, re, sys, time, unicodedata

ENG = set("the of and to in is that for it as was with be by on not he she this are or his from at which but have an had they you were their one all we can her has there been if more when will would who so no".split())
URL = re.compile(r"(https?://|www\.|\S+@\S+\.\S+)", re.I)
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WORD = re.compile(r"[^\W\d_]+", re.U)


def rep3(t):
    w = t.lower().split()
    if len(w) < 6: return 0.0
    g = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    c = collections.Counter(g)
    return sum(v for v in c.values() if v > 1) / len(g)


def eng_share(t):
    w = [x.lower() for x in WORD.findall(t)]
    return (sum(1 for x in w if x in ENG) / len(w)) if w else 0.0


def cheap_filter(src, tgt):
    if not src or not tgt: return "empty"
    if src.casefold() == tgt.casefold(): return "copy"
    if not (3 <= len(src) <= 1500 and 3 <= len(tgt) <= 2500): return "length"
    r = len(tgt) / max(1, len(src))
    if not (0.25 <= r <= 4.5): return "ratio"
    if URL.search(src) or URL.search(tgt): return "url"
    if CTRL.search(src) or CTRL.search(tgt): return "control_char"
    if eng_share(tgt) >= 0.45: return "english_in_target"
    if rep3(tgt) >= 0.30: return "repetition"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--index", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--decon-dir", required=True)
    ap.add_argument("--limit-per-lang", type=int, default=0); ap.add_argument("--plant", default=None, help="jsonl with a `text` field of benchmark English sentences for the positive control")
    ap.add_argument("--langs", default="")
    a = ap.parse_args()
    sys.path.insert(0, a.decon_dir)
    import protected_index_loader as L
    t0 = time.time()
    ix, man = L.load(a.index, a.manifest)
    print("index", man["version"], "loaded in %.0fs" % (time.time() - t0), flush=True)
    os.makedirs(a.out, exist_ok=True)
    rules = ("R1_exact", "R2_contiguous", "R3_8gram")
    report = {"index_version": man["version"], "index_file_sha256": man["index_file_sha256"], "per_language": {}, "hits_by_set": {"source": collections.Counter(), "target": collections.Counter()},
              "hits_by_rule": {"source": collections.Counter(), "target": collections.Counter()},
              "rows_by_set": {"source": collections.Counter(), "target": collections.Counter()}}   # rows with >= 1 hit (union over the three rules), per protected set

    if a.plant:
        texts = [json.loads(l)["text"] for l in open(a.plant, encoding="utf-8")][:1012]
        res = ix.check_texts(texts)
        flagged = sum(1 for r in res if any(r[:3]))
        report["positive_control"] = {"file": os.path.basename(a.plant), "n": len(texts), "flagged": flagged,
                                      "sets": collections.Counter(n for r in res for m in r[:3] for n in ix.names(m))}
        print("positive control", flagged, "of", len(texts), flush=True)

    langs = [l for l in a.langs.split(",") if l] or sorted(f[5:-6] for f in os.listdir(a.inp) if f.startswith("en2x_") and f.endswith(".jsonl"))
    rows_csv = []
    for iso in langs:
        st = collections.Counter(); seen = set(); keep = []
        for line in open(f"{a.inp}/en2x_{iso}.jsonl", encoding="utf-8"):
            r = json.loads(line); st["read"] += 1
            if a.limit_per_lang and st["read"] > a.limit_per_lang: st["read"] -= 1; break
            s = unicodedata.normalize("NFC", r["source"]).strip(); t = unicodedata.normalize("NFC", r["target"]).strip()
            why = cheap_filter(s, t)
            if why: st["drop_" + why] += 1; continue
            k = (s.casefold(), t.casefold())
            if k in seen: st["drop_duplicate"] += 1; continue
            seen.add(k); keep.append((s, t, r["shard"], r["row"]))
        st["after_cheap"] = len(keep)
        out = []
        B = 2000
        for i in range(0, len(keep), B):
            chunk = keep[i:i + B]
            rs = ix.check_texts([c[0] for c in chunk]); rt = ix.check_texts([c[1] for c in chunk])
            for c, x, y in zip(chunk, rs, rt):
                hs, ht = any(x[:3]), any(y[:3])
                for side, h, res in (("source", hs, x), ("target", ht, y)):
                    if h:
                        for rule, m in zip(rules, res[:3]):
                            if m:
                                report["hits_by_rule"][side][rule] += 1
                                for nm in ix.names(m): report["hits_by_set"][side][nm] += 1
                for side, res in (("source", x), ("target", y)):
                    for nm in ix.names(res[0] | res[1] | res[2]): report["rows_by_set"][side][nm] += 1
                if hs or ht:
                    st["drop_protected_" + ("both" if hs and ht else "source" if hs else "target")] += 1
                else:
                    out.append(c)
        st["kept"] = len(out)
        with open(f"{a.out}/en2x_{iso}.jsonl", "w", encoding="utf-8") as f:
            for s, t, sh, rw in out:
                f.write(json.dumps({"english": s, "target": t, "iso": iso, "shard": sh, "row": rw}, ensure_ascii=False) + "\n")
        report["per_language"][iso] = dict(st)
        rows_csv.append({"iso": iso, **{k: st.get(k, 0) for k in ("read", "drop_empty", "drop_copy", "drop_length", "drop_ratio", "drop_url", "drop_control_char", "drop_english_in_target", "drop_repetition", "drop_duplicate", "after_cheap", "drop_protected_source", "drop_protected_target", "drop_protected_both", "kept")}})
        print(iso, dict(st), "%.0fs" % (time.time() - t0), flush=True)
    for k in ("hits_by_set", "hits_by_rule", "rows_by_set"):
        report[k] = {side: dict(c) for side, c in report[k].items()}
    if "positive_control" in report: report["positive_control"]["sets"] = dict(report["positive_control"]["sets"])
    json.dump(report, open(f"{a.out}/prepare_report.json", "w"), indent=1)
    with open(f"{a.out}/per_language.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys())); w.writeheader(); w.writerows(rows_csv)
    print("done", "%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
