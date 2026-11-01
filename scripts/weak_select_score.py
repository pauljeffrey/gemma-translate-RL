#!/usr/bin/env python3
"""W1 (2026-10-26): score the dev generations and apply DECISION_RULE.txt (reports/data/weak_selection_2026-10-26/). Nothing here is tuned after generation.

Inputs : dev_sample.jsonl, gen_raw.jsonl (+ optional --pair-flags file {"<iso>:<id>": true if the REFERENCE PAIR was seen in SFT}; a language without keys has an unknown flag)
Outputs: per_sentence.jsonl, per_language.csv (all 77), weak_list.json, ranking_table.md
"""
import argparse, csv, json, os, re, sys, unicodedata
import numpy as np
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "reports", "data", "weak_selection_2026-10-26")
sys.path.insert(0, os.path.join(ROOT, "gemma_mt"))
from gemma_mt.languages import LANGUAGES, SUPPORTED_RL
from sacrebleu.metrics import CHRF, BLEU

N_TOP, B, SEED = 20, 1000, 2026
FIN_MIN, LOOP_MAX, WRONG_MAX = 0.95, 0.05, 0.10
GLOTLID = os.path.expanduser("~/.cache/huggingface/hub/models--cis-lmu--glotlid/snapshots/85cd6716494360367b75f642b5bc78667605d0b4/model_v3.bin")
WORD_LOOP = re.compile(r"(?<!\S)(\S+(?:\s+\S+){0,5})(?:\s+\1){3,}(?!\S)")
CHAR_LOOP = re.compile(r"(.{3,20}?)\1{3,}")


def nfc(s): return unicodedata.normalize("NFC", s or "")


def is_loop(text, stop_type):
    if stop_type != "eos": return True
    t = text.strip()
    return bool(WORD_LOOP.search(t)) if " " in t else bool(CHAR_LOOP.search(t))


def load_lid():
    import fasttext
    m = fasttext.load_model(GLOTLID)
    def pred(text):
        t = " ".join(text.split())
        if not t: return None
        r = m.f.predict(t + "\n", 1, 0.0, "strict")        # fasttext's python predict() breaks with numpy 2; the C++ binding does not
        return r[0][1].replace("__label__", "") if r else None
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default=os.path.join(OUT, "gen_raw.jsonl"))
    ap.add_argument("--pair-flags", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sfx = a.tag
    sample = [json.loads(l) for l in open(os.path.join(OUT, "dev_sample.jsonl"), encoding="utf-8")]
    gen = {}
    for l in open(a.gen, encoding="utf-8"):
        r = json.loads(l); gen[(r["iso"], r["id"])] = r
    flags = json.load(open(a.pair_flags)) if a.pair_flags else None
    lid = load_lid()
    chrf, bleu = CHRF(word_order=2), BLEU(tokenize="flores200")
    per = defaultdict(list)
    for r in sample:
        g = gen.get((r["iso"], r["id"]))
        if g is None: continue
        raw = g["raw"].strip(); iso = r["iso"]
        tag_ok = raw.startswith(iso + " ")
        hyp = nfc(raw[len(iso) + 1:].strip() if tag_ok else raw)
        rec = dict(iso=iso, tier=r["tier"], id=r["id"], english=r["english"], reference=r["reference"], hyp=hyp, tag_ok=tag_ok,
                   finished=g["stop_type"] == "eos", loop=is_loop(hyp, g["stop_type"]), tokens=g["tokens"])
        gl = LANGUAGES[iso].glotlid
        rec["lid"] = lid(hyp) if hyp else None
        rec["wrong_lang"] = (rec["lid"] not in gl) if gl else None
        if flags is not None: rec["pair_seen"] = flags.get(f"{iso}:{r['id']}")   # None = flag not available
        per[iso].append(rec)
    with open(os.path.join(OUT, f"per_sentence{sfx}.jsonl"), "w", encoding="utf-8") as f:
        for iso in per:
            for rec in per[iso]: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 100, size=(B, 100))
    res = {}
    for iso in SUPPORTED_RL:
        rs = per.get(iso, [])
        if not rs: continue
        n = len(rs)
        d = dict(iso=iso, name=LANGUAGES[iso].name, tier=rs[0]["tier"], n=n,
                 finished=np.mean([r["finished"] for r in rs]), loop=np.mean([r["loop"] for r in rs]), tag_ok=np.mean([r["tag_ok"] for r in rs]),
                 mean_tokens=np.mean([r["tokens"] for r in rs]))
        gl = LANGUAGES[iso].glotlid
        d["wrong_lang"] = np.mean([bool(r["wrong_lang"]) or not r["hyp"] for r in rs]) if gl else None
        d["empty"] = np.mean([not r["hyp"] for r in rs])
        if rs[0]["reference"]:
            hyps = [r["hyp"] for r in rs]; refs = [nfc(r["reference"]) for r in rs]
            d["chrf"] = chrf.corpus_score(hyps, [refs]).score
            d["spbleu"] = bleu.corpus_score(hyps, [refs]).score
            st = np.array(chrf._extract_corpus_statistics(hyps, [refs]))
            d["_boot"] = np.array([chrf._compute_score_from_stats(st[ix].sum(0)).score for ix in idx])
            assert abs(chrf._compute_score_from_stats(st.sum(0)).score - d["chrf"]) < 1e-6
            d["ci_lo"], d["ci_hi"] = np.percentile(d["_boot"], [2.5, 97.5])
            if flags is not None and all(r.get("pair_seen") is not None for r in rs):
                d["pair_seen_rate"] = float(np.mean([r["pair_seen"] for r in rs]))
                cl = [i for i, r in enumerate(rs) if not r["pair_seen"]]
                d["n_nopair"] = len(cl)
                d["chrf_nopair"] = chrf.corpus_score([hyps[i] for i in cl], [[refs[i] for i in cl]]).score if len(cl) >= 10 else None
        res[iso] = d
    A = sorted([i for i in res if res[i]["tier"] == "A"], key=lambda i: res[i]["chrf"])
    c20, c21 = res[A[N_TOP - 1]]["chrf"], res[A[N_TOP]]["chrf"]
    mid = (c20 + c21) / 2
    boot = np.stack([res[i]["_boot"] for i in A])               # (45, B), paired over sentences
    rank = boot.argsort(0).argsort(0)                          # 0 = lowest
    for k, i in enumerate(A):
        res[i]["p_bottom20"] = float((rank[k] < N_TOP).mean())
        res[i]["rank_asc"] = k + 1
    sft = {r["iso"]: float(r["sft_share_pct"]) for r in csv.DictReader(open(os.path.join(ROOT, "reports/data/language_token_proportions_2026-10-25.csv")))}
    dev = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "reports/data/final2_eval_2026-10-23/per_language_bench.csv"))):
        if r["benchmark"] == "flores_plus" and r["subset"] == "all": dev[r["iso"]] = float(r["final2_chrf"])
    for i, d in res.items():
        reasons = []
        if d["tier"] == "A" and d["rank_asc"] <= N_TOP: reasons.append("S1 bottom-20 chrF++")
        if d["tier"] == "B" and d["chrf"] < c20: reasons.append("S2 below c20")
        if d["finished"] < FIN_MIN: reasons.append("S3 finished<95%")
        if d["loop"] > LOOP_MAX: reasons.append("S3 loop>5%")
        if d["wrong_lang"] is not None and d["wrong_lang"] > WRONG_MAX: reasons.append("S3 wrong-lang>10%")
        d["weak"] = bool(reasons); d["reasons"] = "; ".join(reasons)
        d["near_cut"] = bool(d["tier"] == "A" and d["ci_lo"] <= mid <= d["ci_hi"])
        d["sft_share_pct"] = sft.get(i); d["devtest_chrf_posthoc"] = dev.get(i)
    cols = ["iso", "name", "tier", "n", "rank_asc", "chrf", "ci_lo", "ci_hi", "p_bottom20", "near_cut", "spbleu", "finished", "loop", "wrong_lang", "tag_ok", "empty", "mean_tokens",
            "weak", "reasons", "sft_share_pct", "devtest_chrf_posthoc"] + (["pair_seen_rate", "n_nopair", "chrf_nopair"] if flags is not None else [])
    order = A + sorted([i for i in res if res[i]["tier"] == "B"], key=lambda i: res[i]["chrf"]) + sorted([i for i in res if res[i]["tier"] == "C"], key=lambda i: (res[i]["wrong_lang"] is None, -(res[i]["wrong_lang"] or 0)))
    with open(os.path.join(OUT, f"per_language{sfx}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for i in order:
            w.writerow([("" if res[i].get(c) is None else (round(float(res[i][c]), 4) if isinstance(res[i].get(c), (float, np.floating)) else res[i][c])) for c in cols])
    weak = [i for i in order if res[i]["weak"]]
    json.dump(dict(n_languages_scored=len(res), c20=c20, c21=c21, midpoint=mid, weak=weak, n_weak=len(weak),
                   tierA_bottom20=A[:N_TOP], tierB={i: res[i]["chrf"] for i in res if res[i]["tier"] == "B"},
                   guard_only=[i for i in weak if "S1" not in res[i]["reasons"] and "S2" not in res[i]["reasons"]],
                   unmeasured_not_weak=[i for i in res if res[i]["tier"] == "C" and not res[i]["weak"]],
                   near_cut=[i for i in A if res[i]["near_cut"]]), open(os.path.join(OUT, f"weak_list{sfx}.json"), "w"), indent=1)
    print("scored", len(res), "c20", round(c20, 2), "c21", round(c21, 2), "weak", len(weak)); print(" ".join(weak))


main()
