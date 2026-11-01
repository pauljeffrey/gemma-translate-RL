#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mixture v2, component B step 3: decode the sentence-level generations, judge every candidate, pick one per sentence, assemble and score the documents.

    python scripts/mix2_longdoc_score.py --work /workspace/jobs/r11/work/ld

Per sentence unit (language, sentence piece) there are up to 4 candidates: greedy (the R6 recipe) + 3 samples at T = 0.7 (the `sentfix` repair candidates, here for EVERY
unit). A candidate is valid when it finished (<end_of_turn> within the cap), the language tag was there (status ok), it is no copy of the source, its length ratio is
plausible, and the GlotLID guard accepts it where it is on. Pick:
  * QE languages (the QE ensemble of D-55: AfriCOMET-QE + SSA-COMET-MTL; 54 languages): the valid candidate with the highest ensemble reward (ties -> greedy);
  * consensus-only languages (23 languages with no QE): the valid candidate with the highest mean chrF++ against the OTHER candidates (MBR; ties -> greedy);
    its confidence is that mean chrF++ (all 4 identical = 100).
Documents: sentences joined with single spaces (the sources of the RL pools have no paragraph breaks). Written for EVERY document, with its statistics, to
OUT/ld_scored.jsonl, so the thresholds can be chosen afterwards (mix2_longdoc_select.py): mean / p25 / min of the sentence rewards (source-character weighted mean), greedy-only
statistics for comparison, coverage (units without any valid candidate), loops, whole-document LID, diacritic deficit, repetition, length ratio, copied sentences.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mix2_common as C  # noqa: E402

STOP = [106, 1]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_gz(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for l in f:
            yield json.loads(l)


# ---- fast chrF++ consensus (n-gram caches), the function of /workspace/jobs/a2/assemble.py
def _grams(text):
    t = " ".join(text.split())
    chars = t.replace(" ", "")
    cg = [collections.Counter(chars[i:i + n] for i in range(len(chars) - n + 1)) for n in range(1, 7)]
    w = t.split()
    wg = [collections.Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1)) for n in range(1, 3)]
    return cg + wg


def _chrf(h, r, beta=2.0):
    P = R = 0.0
    for a, b in zip(h, r):
        ta, tb = sum(a.values()), sum(b.values())
        if ta == 0 or tb == 0:
            continue
        m = sum((a & b).values())
        P += m / ta
        R += m / tb
    P /= 8
    R /= 8
    if P + R == 0:
        return 0.0
    return 100 * (1 + beta ** 2) * P * R / (beta ** 2 * P + R)


def mbr(args):
    """texts (valid candidates in order greedy, s1, s2, s3 - may repeat) -> (index, confidence = mean chrF++ of the pick against the other candidates)."""
    rid, texts = args
    if len(texts) == 1:
        return rid, (0, None)
    uniq = {}
    for t in texts:
        uniq.setdefault(t, _grams(t))
    best, bs = 0, -1.0
    cache = {}
    for i, t in enumerate(texts):
        s = []
        for j, u in enumerate(texts):
            if i == j:
                continue
            if t == u:
                s.append(100.0)
                continue
            k = (t, u)
            if k not in cache:
                cache[k] = (_chrf(uniq[t], uniq[u]) + _chrf(uniq[u], uniq[t])) / 2
            s.append(cache[k])
        m = sum(s) / len(s)
        if m > bs + 1e-9:
            best, bs = i, m
    return rid, (best, bs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", default="/workspace/sft_build/runs/final_1433")
    ap.add_argument("--limit-docs", type=int, default=0)
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from gemma_mt.eval.longdoc import repetition_loop
    from gemma_mt.rl.guards import is_copy_of_source
    from gemma_mt.rl.text import decode_completion, translation_from

    tok = AutoTokenizer.from_pretrained(a.model)
    J = C.Judge()
    docs = [json.loads(l) for l in open(f"{a.work}/docs.jsonl", encoding="utf-8")]
    if a.limit_docs:
        docs = docs[:a.limit_docs]
    reqs = {}
    need = set()
    for d in docs:
        for s in d["sents"]:
            need.add((d["iso"], s))
    from gemma_mt.eval.longdoc import split_long_sentence

    req_by_text = {}
    for l in open(f"{a.work}/req_sent.jsonl", encoding="utf-8"):
        r = json.loads(l)
        req_by_text[(r["iso"], r["text"])] = r["rid"]
        reqs[r["rid"]] = {"iso": r["iso"], "text": r["text"]}
    gen = {"greedy": {}, "t07x3": {}}
    for job in gen:
        for r in read_gz(f"{a.work}/gen/{job}.jsonl.gz"):
            gen[job][r["rid"]] = r["outs"]
    log(f"requests {len(reqs)}, generations: " + ", ".join(f"{k} {len(v)}" for k, v in gen.items()))

    # ---------------------------------------------------------------- candidates per unit
    cands = {}                                        # rid -> list of dicts (source order: greedy, s1, s2, s3)
    stat = collections.Counter()
    for rid, q in reqs.items():
        iso, src = q["iso"], q["text"]
        outs = list(gen["greedy"].get(rid, [])) + list(gen["t07x3"].get(rid, []))
        cs = []
        for k, o in enumerate(outs):
            dd = decode_completion(tok, o["ids"], STOP)
            t = translation_from(dd, iso)
            c = {"k": k, "text": t.text, "finished": bool(dd.finished), "status": t.status, "ntok": len(o["ids"])}
            why = None
            if t.status != "ok":
                why = "status:" + str(t.status)
            elif not dd.finished:
                why = "unfinished"
            elif dd.dangling or dd.post_stop_tokens:
                why = "dangling"
            elif not t.text.strip():
                why = "empty"
            elif is_copy_of_source(src, t.text, 0.9):
                why = "copy"
            elif not J.len_ok(iso, src, t.text, slack=1.8):
                why = "length"
            elif J.repeat_bad(iso, src, t.text, extra=0.05):
                why = "repeat"
            c["bad"] = why
            stat[("greedy" if k == 0 else "sample", why or "valid")] += 1
            cs.append(c)
        cands[rid] = cs
    log("candidate validity: " + json.dumps({f"{a_}/{b_}": v for (a_, b_), v in sorted(stat.items(), key=lambda kv: -kv[1])[:14]}))

    # sentence-level LID of the otherwise valid candidates
    by_lang = collections.defaultdict(list)
    for rid, cs in cands.items():
        for c in cs:
            if c["bad"] is None:
                by_lang[reqs[rid]["iso"]].append((rid, c))
    nlid = 0
    for iso, lst in by_lang.items():
        got = J.lid_ok_many(iso, [c["text"] for _, c in lst])
        for (rid, c), g in zip(lst, got):
            if g is False and len(c["text"]) >= 40:
                c["bad"] = "lid"
                nlid += 1
    log(f"sentence-level LID rejected {nlid} candidates")

    # ---------------------------------------------------------------- QE of the valid candidates (QE languages)
    pairs, owner = [], []
    seen = {}
    for rid, cs in cands.items():
        iso, src = reqs[rid]["iso"], reqs[rid]["text"]
        if not J.has_qe(iso):
            continue
        for c in cs:
            if c["bad"] is None:
                key = (iso, src, c["text"])
                if key not in seen:
                    seen[key] = len(pairs)
                    pairs.append(key)
                c["pair"] = seen[key]
    log(f"QE: {len(pairs)} unique (source, candidate) pairs in {len({p[0] for p in pairs})} languages")
    t0 = time.time()
    raw = J.qe_raw([p[1] for p in pairs], [p[2] for p in pairs], [p[0] for p in pairs])
    log(f"QE done in {time.time() - t0:.0f}s")
    shaped = [J.shaped(c1, c2, p[0]) for (c1, c2), p in zip(raw, pairs)]
    for cs in cands.values():
        for c in cs:
            if "pair" in c:
                c["c1"], c["c2"] = raw[c["pair"]]
                c["r1"], c["r2"], c["r"] = shaped[c["pair"]]

    # ---------------------------------------------------------------- MBR for the consensus-only languages
    todo = []
    for rid, cs in cands.items():
        iso = reqs[rid]["iso"]
        if J.has_qe(iso):
            continue
        valid = [c for c in cs if c["bad"] is None]
        if valid:
            todo.append((rid, [c["text"] for c in valid]))
    log(f"MBR over {len(todo)} units")
    with mp.Pool(8) as pool:
        mb = dict(pool.imap_unordered(mbr, todo, chunksize=64))
    # ---------------------------------------------------------------- pick
    pick = {}
    for rid, cs in cands.items():
        iso = reqs[rid]["iso"]
        valid = [c for c in cs if c["bad"] is None]
        if not valid:
            pick[rid] = None
            continue
        if J.has_qe(iso):
            best = max(valid, key=lambda c: (c["r"], -c["k"]))
            g = cs[0] if cs and cs[0]["bad"] is None else None
            pick[rid] = {"text": best["text"], "r": best["r"], "k": best["k"], "r_greedy": (g["r"] if g else None), "conf": None, "ntok": best["ntok"]}
        else:
            bi, bc = mb[rid]
            c = valid[bi]
            pick[rid] = {"text": c["text"], "r": None, "k": c["k"], "r_greedy": None, "conf": bc, "ntok": c["ntok"]}

    # ---------------------------------------------------------------- documents
    out = []
    for d in docs:
        iso = d["iso"]
        qe = J.has_qe(iso)
        sent_out, r_list, w_list, g_list, conf_list, ks = [], [], [], [], [], collections.Counter()
        missing = copied = 0
        for s in d["sents"]:
            pieces = split_long_sentence(s, 700)
            texts, rs, rg, cf, ok = [], [], [], [], True
            for p in pieces:
                rid = req_by_text[(iso, p)]
                pk = pick.get(rid)
                if pk is None:
                    ok = False
                    break
                texts.append(pk["text"])
                ks[pk["k"] == 0] += 1
                if qe:
                    rs.append(pk["r"])
                    rg.append(pk["r_greedy"] if pk["r_greedy"] is not None else -2.0)
                else:
                    cf.append(pk["conf"] if pk["conf"] is not None else 0.0)
            if not ok:
                missing += 1
                continue
            sent_out.append(" ".join(texts))
            w = float(max(len(s), 1))
            w_list.append(w)
            if qe:
                r_list.append(sum(rs) / len(rs))
                g_list.append(sum(rg) / len(rg))
            else:
                conf_list.append(sum(cf) / len(cf))
        row = {k: d[k] for k in ("iso", "bin", "domain", "doc_key", "source_dataset", "licence", "english", "prompt_tokens")}
        if "field" in d:
            row["field"] = d["field"]
        row.update({"n_sents": len(d["sents"]), "missing_units": missing, "mode": "qe" if qe else "consensus"})
        if missing or not sent_out:
            row["target"] = None
            out.append(row)
            continue
        text = " ".join(sent_out)
        row["target"] = text
        row["frac_greedy"] = ks[True] / max(sum(ks.values()), 1)

        def wmean(v):
            return sum(x * w for x, w in zip(v, w_list)) / sum(w_list)

        def p25(v):
            sv = sorted(v)
            return sv[int(0.25 * (len(sv) - 1))]

        if qe:
            row.update({"mean_r": wmean(r_list), "p25_r": p25(r_list), "min_r": min(r_list), "mean_r_greedy": wmean(g_list), "p25_r_greedy": p25(g_list)})
        else:
            row.update({"mean_conf": wmean(conf_list), "p25_conf": p25(conf_list), "min_conf": min(conf_list)})
        lp = repetition_loop(text)
        row["loop"] = bool(lp.get("loop")) if isinstance(lp, dict) else bool(lp)
        row["repeat"] = J.repeat_bad(iso, d["english"], text) or None
        row["lid_doc"] = J.lid_ok(iso, text)
        row["diacritic_deficit"] = J.diacritic_deficit(iso, d["english"], text)
        row["len_ratio"] = len(text) / max(len(d["english"]), 1)
        row["len_ok"] = J.len_ok(iso, d["english"], text, slack=1.25)
        out.append(row)
    with open(f"{a.work}/ld_scored.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"{len(out)} documents scored, {sum(1 for r in out if r['target'])} complete")
    J.close()


if __name__ == "__main__":
    main()
