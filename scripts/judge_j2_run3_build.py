"""Run 3 build (no API calls): long documents (LongCtx-MT core, real documents b1024 + b2048), judged by the QE prompt (Figure 9) on ALIGNED UNITS.

    PYTHONPATH=<repo>/gemma_mt python judge_j2_run3_build.py --lc1 <extracted lc1 dir> --pilot-items <scratch j1/work/items.jsonl> --scratch <dir> --out <repo out dir>

Design (owner 2026-10-25): our models translate a whole document in ONE generation; the judge prompt is sentence-level, so the document is never given to
the judge. The output of each system is aligned to the source sentences and the aligned units are judged:
  * whole-document outputs: `plan_document` of gemma_mt/eval/longdoc.py with the language length model of LC1 (the aligner and the omission /
    copy / shortfall rules of the long-document reward v2 and of the LC1 coverage metric: Gale-Church with omission and extra beads).
    A unit = one bead = one or more consecutive source sentences with the output text aligned to them (judged ONCE, however many sampled sentences it
    covers). A source sentence without output (or whose output is a copy of the source, the LC1 rule) is an omission, not a judged unit.
  * wrapper outputs (sentence-by-sentence: final2 sentfix, NLLB-3.3B, TranslatePsy-0.8B): one output per source sentence, judged sentence by sentence;
    empty / copied outputs count as omissions by the same rule.
Sampling (fixed before any API call, seed 20261025): per language 50 source sentences from the b1024 documents and 50 from the b2048 documents (20 + 20
documents), at most 3 per document, only documents that exist for all systems that have that language; identical source positions for all systems.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import random
from pathlib import Path

from gemma_mt.eval.longdoc import make_length_model, plan_document, split_sentences
from gemma_mt.rl.guards import is_copy_of_source

from judge_j2_common import build_qe_prompt

SEED = 20261025
LANGS = "hau yor swh amh ibo kin".split()
BINS = ["b1024", "b2048"]
PER_BIN = 50
MAX_PER_DOC = 3
SYSTEMS = {  # name -> (hyp file, mode)
    "final2_whole": ("final2__whole", "whole"),
    "final2_sentfix": ("final2__sentfix", "sent"),
    "rl_whole": ("rl__whole", "whole"),
    "nllb3b3_sent": ("nllb3b3__sent", "sent"),
    "tpsy08_sent": ("tpsy08__sent", "sent"),
}
SC = {"final2_whole": "final2__whole", "final2_sentfix": "final2__sentfix", "rl_whole": "rl__whole", "nllb3b3_sent": "nllb3b3__sent", "tpsy08_sent": "tpsy08__sent"}


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def length_models(docs, items, nllb):
    """Exactly lc_score.length_model(iso, 'core'): corpus length ratio of the NLLB-3.3B pseudo-reference over ALL core items of the language, w_hi = 95th pct."""
    out = {}
    for iso in LANGS:
        a, b = [], []
        for iid, it in items.items():
            if it["iso"] == iso and it["set"] == "core":
                rs = nllb.get(iid)
                if rs:
                    a.extend(docs[it["doc_id"]]["sents"])
                    b.extend(rs["ref_sents"])
        lm = make_length_model(a, b)
        rr = sorted(len(r) / max(len(s), 1) for s, r in zip(a, b) if len(s) >= 10)
        out[iso] = (lm, min(rr[int(0.95 * (len(rr) - 1))], 2.5) if rr else None)
    return out


def align_whole(src_sents, hyp, lm, w_hi):
    """-> (plan, units) ; units = list of dict(start, end(excl), hyp_text, shortfall) for the scored beads, omitted = plan.omitted_idx (omitted + copied)."""
    plan = plan_document(src_sents, [""] * len(src_sents), hyp, lm, w_hi=w_hi)  # the reference text is not used by the plan itself (only echoed in segments)
    omitted = set(plan.omitted_idx)
    short = set(plan.short_idx)
    p, units = 0, []
    for a, b, _ in plan.segments:
        while p in omitted:
            p += 1
        k = 1
        while " ".join(src_sents[p:p + k]) != a:
            k += 1
            if p + k > len(src_sents) + 0:
                raise RuntimeError(f"cannot recover the source span of a bead (p={p})")
        units.append(dict(start=p, end=p + k, src=a, hyp=b, shortfall=any(i in short for i in range(p, p + k))))
        p += k
    assert all(i in omitted for i in range(p, len(src_sents))), "uncovered source tail that is not an omission"
    return plan, units


def sample_positions(docs_by_bin, lang):
    """-> list of (bin, doc_id, sent_idx); per bin PER_BIN positions, <= MAX_PER_DOC per document, document quotas and positions seeded."""
    out = []
    for b in BINS:
        ds = sorted(docs_by_bin[b])
        random.Random(f"{SEED}-{lang}-{b}-quota").shuffle(ds)
        base, extra = divmod(PER_BIN, len(ds))
        assert base + (1 if extra else 0) <= MAX_PER_DOC, (base, extra)
        for k, d in enumerate(ds):
            q = base + (1 if k < extra else 0)
            n = docs_by_bin[b][d]
            perm = random.Random(f"{SEED}-pos-{d}").sample(range(n), n)
            out += [(b, d, i) for i in sorted(perm[:q])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc1", required=True)
    ap.add_argument("--pilot-items", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lc, scratch, out = Path(a.lc1), Path(a.scratch), Path(a.out)
    docs = {d["doc_id"]: d for d in rd(lc / "work/docs.jsonl")}
    items = {i["iid"]: i for i in rd(lc / "work/items.jsonl")}
    nllb = {}
    for fn in ("refs_nllb_core.jsonl", "refs_nllb_sent.jsonl"):
        for r in rd(lc / "work" / fn):
            nllb[r["iid"]] = r
    LM = length_models(docs, items, nllb)
    demos = {}
    for l in open(a.pilot_items, encoding="utf-8"):
        it = json.loads(l)
        demos.setdefault(it["lp"], it["demos"])
    hyp = {}
    for s, (f, mode) in SYSTEMS.items():
        hyp[s] = {r["iid"]: r for r in rd(lc / "hyp" / f"{f}.jsonl")}
    flags = {}
    for s, f in SC.items():
        flags[s] = {r["iid"]: r for r in pickle.load(open(lc / "sc" / f"{f}.pkl", "rb"))}

    sample_rows, unit_rows, cover_rows, prompts, check = [], {}, [], [], []
    for lang in LANGS:
        systems = [s for s in SYSTEMS if all(f"core|{d}|{lang}" in hyp[s] for d in [i["doc_id"] for i in items.values() if i["iso"] == lang and i["set"] == "core" and i["bin"] in BINS])]
        skipped = [s for s in SYSTEMS if s not in systems]
        docs_by_bin = {b: {} for b in BINS}
        for it in items.values():
            if it["iso"] == lang and it["set"] == "core" and it["bin"] in BINS and all(it["iid"] in hyp[s] for s in systems):
                docs_by_bin[it["bin"]][it["doc_id"]] = len(docs[it["doc_id"]]["sents"])
        print(lang, "systems", systems, "skipped", skipped, "docs", {b: len(v) for b, v in docs_by_bin.items()})
        pos = sample_positions(docs_by_bin, lang)
        for b, d, i in pos:
            sample_rows.append(dict(lang=lang, bin=b, doc_id=d, sent_idx=i, n_sents_doc=docs_by_bin[b][d], src_chars=len(docs[d]["sents"][i]), systems="|".join(systems)))
        by_doc = {}
        for b, d, i in pos:
            by_doc.setdefault((b, d), []).append(i)
        lm, w_hi = LM[lang]
        dm = demos[f"eng-{lang}"]
        for s in systems:
            mode = SYSTEMS[s][1]
            for (b, d), idxs in by_doc.items():
                iid = f"core|{d}|{lang}"
                src = docs[d]["sents"]
                fl = flags[s][iid]
                doc_flag = dict(finished=bool(fl["finished"]), loop=bool(fl.get("loop")), loop_tail=bool(fl.get("loop_tail")))
                if mode == "whole":
                    plan, units = align_whole(src, hyp[s][iid]["hyp"], lm, w_hi)
                    # replication check against the stored LC1 score row
                    check.append((s, iid, len(plan.omitted_idx), int(fl["omitted"]) + int(fl["copied"]), plan.n_hyp, int(fl["n_hyp_sents"])))
                    span = {}
                    for u in units:
                        for k in range(u["start"], u["end"]):
                            span[k] = u
                    omitted = set(plan.omitted_idx)
                else:
                    hs = hyp[s][iid]["hyp_sents"]
                    assert len(hs) == len(src)
                    units, span, omitted = [], {}, set()
                    for k in range(len(src)):
                        if not hs[k].strip() or is_copy_of_source(src[k], hs[k], 0.9):
                            omitted.add(k)
                        else:
                            u = dict(start=k, end=k + 1, src=src[k], hyp=hs[k], shortfall=False)
                            units.append(u)
                            span[k] = u
                for i in idxs:
                    if i in omitted or i not in span:
                        cover_rows.append(dict(system=s, lang=lang, bin=b, doc_id=d, sent_idx=i, covered=0, unit_id="", why=("empty_or_copy" if mode == "sent" else "omitted_or_copied"),
                                               block_sents=0, shortfall=0, **{f"doc_{k}": int(v) for k, v in doc_flag.items()}))
                        continue
                    u = span[i]
                    uid = f"lc:{s}:{lang}:{d}:{u['start']}-{u['end']}"
                    cover_rows.append(dict(system=s, lang=lang, bin=b, doc_id=d, sent_idx=i, covered=1, unit_id=uid, why="", block_sents=u["end"] - u["start"],
                                           shortfall=int(u["shortfall"]), **{f"doc_{k}": int(v) for k, v in doc_flag.items()}))
                    if uid not in unit_rows:
                        unit_rows[uid] = dict(id=uid, system=s, lang=lang, bin=b, doc_id=d, start=u["start"], end=u["end"], block_sents=u["end"] - u["start"], src_chars=len(u["src"]),
                                              hyp_chars=len(u["hyp"]), shortfall=int(u["shortfall"]), **{f"doc_{k}": int(v) for k, v in doc_flag.items()})
                        prompts.append(dict(id=uid, prompt=build_qe_prompt(dm, u["src"], u["hyp"])))
    # replication check
    n_bad = sum(1 for c in check if c[2] != c[3] or c[4] != c[5])
    print(f"replication check vs LC1 stored rows (whole mode): {len(check)} docs, omitted/copied mismatches or n_hyp mismatches: {n_bad}")
    assert n_bad == 0, [c for c in check if c[2] != c[3] or c[4] != c[5]][:5]
    with open(scratch / "items_run3.jsonl", "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with gzip.open(out / "prompts_run3_longdoc_qe.jsonl.gz", "wt", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    def dump(name, rows):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dump("run3_sampled_sentences.csv", sample_rows)
    dump("run3_coverage_by_sampled_sentence.csv", cover_rows)
    dump("run3_units.csv", list(unit_rows.values()))
    for sysn, lang in (("final2_whole", "hau"), ("final2_sentfix", "hau"), ("nllb3b3_sent", "yor")):
        ex = next(p for p in prompts if p["id"].startswith(f"lc:{sysn}:{lang}:"))
        (out / f"prompt_example_run3_{sysn}_{lang}.txt").write_text(ex["prompt"], encoding="utf-8")
    print("sampled sentences", len(sample_rows), "units (requests)", len(prompts))
    import collections
    c = collections.Counter((r["system"], r["lang"]) for r in unit_rows.values())
    print({f"{k[0]}/{k[1]}": v for k, v in sorted(c.items())})


if __name__ == "__main__":
    main()
