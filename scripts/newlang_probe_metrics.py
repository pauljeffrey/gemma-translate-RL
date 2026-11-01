#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Metrics of the zero-shot probe (D-47). Reads the raw outputs of newlang_probe.py and writes ONE row per completion and ONE row per prompt group.

    python scripts/newlang_probe_metrics.py --in-dir out/rl2c --out-dir out/metrics [--calibration calibration/shaping_flores_dev_v6.json] [--lid-model model_v3.bin]

Per completion (samples.csv): the translation text, the tag it wrote, finished, GlotLID top-1 / probability / "in the accepted label set", chrF++ against the human reference (sacrebleu
CHRF(word_order=2)), chrF++ of the same text against a DIFFERENT sentence's reference (the noise floor of unrelated text), copy-of-source, degenerate / self-repetitive (rl/guards.py),
length ratio, mean token log-prob, `valid` (= the hard checks of the reward: right tag or prefilled, finished, non-empty, not a copy, not degenerate, length ratio in [0.2, 5],
GlotLID accepts), and `consensus` (mean chrF++ against the other valid members of the prompt group, rl/consensus.py, groups of the greedy completion + the samples).
Per prompt (groups.csv): n_valid, GlotLID pass rate, consensus level, diversity, length CV, mean log-prob, reference chrF++ of the valid members, best / mean / greedy.

The accepted GlotLID labels and their minimum text length: trained languages from the production calibration (`lid` blocks), new languages from newlang_registry.py (min 30 characters).
Nothing here needs a GPU.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import newlang_registry as R  # noqa: E402

LID_MIN = 15


def accepted_for(key, cal):
    if key in R.NEW:
        return tuple(R.NEW[key]["glot"]), 30
    e = (cal.get(key) or {}).get("lid") or {}
    labels = tuple(e.get("labels") or R.YARD_GLOT_FALLBACK.get(key, ()))
    return labels, int(e.get("min_chars") or 30)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--lid-model", default=None)
    ap.add_argument("--glob", default="*.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from sacrebleu.metrics import CHRF

    from gemma_mt.rl import consensus as cons
    from gemma_mt.rl import guards
    from gemma_mt.rl.lid import GlotLIDClassifier
    from gemma_mt.rl.text import nfc

    chrf = CHRF(word_order=2)
    lid = GlotLIDClassifier(path=a.lid_model)
    lid._load()
    cal = json.load(open(a.calibration, encoding="utf-8")) if a.calibration else {}
    os.makedirs(a.out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    samples_rows, group_rows, summary = [], [], []

    for path in sorted(glob.glob(os.path.join(a.in_dir, a.glob))):
        with open(path, encoding="utf-8") as f:
            meta = json.loads(f.readline())["meta"]
            rows = [json.loads(l) for l in f if l.strip()]
        key, sname, cond, label = meta["key"], meta["set"], meta["cond"], meta["label"]
        english, refs = meta["english"], meta["reference"]
        labels, min_chars = accepted_for(key, cal)
        n = len(english)
        # ---- per completion
        texts = [nfc(r["text"]) for r in rows]
        tops = lid.top(texts, 3)
        recs = []
        placebo_j = {}
        for pi0 in range(n):                                            # the "unrelated sentence" of every prompt, fixed: a placebo reference for chrF++
            j0 = rng.randrange(n)
            while n > 1 and j0 == pi0:
                j0 = rng.randrange(n)
            placebo_j[pi0] = j0
        for r, t, top in zip(rows, texts, tops):
            pi = r["pi"]
            src, ref = english[pi], refs[pi]
            lid1, p1 = top[0]
            p_acc = max([p for lb, p in top if lb in labels] or [0.0])
            checked = len(t) >= max(LID_MIN, min_chars)
            lid_in = (lid1 in labels) if checked else True
            tag = r["tag"]
            if r.get("prefilled"):
                tag_ok = True
            elif meta["cond"] in ("trained",):
                tag_ok = tag == key
            elif meta["cond"] == "base":
                tag_ok = True
            else:                                                       # free / control: the tag the instruction asks for is that of the named language
                tag_ok = tag == (meta.get("tag") or key)
            empty = not t.strip()
            copy = guards.is_copy_of_source(src, t) if not empty else False
            deg = (guards.is_repetitive(t) if not empty else False)
            selfrep = guards.is_self_repetitive(t) if not empty else False
            lr = len(t) / max(1, len(src))
            lp = r["logp"]
            body = lp[1:] if (len(lp) > 1 and tag and not r.get("prefilled")) else lp
            c = chrf.sentence_score(t, [ref]).score if not empty else 0.0
            c_pl = chrf.sentence_score(t, [refs[placebo_j[pi]]]).score if not empty else 0.0
            valid_nolid = bool(tag_ok and r["finished"] and not empty and not copy and not deg and not selfrep and 0.2 <= lr <= 5.0)
            valid = valid_nolid and lid_in
            recs.append(dict(label=label, key=key, set=sname, cond=cond, kind=r["kind"], pi=pi, k=r["k"], tag=tag or "", tag_ok=int(tag_ok), finished=int(r["finished"]), n_tok=r["n_tok"],
                             empty=int(empty), copy=int(copy), degenerate=int(deg), selfrep=int(selfrep), len_ratio=round(lr, 3), lid1=lid1, lid_p1=round(p1, 3), lid_checked=int(checked),
                             lid_in=int(lid_in), p_accept=round(p_acc, 3), mean_logp=round(sum(lp) / max(1, len(lp)), 4), mean_logp_body=round(sum(body) / max(1, len(body)), 4),
                             chrf=round(c, 3), chrf_placebo=round(c_pl, 3), valid=int(valid), valid_nolid=int(valid_nolid), consensus="", consensus_nolid="", text=t, _prof=None))
        # noise floor: greedy text against another sentence's reference
        for rc in recs:
            if rc["kind"] == "greedy" and not rc["empty"]:
                j = rng.randrange(n)
                while n > 1 and j == rc["pi"]:
                    j = rng.randrange(n)
                rc["chrf_mismatch"] = round(chrf.sentence_score(rc["text"], [refs[j]]).score, 3)
        # ---- per group: consensus among the valid members (greedy + samples of the prompt)
        by_pi = {}
        for rc in recs:
            by_pi.setdefault(rc["pi"], []).append(rc)
        for pi, grp in sorted(by_pi.items()):
            profs = [cons.Profile(rc["text"]) if rc["valid"] else None for rc in grp]
            vals = cons.group_consensus(profs, 3)
            for rc, v in zip(grp, vals):
                rc["consensus"] = "" if v is None else round(v, 3)
            profs2 = [cons.Profile(rc["text"]) if rc["valid_nolid"] else None for rc in grp]
            for rc, v in zip(grp, cons.group_consensus(profs2, 3)):
                rc["consensus_nolid"] = "" if v is None else round(v, 3)
            v_ = [rc for rc in grp if rc["valid"]]
            texts_v = [rc["text"] for rc in v_]
            lens = [len(t) for t in texts_v]
            distinct = len(set(" ".join(t.split()).lower() for t in texts_v)) / max(1, len(texts_v))
            cv = (statistics.pstdev(lens) / statistics.mean(lens)) if len(lens) > 1 and statistics.mean(lens) > 0 else 0.0
            cs = [rc["chrf"] for rc in v_]
            gr = [rc for rc in grp if rc["kind"] == "greedy"]
            cvals = [rc["consensus"] for rc in v_ if rc["consensus"] != ""]
            checked = [rc for rc in grp if rc["lid_checked"]]
            cn = [float(rc["consensus_nolid"]) for rc in grp if rc["consensus_nolid"] != ""]
            group_rows.append(dict(label=label, key=key, set=sname, cond=cond, pi=pi, n=len(grp), n_valid=len(v_), n_valid_nolid=sum(rc["valid_nolid"] for rc in grp),
                                   cons_mean_nolid=round(statistics.mean(cn), 3) if cn else "", n_samples=sum(rc["kind"] == "sample" for rc in grp),
                                   lid_pass=round(sum(rc["lid_in"] for rc in checked) / max(1, len(checked)), 3) if checked else 1.0,
                                   lid_p_accept=round(statistics.mean(rc["p_accept"] for rc in grp), 3),
                                   finished=round(statistics.mean(rc["finished"] for rc in grp), 3),
                                   cons_mean=round(statistics.mean(cvals), 3) if cvals else "", distinct=round(distinct, 3), len_cv=round(cv, 3),
                                   logp_mean=round(statistics.mean(rc["mean_logp_body"] for rc in grp), 4),
                                   logp_valid=round(statistics.mean(rc["mean_logp_body"] for rc in v_), 4) if v_ else "",
                                   chrf_valid_mean=round(statistics.mean(cs), 3) if cs else "", chrf_valid_best=round(max(cs), 3) if cs else "",
                                   chrf_all_mean=round(statistics.mean(rc["chrf"] for rc in grp), 3), chrf_greedy=gr[0]["chrf"] if gr else "",
                                   copy_rate=round(statistics.mean(rc["copy"] for rc in grp), 3), deg_rate=round(statistics.mean(max(rc["degenerate"], rc["selfrep"]) for rc in grp), 3)))
        # ---- summary of this file
        def m(xs):
            xs = list(xs)
            return round(statistics.mean(xs), 4) if xs else None

        gre = [rc for rc in recs if rc["kind"] == "greedy"]
        smp = [rc for rc in recs if rc["kind"] == "sample"]
        tags = {}
        for rc in gre:
            tags[rc["tag"] or "(none)"] = tags.get(rc["tag"] or "(none)", 0) + 1
        l1 = {}
        for rc in gre:
            l1[rc["lid1"]] = l1.get(rc["lid1"], 0) + 1
        summary.append(dict(label=label, key=key, set=sname, cond=cond, n=n, name=meta["name"], tag=meta["tag"], seconds=round(meta["seconds"]),
                            greedy_chrf=m(rc["chrf"] for rc in gre), sample_chrf=m(rc["chrf"] for rc in smp), greedy_chrf_valid=m(rc["chrf"] for rc in gre if rc["valid"]),
                            mismatch_floor=m(rc.get("chrf_mismatch") for rc in gre if rc.get("chrf_mismatch") is not None),
                            greedy_finished=m(rc["finished"] for rc in gre), sample_finished=m(rc["finished"] for rc in smp),
                            greedy_lid_right=m(rc["lid_in"] for rc in gre if rc["lid_checked"]), sample_lid_right=m(rc["lid_in"] for rc in smp if rc["lid_checked"]),
                            lid_checked_share=m(rc["lid_checked"] for rc in recs), greedy_copy=m(rc["copy"] for rc in gre), sample_copy=m(rc["copy"] for rc in smp),
                            greedy_degenerate=m(max(rc["degenerate"], rc["selfrep"]) for rc in gre), sample_degenerate=m(max(rc["degenerate"], rc["selfrep"]) for rc in smp),
                            greedy_valid=m(rc["valid"] for rc in gre), sample_valid=m(rc["valid"] for rc in smp), greedy_tag_ok=m(rc["tag_ok"] for rc in gre),
                            tags=dict(sorted(tags.items(), key=lambda kv: -kv[1])[:4]), greedy_lid_top=dict(sorted(l1.items(), key=lambda kv: -kv[1])[:4]),
                            sample_logp=m(rc["mean_logp_body"] for rc in smp), greedy_logp=m(rc["mean_logp_body"] for rc in gre),
                            sample_len_ratio=m(rc["len_ratio"] for rc in smp), greedy_len_ratio=m(rc["len_ratio"] for rc in gre),
                            consensus_mean=m(float(rc["consensus"]) for rc in recs if rc["consensus"] != "")))
        samples_rows += recs
        print("metrics", label, key, sname, cond, summary[-1]["greedy_chrf"], summary[-1]["sample_lid_right"], flush=True)

    cols = [c for c in (samples_rows[0].keys() if samples_rows else []) if c != "_prof"]
    cols = cols + (["chrf_mismatch"] if "chrf_mismatch" not in cols else [])
    with open(os.path.join(a.out_dir, "samples.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(samples_rows)
    if group_rows:
        with open(os.path.join(a.out_dir, "groups.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
            w.writeheader()
            w.writerows(group_rows)
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
