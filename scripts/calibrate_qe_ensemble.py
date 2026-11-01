#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Calibration of the QE ensemble (D-55): shaping_flores_dev_v7.json -> shaping_flores_dev_v8.json.

v8 = v7 (every v7 byte-for-byte: AfriCOMET anchors, reliabilities, content modes, lid, ... of the running reward) + per language a `qe2` block with the anchors,
reliabilities and diagnostics of the SECOND model, McGill-NLP/ssa-comet-mtl run REFERENCE-FREE (src + mt), measured with calibrate_shaping.py's semantics, and an `ens`
block (the ensemble's own same-source accuracy / reliability / min-branch shares for the chosen delta).

    # 1. GPU, ~20 min on a shared card (<= 20 GB). Both models on the SAME corruption sets, per-sentence scores kept in --work (resumable)
    python scripts/calibrate_qe_ensemble.py score --work /workspace/jobs/b1/calib --models africomet=masakhane/africomet-qe-stl-1.1 ssa_mtl=McGill-NLP/ssa-comet-mtl
    # 2. CPU: anchors, AUC, same-source accuracy per language per model; pooled priors; delta; the ensemble's reliability -> v8 + a markdown table
    python scripts/calibrate_qe_ensemble.py compose --work ... --v7 calibration/shaping_flores_dev_v7.json --out calibration/shaping_flores_dev_v8.json --delta 0.7

Semantics (calibrate_shaping.calibrate): human = FLORES+ DEV (997 sentences per language; devtest is never touched; efi = the IbomMT linguist translation of the
FLORES+ devtest English, calibration only, exactly as v7; pcm = the bench_dev rows v7 used), fail_floor 0.0 (t_fail = the language's own P90 on mismatched
references), corruptions: mismatch (the reference of another sentence), copy (the English source), half (first half of the words), append (reference + another reference).
`usable` = AUC(human vs mismatch) >= 0.7; reliability = linear in that AUC between 0.65 and 0.90; reliability_same_source = ((mismatch + half-omitted accuracy)/2 - 0.5)/0.4.
Languages without a dev reference keep the POOLED PRIOR (median of the calibrated languages, as D-38 did for AfriCOMET); swc borrows swh. Both are listed in the table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma_mt.rl.reward_shaping import auc, compute_anchors, pair_anchors, reliability_from_auc  # noqa: E402

SETS = ("human", "mismatch", "copy", "half", "append")
MIN_AUC = 0.7
N_MAX = 997


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------------------------------------------------ data
def load_dev(hub_snapshot: str, bench_dev: str, ibom_efi: str, token: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """iso -> {"pairs": [(english, reference)], "source": str}. Same sources as scripts/build_flores_dev_all.py (+ efi as v7 did)."""
    from gemma_mt.languages import LANGUAGES, SUPPORTED_RL

    def hub(code: str) -> List[dict]:
        p = os.path.join(hub_snapshot, "dev", f"{code}.jsonl")
        if os.path.exists(p):
            return read_jsonl(p)
        from huggingface_hub import hf_hub_download

        return read_jsonl(hf_hub_download("openlanguagedata/flores_plus", f"dev/{code}.jsonl", repo_type="dataset", token=token))

    en = {int(r["id"]): r["text"] for r in hub("eng_Latn")}
    bench: Dict[Tuple[str, str], List[dict]] = {}
    if os.path.exists(bench_dev):
        for r in read_jsonl(bench_dev):
            bench.setdefault((r["benchmark"], r["iso"]), []).append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for iso in SUPPORTED_RL:
        code = LANGUAGES[iso].flores
        pairs: List[Tuple[str, str]] = []
        src = None
        if code == "twi_Latn":
            by_id: Dict[int, Dict[int, str]] = {}
            for k, c in enumerate(("twi_Latn_akua1239", "twi_Latn_asan1239")):
                for r in hub(c):
                    by_id.setdefault(int(r["id"]), {})[k] = r["text"]
            for i, d in sorted(by_id.items()):
                t = d.get(i % 2) or d.get(1 - i % 2)
                if t and i in en:
                    pairs.append((en[i], t))
            src = "FLORES+ dev (twi: akua/asan by id parity)"
        elif code:
            try:
                for r in hub(code):
                    if int(r["id"]) in en and r["text"].strip():
                        pairs.append((en[int(r["id"])], r["text"]))
                src = "FLORES+ dev"
            except Exception as e:
                print(f"[dev] {iso} ({code}) not available: {str(e)[:80]}", file=sys.stderr)
        if not pairs and bench.get(("flores_plus_dev", iso)):
            pairs = [(r["english"], r["reference"]) for r in bench[("flores_plus_dev", iso)]]
            src = "bench_dev flores_plus_dev rows"
        if not pairs and bench.get(("mafand_dev", iso)):
            pairs = [(r["english"], r["reference"]) for r in bench[("mafand_dev", iso)]]
            src = "MAFAND dev (bench_dev)"
        if not pairs and iso == "efi" and os.path.exists(ibom_efi):
            pairs = [(r["src"], r["trg"]) for r in read_jsonl(ibom_efi)]
            src = "IbomMT efi (FLORES+ devtest English translated by linguists; calibration only, as v7)"
        pairs = [(nfc(a), nfc(b)) for a, b in pairs if a.strip() and b.strip()]
        if pairs:
            out[iso] = {"pairs": pairs, "source": src}
    return out


def make_sets(pairs: Sequence[Tuple[str, str]], iso: str, n_max: int = N_MAX, seed: str = "b1") -> Dict[str, Any]:
    rs = list(pairs)
    random.Random(f"{seed}-{iso}").shuffle(rs)
    rs = rs[:n_max]
    src = [a for a, _ in rs]
    ref = [b for _, b in rs]
    return {"src": src, "ref": ref,
            "sets": {"human": ref, "mismatch": ref[1:] + ref[:1], "copy": src,
                     "half": [" ".join(r.split()[: max(1, len(r.split()) // 2)]) for r in ref],
                     "append": [r + " " + ref[(k + 1) % len(ref)] for k, r in enumerate(ref)]}}


# ------------------------------------------------------------------------------------------------ statistics (calibrate_shaping.calibrate semantics)
def lang_stats(scores: Dict[str, Sequence[float]], fail_floor: float = 0.0, min_auc: float = MIN_AUC) -> Dict[str, Any]:
    human, mism, copy, half, append = (scores[k] for k in SETS)
    acc = lambda bad: float(sum(h > b for h, b in zip(human, bad)) / len(human))
    ss = {"mismatch": acc(mism), "copy": acc(copy), "half_omitted": acc(half), "append_unrelated": acc(append)}
    e: Dict[str, Any] = {
        "n": len(human), "anchors": list(pair_anchors(human, mism, fail_floor)), "fail_floor": fail_floor,
        "human_quantiles_p10_p50_p90": list(compute_anchors(human)),
        "mismatched_p90": float(sorted(mism)[int(0.9 * (len(mism) - 1))]),
        "same_source_accuracy": ss,
        "auc_vs_mismatched": auc(human, mism), "auc_vs_copy": auc(human, copy), "auc_vs_half_omitted": auc(human, half), "auc_vs_append": auc(human, append),
        "mean_human": sum(human) / len(human), "mean_mismatched": sum(mism) / len(mism), "mean_copy": sum(copy) / len(copy),
        "mean_half_omitted": sum(half) / len(half), "mean_append": sum(append) / len(append),
    }
    e["usable"] = bool(e["auc_vs_mismatched"] >= min_auc)
    e["reliability"] = reliability_from_auc(e["auc_vs_mismatched"]) if e["usable"] else 0.0
    hard = (ss["mismatch"] + ss["half_omitted"]) / 2
    e["reliability_same_source"] = float(min(1.0, max(0.0, (hard - 0.5) / 0.4))) if e["usable"] else 0.0
    return e


# ------------------------------------------------------------------------------------------------ step 1: score
def cmd_score(a) -> int:
    import torch

    from gemma_mt.rl.comet import CometScorer

    if a.gpu_mem_gb > 0 and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_mem_gb / total))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
    data = load_dev(a.hub_snapshot, a.bench_dev, a.ibom_efi, os.environ.get("HF_TOKEN"))
    langs = [l for l in sorted(data) if not a.languages or l in a.languages]
    print(f"{len(langs)} languages with human dev references: " + " ".join(f"{l}:{len(data[l]['pairs'])}" for l in langs), flush=True)
    os.makedirs(a.work, exist_ok=True)
    for spec in a.models:
        name, model = spec.split("=", 1)
        os.makedirs(os.path.join(a.work, name), exist_ok=True)
        todo = [l for l in langs if not os.path.exists(os.path.join(a.work, name, l + ".json"))]
        if not todo:
            print(f"{name}: all done", flush=True)
            continue
        sc = CometScorer(model, batch_size=a.batch_size)
        import time

        for l in todo:
            t0 = time.time()
            ms = make_sets(data[l]["pairs"], l, a.max_per_language)
            n = len(ms["src"])
            srcs = [s for k in SETS for s in ms["src"]]
            mts = [m for k in SETS for m in ms["sets"][k]]
            sc_all = sc(srcs, mts)
            sc_by = {k: [round(float(x), 6) for x in sc_all[i * n:(i + 1) * n]] for i, k in enumerate(SETS)}
            json.dump({"iso": l, "model": model, "n": n, "source": data[l]["source"], "scores": sc_by, "joint_split": sc.last_stats.get("joint_split", 0)},
                      open(os.path.join(a.work, name, l + ".json.tmp"), "w"))
            os.replace(os.path.join(a.work, name, l + ".json.tmp"), os.path.join(a.work, name, l + ".json"))
            st = lang_stats(sc_by)
            print(f"{name} {l:4} n={n} {time.time() - t0:5.0f}s anchors={[round(x, 3) for x in st['anchors']]} auc_mismatch={st['auc_vs_mismatched']:.3f} "
                  f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
        del sc
        torch.cuda.empty_cache()
    # the sentences themselves (for the tests of compose / the delta analysis): src + ref per language, same order as the scores
    for l in langs:
        p = os.path.join(a.work, "rows_" + l + ".json")
        if not os.path.exists(p):
            ms = make_sets(data[l]["pairs"], l, a.max_per_language)
            json.dump({"src": ms["src"], "ref": ms["ref"]}, open(p, "w"), ensure_ascii=False)
    return 0


# ------------------------------------------------------------------------------------------------ step 2: compose v8
def load_work(work: str, name: str) -> Dict[str, dict]:
    d = os.path.join(work, name)
    out = {}
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if fn.endswith(".json"):
            r = json.load(open(os.path.join(d, fn)))
            out[r["iso"]] = r
    return out


def combined_sets(c1: Dict[str, Sequence[float]], c2: Dict[str, Sequence[float]], sh1, sh2, lang: str, delta: float) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """Per calibration set: the ensemble reward of every sentence (rule of qe_ensemble.combine_pair on the two shaped rewards) and the share of sentences in the min branch."""
    from gemma_mt.rl.qe_ensemble import MIN, combine_pair

    out, share = {}, {}
    for k in SETS:
        rs, nmin = [], 0
        for a_, b_ in zip(c1[k], c2[k]):
            r, br = combine_pair(sh1(a_, lang), sh2(b_, lang), delta)
            rs.append(r)
            nmin += br == MIN
        out[k] = rs
        share[k] = nmin / len(rs)
    return out, share


def cmd_compose(a) -> int:
    import datetime

    from gemma_mt.rl.reward_shaping import ContentShaper

    v7 = json.load(open(a.v7, encoding="utf-8"))
    s1, s2 = load_work(a.work, a.primary), load_work(a.work, a.secondary)
    langs7 = [k for k in v7 if not k.startswith("_") and isinstance(v7[k], dict)]
    both = sorted(set(s1) & set(s2))
    missing = sorted((set(s1) ^ set(s2)))
    if missing:
        print(f"WARNING: languages scored by only one model: {missing}", file=sys.stderr)
    out = json.loads(json.dumps(v7))                       # v8 = v7 byte-for-byte + the new blocks
    consensus = {k for k in langs7 if (v7[k].get("content") or {}).get("mode") == "consensus"}
    own: Dict[str, Dict[str, Any]] = {}
    recheck = {}
    for iso in both:
        st1 = lang_stats(s1[iso]["scores"])
        st2 = lang_stats(s2[iso]["scores"])
        own[iso] = {"m1": st1, "m2": st2}
        if iso in v7 and v7[iso].get("anchors"):
            recheck[iso] = max(abs(x - y) for x, y in zip(st1["anchors"], v7[iso]["anchors"]))
    # languages whose QE anchors come from a dev calibration of the SECOND model (consensus-mode languages are measured for the report only: they never call a QE model)
    cal_langs = [l for l in sorted(own) if l not in consensus and l in v7 and own[l]["m2"]["usable"]]
    if cal_langs:
        pooled = [float(statistics.median(own[l]["m2"]["anchors"][i] for l in cal_langs)) for i in range(3)]
    else:
        from gemma_mt.rl.reward_shaping import DEFAULT_ANCHORS

        pooled = [float(x) for x in DEFAULT_ANCHORS]
    sh1 = ContentShaper.from_file(a.v7, fail_floor=0.0)
    sh2 = ContentShaper(calibration={l: tuple(own[l]["m2"]["anchors"]) for l in own}, default_anchors=tuple(pooled), fail_floor=0.0)
    unusable2, unusable_both, prior, raised = [], [], [], {}
    for iso in langs7:
        v = out[iso]
        if iso in consensus:
            if iso in own:
                v["qe2_diagnostic"] = {k: own[iso]["m2"][k] for k in ("n", "anchors", "auc_vs_mismatched", "same_source_accuracy", "usable")}
                v["qe2_diagnostic"]["note"] = "consensus-mode language: measured for the report only, no QE model is ever called for it"
            continue
        owner_set = bool(v7[iso].get("weight_reason")) and "owner" in v7[iso]["weight_reason"]
        if iso in own:
            st = dict(own[iso]["m2"])
            st.update({"model": a.secondary_id, "anchors_source": "own FLORES+ dev calibration", "source": s2[iso].get("source"), "joint_split": s2[iso].get("joint_split", 0)})
            if not st["usable"]:
                unusable2.append(iso)
                if not (v7[iso].get("usable", True) and own[iso]["m1"]["usable"]):
                    unusable_both.append(iso)
            v["qe2"] = st
            if st["usable"] and v7[iso].get("usable", True) and v7[iso].get("anchors"):
                comb, share = combined_sets(s1[iso]["scores"], s2[iso]["scores"], sh1, sh2, iso, a.delta)
                ev = lang_stats(comb, fail_floor=-9.0)             # accuracy / AUC on the combined REWARD (anchors of this call are meaningless and dropped)
                hard = (ev["same_source_accuracy"]["mismatch"] + ev["same_source_accuracy"]["half_omitted"]) / 2
                from gemma_mt.rl.reward_shaping import reliability_from_auc

                m_rel = reliability_from_auc(ev["auc_vs_mismatched"])
                m_rss = float(min(1.0, max(0.0, (hard - 0.5) / 0.4)))
                r7, rss7 = float(v7[iso].get("reliability", 1.0)), float(v7[iso].get("reliability_same_source", 1.0))
                b_rel, b_rss = (r7, rss7) if owner_set else (max(r7, m_rel), max(rss7, m_rss))
                if b_rel > r7 + 1e-9 or b_rss > rss7 + 1e-9:
                    raised[iso] = (round(r7, 3), round(b_rel, 3))
                v["ens"] = {"delta": a.delta, "auc_vs_mismatched": ev["auc_vs_mismatched"], "same_source_accuracy": ev["same_source_accuracy"],
                            "reliability_measured": m_rel, "reliability_same_source_measured": m_rss, "reliability": b_rel, "reliability_same_source": b_rss,
                            "reliability_rule": "owner-set weight of v7 kept" if owner_set else "max(v7, measured ensemble)", "min_branch_share": share,
                            "mean_human": sum(comb["human"]) / len(comb["human"])}
        else:
            # no human dev reference: the pooled prior (median of the calibrated languages), as D-38; swc borrows swh. Weight = the owner-set / prior weight of v7 (unchanged).
            borrowed = out["swh"].get("qe2", {}).get("anchors") if "borrowed" in str(v7[iso].get("anchors_source", "")) and "swh" in out else None
            v["qe2"] = {"model": a.secondary_id, "anchors": list(borrowed or pooled), "anchors_source": "borrowed" if borrowed else "pooled_prior",
                        "anchors_source_detail": ("borrowed from swh (qe2)" if borrowed else f"pooled prior: median of the {len(cal_langs)} calibrated languages (t_fail, t_mid, t_great)"),
                        "usable": True, "reliability": v7[iso].get("reliability", 1.0), "reliability_same_source": v7[iso].get("reliability_same_source", 1.0), "n": 0}
            prior.append(iso)
    out["_meta_v8"] = {
        "created": datetime.date.today().isoformat(), "base": os.path.basename(a.v7) + " (every v7 key unchanged; v8 only ADDS `qe2` / `ens` blocks and this meta)",
        "decision": "D-55", "primary_model": "masakhane/africomet-qe-stl-1.1", "secondary_model": a.secondary_id, "delta": a.delta,
        "rule": "r = mean(r1, r2) if |r1 - r2| <= delta else min(r1, r2); r1 = shaped AfriCOMET-QE (v7 anchors), r2 = shaped SSA-COMET-MTL reference-free (qe2 anchors); same reward scale",
        "calibration": "FLORES+ dev (devtest never used), max 997 sentences per language, fail_floor 0.0, corruptions mismatch / copy / half / append, same sentences for both models",
        "pooled_prior_qe2_anchors": pooled, "pooled_prior_from": f"median of {len(cal_langs)} calibrated non-consensus languages",
        "languages_with_own_qe2_calibration": sorted(l for l in own if l not in consensus), "languages_with_pooled_or_borrowed_qe2": sorted(prior),
        "languages_consensus_only": sorted(consensus), "languages_unusable_for_secondary": sorted(unusable2), "languages_unusable_for_both": sorted(unusable_both),
        "weights_raised_vs_v7": raised,
        "primary_recheck_max_abs_anchor_diff_vs_v7": max(recheck.values()) if recheck else None,
    }
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if a.table:
        write_table(a.table, out, own, a.delta)
    print(f"wrote {a.out}: {len(own)} languages measured, {len(prior)} on a pooled/borrowed prior, delta {a.delta}; secondary unusable: {sorted(unusable2)}; "
          f"weights raised: {len(raised)}; max |anchor diff| of the AfriCOMET re-measure vs v7: {max(recheck.values()) if recheck else None:.4f}")
    return 0


def write_table(path: str, v8: Dict[str, Any], own: Dict[str, Dict[str, Any]], delta: float) -> None:
    f3 = lambda x: f"{x:.3f}"
    lines = [f"Per-language calibration (FLORES+ dev, n=997 unless stated; delta {delta}). A = AfriCOMET-QE-STL-1.1, S = SSA-COMET-MTL reference-free; t_fail/t_mid/t_great = reward 0/1/2 anchors; "
             "AUC = human vs mismatched reference; acc = human > corrupted version of the SAME source.", "",
             "| lang | mode | n | A anchors | A AUC | A acc mism/half/copy/append | S anchors | S AUC | S acc mism/half/copy/append | S usable | ens AUC | ens acc mism/half/copy/append | ens min-branch share human/copy | weight v7 -> v8 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for iso in sorted(own):
        m1, m2 = own[iso]["m1"], own[iso]["m2"]
        v = v8.get(iso, {})
        e = v.get("ens") or {}
        acc = lambda m: "/".join(f"{m['same_source_accuracy'][k]:.2f}" for k in ("mismatch", "half_omitted", "copy", "append_unrelated"))
        mode = (v.get("content") or {}).get("mode", "qe")
        w7 = v.get("reliability")
        ens_acc = "/".join("%.2f" % e["same_source_accuracy"][k] for k in ("mismatch", "half_omitted", "copy", "append_unrelated")) if e else "-"
        ens_min = "/".join("%.2f" % e["min_branch_share"][k] for k in ("human", "copy")) if e else "-"
        ens_auc = f3(e["auc_vs_mismatched"]) if e else "-"
        if e:
            wtxt = "%.2f -> %.2f" % (w7, e["reliability"])
        else:
            wtxt = ("%.2f (unchanged)" % w7) if w7 is not None else "-"
        lines.append(f"| {iso} | {mode} | {m1['n']} | {'/'.join(f3(x) for x in m1['anchors'])} | {f3(m1['auc_vs_mismatched'])} | {acc(m1)} | {'/'.join(f3(x) for x in m2['anchors'])} | "
                     f"{f3(m2['auc_vs_mismatched'])} | {acc(m2)} | {'yes' if m2['usable'] else 'NO'} | {ens_auc} | {ens_acc} | {ens_min} | {wtxt} |")
    meta = v8["_meta_v8"]
    lines += ["", f"Pooled prior for languages without a dev reference (qe2 anchors): {[round(x, 3) for x in meta['pooled_prior_qe2_anchors']]} ({meta['pooled_prior_from']}).",
              f"Languages on the pooled / borrowed prior (weight unchanged): {', '.join(meta['languages_with_pooled_or_borrowed_qe2'])}.",
              f"Unusable for SSA-COMET-MTL (AUC < {MIN_AUC}; fall back to AfriCOMET alone): {', '.join(meta['languages_unusable_for_secondary']) or 'none'}.",
              f"Unusable for both: {', '.join(meta['languages_unusable_for_both']) or 'none'}."]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------------------------ step 3: choose delta
def load_scores(path: str) -> List[float]:
    return [float(x) for x in json.load(open(path))]


def cmd_delta(a) -> int:
    """delta from the policy samples (share of chunks in the min branch) and the hack classes (does the min branch fire on copies / partial copies)."""
    from collections import defaultdict

    from gemma_mt.rl.qe_ensemble import QEEnsemble
    from gemma_mt.rl.reward_shaping import ContentShaper

    v8 = a.v8
    sh1 = ContentShaper.from_file(v8, fail_floor=0.0)
    ens = QEEnsemble.from_file(v8, sh1, delta=1.0, fail_floor=0.0)
    m1, m2 = "africomet_qe_stl11.json", "ssa_mtl_qe.json"
    pol = [json.loads(l) for l in open(os.path.join(a.a5, "policy_samples.jsonl"), encoding="utf-8") if l.strip()]
    prow = [json.loads(l) for l in open(os.path.join(a.a5, "policy_rows.jsonl"), encoding="utf-8") if l.strip()]
    assert len(pol) == len(prow), "policy_samples / policy_rows are not aligned"
    p1, p2 = load_scores(os.path.join(a.a5, "policy_scores", m1)), load_scores(os.path.join(a.a5, "policy_scores", m2))
    hk = [json.loads(l) for l in open(os.path.join(a.a5, "hack_rows.jsonl"), encoding="utf-8") if l.strip()]
    h1, h2 = load_scores(os.path.join(a.a5, "hack_scores", m1)), load_scores(os.path.join(a.a5, "hack_scores", m2))
    consensus = set(ens.shaper1.content and [k for k, c in ens.shaper1.content.items() if c.get("mode") == "consensus"] or [])

    def shaped(iso, c1, c2):
        return ens.shaper1(c1, iso), ens.shaper2(c2, iso)

    pol_pairs = [(pol[i]["iso"], *shaped(pol[i]["iso"], p1[i], p2[i])) for i in range(len(pol)) if pol[i]["iso"] not in consensus and ens.plan(pol[i]["iso"]) == "both"
                 and pol[i].get("prefix_status", "ok") == "ok" and prow[i]["mt"].strip()]
    # all samples, also the ones the reward would gate before QE (the gate drops those): statistics are reported for the gate-open ones above
    hack_by = defaultdict(dict)
    for i, r in enumerate(hk):
        if r["iso"] in consensus or ens.plan(r["iso"]) != "both":
            continue
        hack_by[(r["iso"], r["cls"])][r["j"]] = shaped(r["iso"], h1[i], h2[i])
    classes = sorted({c for _, c in hack_by})
    grid = [round(0.05 * k, 2) for k in range(2, 61)]

    def rule(r1, r2, d):
        return 0.5 * (r1 + r2) if abs(r1 - r2) <= d else min(r1, r2)

    rows, per_cls = [], {}
    d_all = sorted(abs(x - y) for _, x, y in pol_pairs)
    for d in grid:
        share = sum(1 for _, x, y in pol_pairs if abs(x - y) > d) / len(pol_pairs)
        cls_share, cls_fail = {}, {}
        for c in classes:
            if c == "human":
                continue
            n_min = n_all = n_fail = 0
            for (iso, cc), rowsj in hack_by.items():
                if cc != c:
                    continue
                for j, (x, y) in rowsj.items():
                    n_all += 1
                    n_min += abs(x - y) > d
                    hum = hack_by[(iso, "human")].get(j)
                    if hum is not None:
                        n_fail += rule(x, y, d) >= rule(hum[0], hum[1], d)
            cls_share[c], cls_fail[c] = n_min / max(n_all, 1), n_fail / max(n_all, 1)
        hs = [abs(x - y) > d for (iso, cc), rj in hack_by.items() if cc == "human" for x, y in rj.values()]
        rows.append({"delta": d, "policy_min_share": share, "human_min_share": sum(hs) / max(len(hs), 1), "class_min_share": cls_share, "class_fail": cls_fail})
    # pure rules for reference
    ref = {}
    for name, f in (("afri_only", lambda x, y: x), ("ssa_only", lambda x, y: y), ("mean", lambda x, y: 0.5 * (x + y)), ("min", min)):
        ref[name] = {c: sum(1 for (iso, cc), rj in hack_by.items() if cc == c for j, (x, y) in rj.items()
                            if (lambda hum: hum is not None and f(x, y) >= f(*hum))(hack_by[(iso, "human")].get(j))) / max(sum(len(rj) for (iso, cc), rj in hack_by.items() if cc == c), 1)
                     for c in classes if c != "human"}
    lo, hi = a.target_low, a.target_high
    tgt = (lo + hi) / 2
    inside = [r for r in rows if lo <= r["policy_min_share"] <= hi]
    # the SMALLEST delta whose min-branch share on the policy samples is <= target_high (the smaller the delta, the more of the disagreeing - copy-like - chunks the pessimist takes)
    ok = [r for r in rows if r["policy_min_share"] <= hi]
    best = ok[0] if ok else rows[-1]
    delta = best["delta"]
    # per language and per content mode, at the chosen delta
    by_lang = defaultdict(list)
    for iso, x, y in pol_pairs:
        by_lang[iso].append((x, y))
    lang_rows = {iso: {"n": len(v), "min_share": sum(1 for x, y in v if abs(x - y) > delta) / len(v), "mean_signed_diff_ssa_minus_afri": sum(y - x for x, y in v) / len(v),
                       "mean_r1": sum(x for x, _ in v) / len(v), "mean_r2": sum(y for _, y in v) / len(v)} for iso, v in by_lang.items()}
    mode_share = {}
    for mname in ("qe", "blend"):
        vv = [(x, y) for iso, x, y in pol_pairs if (ens.shaper1.content.get(iso) or {}).get("mode", "qe") == mname]
        mode_share[mname] = {"n": len(vv), "min_share": sum(1 for x, y in vv if abs(x - y) > delta) / max(len(vv), 1)}
    q = lambda p: d_all[min(len(d_all) - 1, int(p * len(d_all)))]
    report = {"delta": delta, "per_language": lang_rows, "per_mode": mode_share, "target": [lo, hi], "policy_chunks": len(pol_pairs), "abs_diff_quantiles": {str(p): q(p) for p in (0.5, 0.75, 0.85, 0.9, 0.95, 0.99)},
              "grid": rows, "reference_rules_fail": ref, "inside_target_deltas": [r["delta"] for r in inside]}
    json.dump(report, open(a.out, "w"), indent=1)
    cls_show = [c for c in classes if c != "human"]
    L = [f"delta search on {len(pol_pairs)} gate-open policy samples (languages judged by both models) and the hack classes of {len(set(i for i, _ in hack_by))} languages",
         f"|r1 - r2| quantiles on policy samples: " + ", ".join(f"p{int(p * 100)}={q(p):.2f}" for p in (0.5, 0.75, 0.85, 0.9, 0.95, 0.99)), "",
         "| delta | policy min-share | human min-share | " + " | ".join(f"{c} min / fail" for c in cls_show) + " |", "|---|---|---|" + "---|" * len(cls_show)]
    for r in rows:
        if r["delta"] in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0) or r["delta"] == delta:
            L.append(f"| {r['delta']:.1f}{' **' if r['delta'] == delta else ''} | {r['policy_min_share']:.3f} | {r['human_min_share']:.3f} | " +
                     " | ".join(f"{r['class_min_share'][c]:.2f} / {r['class_fail'][c]:.3f}" for c in cls_show) + " |")
    L += ["", "failure rate (corrupted reward >= human reward of the same sentence) of the pure rules: " + "; ".join(f"{k}: " + " ".join(f"{c}={v[c]:.3f}" for c in cls_show) for k, v in ref.items())]
    L += ["", f"At delta {delta}: min-branch share by content mode {mode_share}; per language (policy samples): " +
          ", ".join(f"{iso}={v['min_share']:.2f}" for iso, v in sorted(lang_rows.items(), key=lambda kv: -kv[1]['min_share']))]
    L += ["mean signed shaped difference (SSA minus AfriCOMET) per language: " + ", ".join(f"{iso}={v['mean_signed_diff_ssa_minus_afri']:+.2f}" for iso, v in sorted(lang_rows.items()))]
    open(a.out.replace(".json", ".md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("CHOSEN delta:", delta)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--work", required=True)
    s.add_argument("--models", nargs="+", required=True, help="NAME=MODEL_ID ...")
    s.add_argument("--languages", nargs="*", default=None)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--max-per-language", type=int, default=N_MAX)
    s.add_argument("--gpu-mem-gb", type=float, default=0.0)
    s.add_argument("--hub-snapshot", default=os.path.expanduser("~/.cache/huggingface/hub/datasets--openlanguagedata--flores_plus/snapshots/x"))
    s.add_argument("--bench-dev", default="/workspace/sft_build/bench/bench_dev.jsonl")
    s.add_argument("--ibom-efi", default="/workspace/sft_build/bench/test_sets/ibom_mt/en_efi.jsonl")
    s.set_defaults(fn=cmd_score)
    c = sub.add_parser("compose")
    c.add_argument("--work", required=True)
    c.add_argument("--v7", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--table", default=None, help="markdown table output")
    c.add_argument("--delta", type=float, required=True)
    c.add_argument("--primary", default="africomet")
    c.add_argument("--secondary", default="ssa_mtl")
    c.add_argument("--secondary-id", default="McGill-NLP/ssa-comet-mtl")
    c.set_defaults(fn=cmd_compose)
    d = sub.add_parser("delta")
    d.add_argument("--v8", required=True, help="a v8 file composed with any provisional delta (the qe2 anchors do not depend on it)")
    d.add_argument("--a5", default="/workspace/jobs/a5")
    d.add_argument("--out", required=True)
    d.add_argument("--target-low", type=float, default=0.10)
    d.add_argument("--target-high", type=float, default=0.15)
    d.set_defaults(fn=cmd_delta)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
