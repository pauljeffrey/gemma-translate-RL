#!/usr/bin/env python3
"""(c) RL best-translation archive -> SFT windows, from the NEW clean RL run's archive.sqlite (agent C2, 2026-10-28).

    python s4_archive.py --archive newrl=/path/archive.sqlite --out OUT --index protected_index.npz --manifest protected_manifest.json \
        --calibration CAL.json --anchors ANCHORS.json --glotlid model_v3.bin --scorer scorer.json --pools clean_v3,clean_sci_v2 [--model-label NEW_RL_MODEL]

Same selection as the A2W component A (scripts/mix2_archive.py, D-63), with the three differences the clean rerun needs:
  * the hold-out is the strict protected-index check (decon.py, no exemption) on the English source AND the translation of every row;
  * the licence/provenance of a row is joined from the CLEAN prompt pools (key = sha1(iso + NFC(source))[:16]); a row whose source is not in the clean pools is dropped
    (`source_not_in_clean_pool`; `--allow-unmatched` keeps it with the licence `unknown`);
  * the judges use the REFIT calibration (--calibration, --anchors), not the 2026-10-19 files (those were fitted partly on benchmark text and must not be used for a real run).
Selection steps (each counts the rows it removes, per language, in archive_stats.json):
  1 row_problem (copy of the source, repetition, length ratio); 2 word 3/4/5-gram repetition above max(human p97 of the length bucket, the source's own);
  3 GlotLID guard, diacritic deficit == 0, length ratio inside the human window; 4 raised entry thresholds (qe reward >= 1.4, blend >= 1.25 and consensus chrF++ >= 60,
  consensus-only chrF++ >= 70); 5 re-judged by the QE ENSEMBLE (AfriCOMET-QE + SSA-COMET-MTL, rule of D-55) on the aligned sentences: weighted mean shaped reward >= 1.0 and
  p25 >= 0.3, no failed chunk; 6 protected-index check; 7 licence join, forbidden-source filter, optional --exclude-source, per-language token cap (--cap-tokens 60,000; b1024 rows first,
  then b512, then the rest, highest ensemble mean first).
A2W reference: 2,158 rows, 1,043,557 tokens, 65 languages.
Output: OUT/archive_selected.jsonl (english, target, iso, key, mode, reward, qe, consensus_chrf, ens_mean, ens_p25, length_bin, step_best, run, n_tokens, target_tokens + provenance
and licence columns) and OUT/archive_stats.json.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s4_common as K  # noqa: E402


def pool_licences(pool_dirs, isos, keys):
    """key -> (source_dataset, licence, prov_pool, prov_file, prov_index) from the clean pools."""
    import pyarrow.parquet as pq
    from gemma_mt.rl.archive import make_key

    out = {}
    for d in pool_dirs:
        for iso in sorted(isos):
            p = os.path.join(d, f"{iso}.parquet")
            if not os.path.exists(p):
                continue
            cols = [c for c in ("english", "iso", "source_dataset", "licence", "prov_pool", "prov_file", "prov_index") if c in pq.ParquetFile(p).schema_arrow.names]
            t = pq.read_table(p, columns=cols).to_pydict()
            for i in range(len(t["english"])):
                k = make_key(t["iso"][i] if "iso" in t else iso, t["english"][i])
                if k in keys and k not in out:
                    out[k] = (t["source_dataset"][i] or "", t["licence"][i] or "", t.get("prov_pool", [""] * len(t["english"]))[i], t.get("prov_file", [""] * len(t["english"]))[i],
                              t.get("prov_index", [""] * len(t["english"]))[i])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--archive", action="append", required=True, help="name=path of an archive.sqlite; repeat for every stage archive of the clean RL run (A B1 B2 B3 mixed)")
    ap.add_argument("--drop-after", action="append", default=[], help="name=STEP: drop rows of a run whose best step is later (a run that diverged)")
    ap.add_argument("--pools", default="", help="comma separated CLEAN pool directories (clean_v3, clean_sci_v2): licence join")
    ap.add_argument("--index", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--calibration", default="")
    ap.add_argument("--anchors", default="")
    ap.add_argument("--glotlid", default="")
    ap.add_argument("--scorer", default="", help="json file with the scorer spec (type worker|fake ...)")
    ap.add_argument("--no-lid", action="store_true", help="SMOKE only: no GlotLID")
    ap.add_argument("--model", default="google/gemma-3-270m-it", help="tokenizer used for the token counts")
    ap.add_argument("--model-label", default="", help="path/name of the RL model that wrote the archive (provenance)")
    ap.add_argument("--min-reward-qe", type=float, default=1.4)
    ap.add_argument("--min-reward-blend", type=float, default=1.25)
    ap.add_argument("--min-chrf-blend", type=float, default=60.0)
    ap.add_argument("--min-chrf-consensus", type=float, default=70.0)
    ap.add_argument("--min-ens-mean", type=float, default=1.0)
    ap.add_argument("--min-ens-p25", type=float, default=0.3)
    ap.add_argument("--cap-tokens", type=int, default=60000)
    ap.add_argument("--exclude-source", default="", help="comma separated substrings of source_dataset to drop (e.g. libretexts)")
    ap.add_argument("--allow-unmatched", action="store_true")
    ap.add_argument("--no-qe", action="store_true", help="skip the ensemble re-judgement (testing)")
    ap.add_argument("--nproc", type=int, default=2)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    arch = dict(x.split("=", 1) for x in a.archive)
    drop_after = {x.split("=")[0]: int(x.split("=")[1]) for x in a.drop_after}
    excl = [x.lower() for x in a.exclude_source.split(",") if x]

    K.gemma_pkg()
    C = K.setup_judge(a.calibration or None, a.anchors or None, a.glotlid or None, a.scorer or None)
    from gemma_mt import format as fmt
    from gemma_mt.rl.archive import make_key, read_rows
    from gemma_mt.rl.chunk_plan import plan_chunks

    AP = C.load_script("archive_publish")
    tok = K.tokenizer(a.model)
    J = C.Judge(with_scorer=not a.no_qe, with_lid=not a.no_lid)
    P = K.Protected(a.index, a.manifest, a.nproc)
    drops = collections.defaultdict(collections.Counter)
    rows = []
    for name, path in arch.items():
        for r in read_rows(path):
            r["run"] = name
            if name in drop_after and r["step_best"] > drop_after[name]:
                drops["suspect_run_tail"][r["iso"]] += 1
                continue
            rows.append(r)
    K.log(f"{len(rows)} archive rows read from {list(arch)}")
    n_in = collections.Counter(r["iso"] for r in rows)
    if len(arch) > 1:                                          # the clean RL run has one archive per stage (A, B1, B2, B3, mixed): the same source in several keeps the highest reward
        best = {}
        for r in rows:
            if r["key"] not in best or r["reward"] > best[r["key"]]["reward"]:
                if r["key"] in best:
                    drops["same_source_other_stage_lower_reward"][r["iso"]] += 1
                best[r["key"]] = r
            else:
                drops["same_source_other_stage_lower_reward"][r["iso"]] += 1
        rows = list(best.values())
        K.log(f"{len(rows)} distinct sources after merging the stages")
    keep = []
    for r in rows:
        iso = r["iso"]
        src, tr = unicodedata.normalize("NFC", r["source"]), unicodedata.normalize("NFC", r["translation"])
        r["source"], r["translation"] = src, tr
        p = AP.row_problem(r)
        if p:
            drops["row_problem:" + p][iso] += 1
            continue
        if J.repeat_bad(iso, src, tr):
            drops["repetition"][iso] += 1
            continue
        if not a.no_lid and J.lid_ok(iso, tr) is False:
            drops["lid"][iso] += 1
            continue
        if (r["diacritic_deficit"] or 0) > 1e-9 or J.diacritic_deficit(iso, src, tr) > 1e-9:
            drops["diacritics"][iso] += 1
            continue
        if not J.len_ok(iso, src, tr):
            drops["length_window"][iso] += 1
            continue
        if r["mode"] == "qe":
            ok = r["reward"] >= a.min_reward_qe
        elif r["mode"] == "blend":
            ok = r["reward"] >= a.min_reward_blend and (r["consensus_chrf"] or 0) >= a.min_chrf_blend
        else:
            ok = (r["consensus_chrf"] or 0) >= a.min_chrf_consensus
        if not ok:
            drops["below_raised_threshold:" + r["mode"]][iso] += 1
            continue
        keep.append(r)
    K.log(f"after cheap filters and raised thresholds: {len(keep)}")
    # ---- 5 ensemble re-judgement on the aligned sentences
    if not a.no_qe:
        segs, owner = [], []
        for i, r in enumerate(keep):
            if not J.has_qe(r["iso"]):
                continue
            plan = plan_chunks(r["source"], r["translation"], w_hi=(J.shaper.len_window.get(r["iso"]) or (None, None))[1])
            r["_wts"] = []
            for it in plan.items:
                if it[0] is not None:
                    s, m = plan.segments[it[0]]
                    segs.append((s, m, r["iso"]))
                    owner.append(i)
                    r["_wts"].append(float(max(len(s), 1)))
            r["_plan_failed"] = sum(1 for it in plan.items if it[0] is None)
        K.log(f"scoring {len(segs)} aligned sentence pairs of {len(set(owner))} rows with both judges")
        raw = J.qe_raw([s for s, _, _ in segs], [m for _, m, _ in segs], [l for _, _, l in segs]) if segs else []
        per = collections.defaultdict(list)
        for (s, m, l), (c1, c2), i in zip(segs, raw, owner):
            per[i].append(J.shaped(c1, c2, l)[2])
        for i, r in enumerate(keep):
            vals = per.get(i)
            if vals:
                w = r["_wts"]
                r["ens_mean"] = sum(v * x for v, x in zip(vals, w)) / sum(w)
                sv = sorted(vals)
                r["ens_p25"] = sv[int(0.25 * (len(sv) - 1))]
                r["n_sent"] = len(vals)
    final = []
    for r in keep:
        iso = r["iso"]
        if J.has_qe(iso) and not a.no_qe:
            if r.get("ens_mean") is None or r["ens_mean"] < a.min_ens_mean:
                drops["ensemble_mean"][iso] += 1
                continue
            if r["ens_p25"] < a.min_ens_p25:
                drops["ensemble_p25"][iso] += 1
                continue
            if r.get("_plan_failed"):
                drops["failed_chunk_in_plan"][iso] += 1
                continue
        final.append(r)
    K.log(f"after the ensemble: {len(final)}")
    # ---- 6 protected index, both sides (strict)
    hits = P.names([r["source"] for r in final] + [r["translation"] for r in final])
    keep2 = []
    for k, r in enumerate(final):
        h = hits[k] + hits[len(final) + k]
        if h:
            for nm in set(h):
                drops["protected:" + nm][r["iso"]] += 1
            drops["protected_any"][r["iso"]] += 1
            continue
        keep2.append(r)
    final = keep2
    K.log(f"after the protected-index check ({P.version}): {len(final)}")
    # ---- 7 licence join, tokens, per-language cap
    pool_dirs = [x for x in a.pools.split(",") if x]
    lic = pool_licences(pool_dirs, {r["iso"] for r in final}, {r["key"] for r in final}) if pool_dirs else {}
    sel_rows = []
    for r in final:
        sd, lc, pp, pf, pi = lic.get(r["key"], ("", "", "", "", ""))
        if not sd and not a.allow_unmatched:
            drops["source_not_in_clean_pool"][r["iso"]] += 1
            continue
        if K.is_forbidden_source(sd):
            drops["forbidden_source"][r["iso"]] += 1
            continue
        if any(x in sd.lower() for x in excl):
            drops["excluded_source"][r["iso"]] += 1
            continue
        r["source_dataset"], r["licence"], r["prov_pool"], r["prov_file"], r["prov_index_pool"] = sd or "unmatched", lc, pp, pf, pi
        ex = fmt.build_example(r["source"], r["iso"], r["translation"], tok)
        r["n_tokens"], r["target_tokens"] = ex.n_tokens, ex.target_len
        sel_rows.append(r)
    by = collections.defaultdict(list)
    for r in sel_rows:
        by[r["iso"]].append(r)
    sel, capped = [], collections.Counter()
    order_bin = {"b2048": 0, "b1024": 1, "b512": 2, "b256": 3, "b64": 4}
    for iso, rs in by.items():
        rs.sort(key=lambda r: (order_bin.get(r["length_bin"], 9), -(r.get("ens_mean") if r.get("ens_mean") is not None else r["reward"])))
        used = 0
        for r in rs:
            if r["n_tokens"] > K.SEQ_LEN:
                drops["longer_than_pack"][iso] += 1
                continue
            if used + r["n_tokens"] > a.cap_tokens:
                capped[iso] += 1
                continue
            used += r["n_tokens"]
            sel.append(r)
    K.log(f"selected {len(sel)} rows, {sum(r['n_tokens'] for r in sel):,} tokens, {len({r['iso'] for r in sel})} languages (A2W: 2,158 rows, 1,043,557 tokens, 65 languages)")
    out = []
    for r in sorted(sel, key=lambda r: (r["iso"], r["key"])):
        prov = K.provenance("archive", source_dataset=r["source_dataset"], licence=r["licence"], prov_repo=r.get("prov_file", ""),
                            prov_ref=f"archive:{r['run']};key={r['key']};step_best={r['step_best']};pool={r.get('prov_pool', '')}", prov_index=r.get("prov_index_pool", ""),
                            generator=f"RL policy {a.model_label or '(new RL model)'} best-of-run translation, re-judged by the QE ensemble")
        out.append({"english": r["source"], "target": r["translation"], "iso": r["iso"], "key": r["key"], "mode": r["mode"], "reward": r["reward"], "qe": r["qe"],
                    "consensus_chrf": r["consensus_chrf"], "ens_mean": r.get("ens_mean"), "ens_p25": r.get("ens_p25"), "length_bin": r["length_bin"], "step_best": r["step_best"],
                    "run": r["run"], "n_tokens": r["n_tokens"], "target_tokens": r["target_tokens"], **prov})
    K.write_jsonl(os.path.join(a.out, "archive_selected.jsonl"), out)
    langs = sorted(set(n_in) | {r["iso"] for r in sel})
    per_lang = {l: {"in": n_in.get(l, 0), "selected": sum(1 for r in sel if r["iso"] == l), "tokens": sum(r["n_tokens"] for r in sel if r["iso"] == l),
                    "by_bin": dict(collections.Counter(r["length_bin"] for r in sel if r["iso"] == l)), "capped": capped.get(l, 0),
                    "drops": {k: v[l] for k, v in drops.items() if v.get(l)}} for l in langs}
    stats = {"in": len(rows), "selected": len(sel), "tokens": sum(r["n_tokens"] for r in sel), "bins_selected": dict(collections.Counter(r["length_bin"] for r in sel)),
             "mode_selected": dict(collections.Counter(r["mode"] for r in sel)), "drops": {k: sum(v.values()) for k, v in drops.items()}, "index_version": P.version,
             "config": {k: v for k, v in vars(a).items()}, "per_language": per_lang}
    json.dump(stats, open(os.path.join(a.out, "archive_stats.json"), "w"), indent=1)
    J.close()
    P.close()
    K.log("done")


if __name__ == "__main__":
    main()
