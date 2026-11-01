#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build calibration/anchors_v3.json (D-68): anchors_v2.json + domain-robust entries for the languages whose v2 'pivot_held' calibration is religious-text only (or too thin).

    python scripts/build_anchors_v3_calibration.py --v2 calibration/anchors_v2.json --funcwords calibration/funcwords_v1.json --out calibration/anchors_v3.json \
        [--langs urh,sid,wal,bas,dua,yao,wes,oke,bbj,bin,ish,iso,swc,kon,kri,pcm] [--bench urh=owner_bench.jsonl]

For each language every available human-style set is one domain (scripts/v3_data.py): R3 pivot held-out rows, R15 JW windows (val / seen), SmolSent, GlotStoryBook, Wikimedia, FineTranslations web text ...
`--bench ISO=file.jsonl` adds a NATIVE-VERIFIED benchmark (fields english, reference) as an out-of-domain set. A Gemini / LLM silver set must NOT be given here (scripts/urh_silver_check.py is its only user).
The v2 blocks of the listed languages are replaced by blocks with `usable` / `gate_ok` / `basis`; the other languages are copied unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3_data as D  # noqa: E402
from gemma_mt.rl import anchors_calibrate_v2 as C  # noqa: E402
from gemma_mt.rl import anchors_calibrate_v3 as C3  # noqa: E402
from gemma_mt.rl import textstats as T  # noqa: E402

DEFAULT_LANGS = "urh,sid,wal,bas,dua,yao,wes,oke,bbj,bin,ish,iso,swc,kon,kri,pcm"
BASIS_NOTES = {"urh": "jw_pivot + gemini_silver_check (the 230 Gemini-generated pairs are NOT native-verified: used only to check the detectors' flag rates and coarse statistics, no threshold, dictionary or loanword list comes from them)"}


def load_bench(path: str):
    out = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        src, ref = r.get("english"), r.get("reference") or r.get("urhobo") or r.get("target")
        if src and ref:
            out.append({"src": src.strip(), "ref": ref.strip(), "set": "bench", "dom": "ood"})
    return out


def build(cal: dict, fwd: dict, langs, bench=None, lam=C3.DEFAULT_LAM, human="/workspace/jobs/b2/human_units.jsonl", log=print):
    cal = copy.deepcopy(cal)
    L = cal.setdefault("languages", {})
    report = {}
    for iso in langs:
        sets = D.load_sets(iso, human=human)
        extra = BASIS_NOTES.get(iso, "")
        if bench and iso in bench:
            have = {(" ".join(u["src"].split()).casefold(), " ".join(u["ref"].split()).casefold()) for us in sets.values() for u in us}
            sets["bench"] = [u for u in bench[iso] if (" ".join(u["src"].split()).casefold(), " ".join(u["ref"].split()).casefold()) not in have]
            extra = (extra + "; " if extra else "") + f"native-verified benchmark ({len(bench[iso])})"
        sets = {n: us for n, us in sets.items() if len(us) >= 8}
        if fwd.get(iso):                            # drop rows whose "reference" is English text (untranslated web / UI strings are not translations)
            ex0 = frozenset(C.english_excl(fwd[iso]["en_tf"], fwd[iso]["n_tok"], []))
            for n in list(sets):
                keep = [u for u in sets[n] if T.func_features(T.out_tokens(u["ref"]), frozenset(), ex0)["en"] < 0.12]
                if len(keep) < len(sets[n]):
                    log(f"[v3] {iso}/{n}: {len(sets[n]) - len(keep)} of {len(sets[n])} rows dropped (English-language reference)")
                sets[n] = keep
            sets = {n: us for n, us in sets.items() if len(us) >= 8}
        if not sets:
            log(f"[v3] {iso}: no human-style text at all -> every v2 block of it is switched off (usable false)")
            for det in ("copy", "func", "pad"):
                if (L.get(iso) or {}).get(det):
                    L[iso][det]["usable"] = False
                    L[iso][det]["gate_ok"] = False
                    L[iso][det]["basis"] = "none available"
            continue
        r = C3.build_language_v3(iso, sets, fwd.get(iso), lam=lam, basis_extra=extra)
        e = L.setdefault(iso, {})
        for det in ("copy", "func", "pad"):
            if r[det] is None:
                if e.get(det):
                    e[det]["usable"] = False
                    e[det]["gate_ok"] = False
                    e[det]["basis"] = "v3: not fittable from the available text"
            else:
                e[det] = r[det]
        if all(u.get("dom") == "rel" for us in sets.values() for u in us):
            e.pop("entity2", None)                 # religious-only drop rates (verse numbers, Bible names) are not evidence: the pooled prior (half weight) applies
        e["v3"] = {"sets": r["diagnostics"]["sets"], "ood_sets": r["diagnostics"]["ood_sets"], "lam": lam}
        report[iso] = {det: ({k: r[det].get(k) for k in ("enabled", "usable", "gate_ok", "t_w", "n")} if r[det] else None) for det in ("copy", "func", "pad")}
        log(f"[v3] {iso}: sets {r['diagnostics']['sets']} | " + " | ".join(f"{det} usable={r[det]['usable']} gate_ok={r[det]['gate_ok']}" if r[det] else f"{det} none" for det in ("copy", "func", "pad")))
    cal["v3_meta"] = {"lam": lam, "copy_floor": C3.DEFAULT_COPY_FLOOR, "copy_scale": C3.DEFAULT_COPY_SCALE, "languages": list(langs), "rule": "single-domain: copy+pad graded, func off, no gate; "
                      "with an out-of-domain held-out set: leave-one-set-out FP <= 3 % per detector (gate: saturated share <= 1 %)"}
    return cal, report


def retime_pad(cal: dict, human: str, bigrams_path, skip, log=print) -> dict:
    """Refit the `pad` block of every language that has human references with the length-independent segmental type / token ratio (ttr_mode seg100) and the real human DOCUMENTS (smol_doc, AfriScience
    papers: up to ~7,000 words) in the rows. The v2 thresholds came from windows of <= 250 words, so the plain type / token ratio of a 400-1,000-word human document (0.32-0.43) fell below
    them: measured v2 pad deficit > 0 on 5 % / 14 % / 31 % / 81 % / 100 % of human documents of < 100 / 100-200 / 200-400 / 400-800 / > 800 words."""
    import collections

    import numpy as np

    from gemma_mt.rl import anchors_calibrate_v2 as C

    bgf = np.load(bigrams_path) if bigrams_path else None
    sents, docs = collections.defaultdict(list), collections.defaultdict(list)
    seen = set()
    for l in open(human, encoding="utf-8"):
        r = json.loads(l)
        k = (r["iso"], " ".join(r["src"].split()).casefold(), " ".join(r["ref"].split()).casefold())
        if k in seen:
            continue
        seen.add(k)
        (docs if r["set"] == "smol_doc" or r["set"].endswith("_doc.jsonl") else sents)[r["iso"]].append(r)
    for iso, e in cal["languages"].items():
        if iso in skip or not e.get("pad") or "v3" in e or not sents.get(iso):
            continue
        bg = bgf[iso] if bgf is not None and iso in bgf.files else None
        use_bg = bg is not None and bg.size >= 20000
        units = sents[iso] + C3.pseudo(sents[iso], (4, 10, 25)) + docs.get(iso, [])
        rows = C.pad_rows(units, bg if use_bg else None, ttr_mode="seg100")
        fpm = C.GEEZ_FP_MAX if C.is_geez([u["ref"] for u in sents[iso][:300]]) else C.FUNC_FP_MAX
        b = C.fit_pad(rows, fp_max=fpm, ttr_mode="seg100")
        if b is None:
            continue
        old = e["pad"]
        b.update({"n_bigrams": int(bg.size) if bg is not None else 0, "enabled": bool(b["fp_cv"] <= fpm), "usable": True, "gate_ok": True, "basis": f"human sentences + pseudo-documents + {len(docs.get(iso, []))} real documents; seg100 type/token ratio (v3 retime)",
                  "v2_fp_cv": old.get("fp_cv"), "geez": fpm < C.FUNC_FP_MAX})
        e["pad"] = b
        of = e.get("func")                       # the same for the stop-word profile: the thresholds of v2 came from sentences and windows, the real documents were not in the rows
        if of and of.get("fw"):
            fws, ex = frozenset(of["fw"]), frozenset(of.get("en_excl") or ())
            fr = C.func_rows(units, fws, ex)
            fb = C.fit_func(fr, fp_max=fpm)
            if fb:
                fb.update({k: of[k] for k in ("fw", "en_excl", "k", "dict_source", "n_train_seg", "detect", "geez") if k in of})
                fb.update({"enabled": bool(of.get("enabled", True) and fb["fp_cv"] <= fpm), "usable": True, "gate_ok": True, "v2_fp_cv": of.get("fp_cv"),
                           "basis": f"human sentences + pseudo-documents + {len(docs.get(iso, []))} real documents (v3 retime)"})
                e["func"] = fb
    log(f"[v3] pad (and func) blocks refitted with the segmental type/token ratio: {sum(1 for e in cal['languages'].values() if (e.get('pad') or {}).get('ttr_mode') == 'seg100')} languages")
    return cal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--funcwords", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default=DEFAULT_LANGS)
    ap.add_argument("--bench", action="append", default=[], help="ISO=file.jsonl (native-verified; english + reference)")
    ap.add_argument("--lam", type=float, default=C3.DEFAULT_LAM)
    ap.add_argument("--retime-all", action="store_true", help="refit the pad block of every human-referenced language with the segmental type/token ratio")
    ap.add_argument("--bigrams", default=None)
    ap.add_argument("--human", default="/workspace/jobs/b2/human_units.jsonl")
    a = ap.parse_args()
    cal = json.load(open(a.v2, encoding="utf-8"))
    fwd = json.load(open(a.funcwords, encoding="utf-8"))
    bench = {s.split("=", 1)[0]: load_bench(s.split("=", 1)[1]) for s in a.bench}
    out, _ = build(cal, fwd, a.langs.split(","), bench, a.lam, human=a.human)
    if a.retime_all:
        out = retime_pad(out, a.human, a.bigrams, set(a.langs.split(",")))
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
