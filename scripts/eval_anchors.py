#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure the constraint penalties of rl/anchors.py (D-56): false positives on HUMAN references, detection of corruptions, and the rates on a model's outputs.

    python scripts/eval_anchors.py --calibration calibration/anchors_v1.json --human human_units.jsonl --pred pred_final1433.jsonl --out reports/anchors_eval_2026-10-21.json \
        [--lid-model model_v3.bin] [--languages ...] [--per-language 300]

--human: the jsonl written by `build_anchors_calibration.py --dump-human` (rows iso, src, ref, set, id; documents carry doc=true).
--pred: greedy outputs of a model (scripts' anchors_gen.py): rows iso, english, reference (may be null), hyp, finished.

Per detector and language group it reports
  fp_human       share of HUMAN references (sentences) with a deficit > 0 (the detector would punish a good human translation)
  fp_doc         the same for pseudo-documents of 10 / 25 consecutive sentences (repetition: also real documents; boilerplate: against a buffer of other documents)
  detect[...]    share of CORRUPTED human references with a deficit > 0: source_copy, half_dropped, appended (another source's reference appended), other_source (another
                 source's reference), looped (last clause repeated 6x), doubled, stock (a stock sentence appended to every output: the exploit the boilerplate detector is for)
  model          share of the final model's outputs punished, and the mean penalty at the default weights, next to the human references of the same sources
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import anchors_common as C  # noqa: E402
from gemma_mt.rl import anchors as A  # noqa: E402

WEIGHTS = {"entity": 0.5, "repeat": 1.0, "boiler": 1.0, "lexicon": 0.0, "lid_prob": 0.0}      # the RewardConfig defaults (rl/reward.py)
CORRUPTIONS = ("source_copy", "half_dropped", "appended", "other_source", "looped", "doubled")


def pct(x: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(x, dtype=float), q)) if len(x) else float("nan")


def rate(flags: Sequence[bool]) -> Optional[float]:
    return float(np.mean(flags)) if len(flags) else None


def detect_sample(an: A.Anchors, lang: str, src: str, mt: str, key: int = 0, with_boiler: bool = False) -> Dict[str, Optional[float]]:
    """deficits (0..1) of the entity and repeat detectors for one (source, output); None = not judged."""
    ent, n = an.entity_deficit(lang, src, mt)
    p = A.prepare(lang, src, mt, key, with_grams=with_boiler)
    return {"entity": ent if n else None, "repeat": an.repeat_deficit(lang, p), "_p": p}


def boiler_scores(an: A.Anchors, lang: str, items: Sequence[Dict[str, str]], key_of, capacity: int = 600, seed: int = 0, n_buffer: Optional[int] = None):
    """Fill a buffer with the outputs of `n_buffer` prompts (each its own prompt) and score the rest: returns the boilerplate deficits of the scored ones."""
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    nb = min(n_buffer or capacity, max(len(idx) - 1, 0))
    score_all = n_buffer is None                                    # hold-one-out: every output (its own prompt key is excluded) is scored against the buffer
    buf = A.BoilerplateBuffer(capacity=capacity, min_outputs=1, min_prompts=1, per_prompt=1, related_min=an.buffer.related_min)
    ps = {i: A.prepare(lang, items[i]["src"], items[i]["mt"], key=i) for i in idx}
    for i in idx[:nb]:
        buf.add(lang, ps[i].key, ps[i].grams, ps[i].src_hashes)
    blk = an._blk(lang, "boiler") or an.pooled.get("boiler") or {}
    t, cap = float(blk.get("thr", 0.03)), float(blk.get("cap", 0.4))
    out = []
    for i in (idx if score_all else idx[nb:]):
        s = buf.score(lang, ps[i].key, ps[i].grams, ps[i].src_hashes)
        if s is not None:
            out.append(A._clip01((s - t) / (cap - t)))
    return out


def corrupt(kind: str, u: Dict[str, str], other: Dict[str, str], rng: random.Random) -> str:
    if kind == "source_copy":
        return u["src"]
    if kind == "half_dropped":
        return C.corrupt_half_dropped(u["ref"], rng)
    if kind == "appended":
        return C.corrupt_appended(u["ref"], other["ref"], rng)
    if kind == "other_source":
        return other["ref"]
    if kind == "looped":
        return C.corrupt_looped(u["ref"], rng)
    if kind == "doubled":
        return C.corrupt_doubled(u["ref"], rng)
    raise KeyError(kind)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--human", required=True)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--languages", nargs="*", default=None)
    ap.add_argument("--per-language", type=int, default=300, help="human units per language for the corruption tests")
    ap.add_argument("--control", nargs="*", default=["yor", "hau", "zul"])
    ap.add_argument("--lid-model", default=None)
    ap.add_argument("--related-min", type=int, default=A.BOILER_RELATED_MIN)
    a = ap.parse_args(argv)

    t0 = time.time()
    cal = json.load(open(a.calibration, encoding="utf-8"))
    weak = set(cal["_meta"]["weak_languages"])
    an = A.Anchors(cal, modes=("qe", "blend", "consensus"), detectors=("entity", "repeat", "boiler"), related_min=a.related_min)
    rng = random.Random(11)
    human: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    docs: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in C.read_jsonl(a.human):
        (docs if r.get("doc") else human)[r["iso"]].append(r)
    langs = a.languages or sorted(set(human) | set(docs))
    res: Dict[str, Any] = {"_meta": {"calibration": os.path.basename(a.calibration), "weights": WEIGHTS, "per_language": a.per_language, "related_min": a.related_min}, "languages": {}}

    for lang in langs:
        us = human.get(lang) or []
        blk: Dict[str, Any] = {"weak": lang in weak, "n_human": len(us), "enabled": {d: an.enabled(lang, d) for d in ("entity", "repeat", "boiler")}}
        if len(us) >= 60:
            sub = rng.sample(us, min(a.per_language, len(us)))
            # ---- false positives on the human references (sentences)
            fe, fr, pe, pr = [], [], [], []
            for u in sub:
                d = detect_sample(an, lang, u["src"], u["ref"])
                if d["entity"] is not None:
                    fe.append(d["entity"] > 0)
                if d["repeat"] is not None:
                    fr.append(d["repeat"] > 0)
                pe.append(WEIGHTS["entity"] * (d["entity"] or 0.0))
                pr.append(WEIGHTS["repeat"] * (d["repeat"] or 0.0))
            items = [{"src": u["src"], "mt": u["ref"]} for u in us[:1200]]
            fb = boiler_scores(an, lang, items, None, seed=1) if an.enabled(lang, "boiler") else []
            blk["fp_human"] = {"entity": rate(fe), "entity_judged": len(fe), "repeat": rate(fr), "boiler": rate([x > 0 for x in fb]), "boiler_judged": len(fb)}
            blk["mean_penalty_human"] = {"entity": float(np.mean(pe)), "repeat": float(np.mean(pr)), "boiler": WEIGHTS["boiler"] * float(np.mean(fb)) if fb else 0.0}
            # ---- pseudo-documents and real documents
            pd = C.pseudo_documents(us, ks=(10, 25))
            fd = {}
            for k in (10, 25):
                ps = [u for u in pd if u["set"].endswith(f"/w{k}")]
                flags = []
                for u in ps:
                    d = detect_sample(an, lang, u["src"], u["ref"])
                    if d["repeat"] is not None:
                        flags.append(d["repeat"] > 0)
                fb2 = boiler_scores(an, lang, [{"src": u["src"], "mt": u["ref"]} for u in ps], None, seed=2) if (len(ps) > 40 and an.enabled(lang, "boiler")) else []
                fd[f"w{k}"] = {"n": len(ps), "repeat": rate(flags), "boiler": rate([x > 0 for x in fb2]), "boiler_judged": len(fb2)}
            dd = docs.get(lang) or []
            if dd:
                flags = [d["repeat"] > 0 for d in (detect_sample(an, lang, u["src"], u["ref"]) for u in dd) if d["repeat"] is not None]
                fd["real_docs"] = {"n": len(dd), "repeat": rate(flags)}
            blk["fp_doc"] = fd
            # ---- corruptions
            det: Dict[str, Any] = {}
            for kind in CORRUPTIONS:
                fe, fr, pe, pr = [], [], [], []
                for i, u in enumerate(sub):
                    o = sub[(i * 7 + 3) % len(sub)] if len(sub) > 1 else u
                    if o is u:
                        o = sub[(i + 1) % len(sub)]
                    mt = corrupt(kind, u, o, rng)
                    d = detect_sample(an, lang, u["src"], mt)
                    if d["entity"] is not None:
                        fe.append(d["entity"] > 0)
                    if d["repeat"] is not None:
                        fr.append(d["repeat"] > 0)
                    pe.append(WEIGHTS["entity"] * (d["entity"] or 0.0))
                    pr.append(WEIGHTS["repeat"] * (d["repeat"] or 0.0))
                det[kind] = {"entity": rate(fe), "entity_judged": len(fe), "repeat": rate(fr), "repeat_judged": len(fr),
                             "mean_penalty_entity": float(np.mean(pe)), "mean_penalty_repeat": float(np.mean(pr))}
            stock = rng.choice([u["ref"] for u in us if 40 <= len(u["ref"]) <= 140] or [us[0]["ref"]])
            it = [{"src": u["src"], "mt": u["ref"].rstrip() + " " + stock} for u in us[:600]]
            sb = boiler_scores(an, lang, it, None, seed=3, n_buffer=300) if an.enabled(lang, "boiler") else []
            det["stock"] = {"boiler": rate([x > 0 for x in sb]), "boiler_mean_deficit": float(np.mean(sb)) if sb else None, "boiler_judged": len(sb),
                           "mean_penalty_boiler": WEIGHTS["boiler"] * float(np.mean(sb)) if sb else None}
            blk["detect"] = det
        res["languages"][lang] = blk

    # ---------------------------------------------------------------- the final model's outputs
    if a.pred:
        by: Dict[str, List[dict]] = defaultdict(list)
        for r in C.read_jsonl(a.pred):
            by[r["iso"]].append(r)
        lid = None
        if a.lid_model:
            from gemma_mt.rl.lid import GlotLIDClassifier
            from gemma_mt.languages import LANGUAGES

            lid = GlotLIDClassifier(path=a.lid_model)
        model: Dict[str, Any] = {}
        for lang, rows in sorted(by.items()):
            fin = [r for r in rows if r.get("finished", True) and (r.get("hyp") or "").strip()]
            m: Dict[str, Any] = {"n": len(rows), "finished": len(fin), "weak": lang in weak}
            pen = {"entity": [], "repeat": [], "boiler": []}
            flags = {"entity": [], "repeat": []}
            for r in fin:
                d = detect_sample(an, lang, r["english"], r["hyp"])
                for k in ("entity", "repeat"):
                    if d[k] is not None:
                        flags[k].append(d[k] > 0)
                        pen[k].append(d[k] * WEIGHTS[k])
            if an.enabled(lang, "boiler") and len(fin) > 5:
                bs = boiler_scores(an, lang, [{"src": r["english"], "mt": r["hyp"]} for r in fin], None, seed=5)
                flags["boiler"] = [x > 0 for x in bs]
                pen["boiler"] = [x * WEIGHTS["boiler"] for x in bs]
            m["enabled"] = {d: an.enabled(lang, d) for d in ("entity", "repeat", "boiler")}
            for k in ("entity", "repeat", "boiler"):
                m[f"flag_{k}"] = rate(flags.get(k, []))
                m[f"judged_{k}"] = len(pen[k])
                m[f"penalty_{k}"] = float(np.mean(pen[k])) if pen[k] else None
            # the human references of the same sources (apples to apples)
            hr = [r for r in fin if r.get("reference")]
            if len(hr) >= 20:
                hf = {"entity": [], "repeat": []}
                for r in hr:
                    d = detect_sample(an, lang, r["english"], r["reference"])
                    for k in hf:
                        if d[k] is not None:
                            hf[k].append(d[k] > 0)
                hb = boiler_scores(an, lang, [{"src": r["english"], "mt": r["reference"]} for r in hr], None, seed=5) if an.enabled(lang, "boiler") and len(hr) > 5 else []
                m["human_same_sources"] = {"n": len(hr), "flag_entity": rate(hf["entity"]), "flag_repeat": rate(hf["repeat"]), "flag_boiler": rate([x > 0 for x in hb]),
                                           "boiler_share_mean": None}
            # boilerplate share (not the deficit): mean share of 5-grams found in >= 3 other prompts' outputs
            shares = []
            if len(fin) > 40:
                buf = A.BoilerplateBuffer(capacity=600, min_outputs=1, min_prompts=1, per_prompt=1, related_min=a.related_min)
                ps = [A.prepare(lang, r["english"], r["hyp"], key=i) for i, r in enumerate(fin)]
                for p in ps:
                    buf.add(lang, p.key, p.grams, p.src_hashes)
                shares = [s for s in (buf.score(lang, p.key, p.grams, p.src_hashes) for p in ps) if s is not None]
                m["boiler_share_mean"] = float(np.mean(shares)) if shares else None
                m["boiler_share_p95"] = pct(shares, 0.95) if shares else None
                if hr:
                    buf = A.BoilerplateBuffer(capacity=600, min_outputs=1, min_prompts=1, per_prompt=1, related_min=a.related_min)
                    hp = [A.prepare(lang, r["english"], r["reference"], key=i) for i, r in enumerate(hr)]
                    for p in hp:
                        buf.add(lang, p.key, p.grams, p.src_hashes)
                    hs = [s for s in (buf.score(lang, p.key, p.grams, p.src_hashes) for p in hp) if s is not None]
                    m["human_same_sources"]["boiler_share_mean"] = float(np.mean(hs)) if hs else None
            if lid is not None:
                from gemma_mt.languages import LANGUAGES as _L

                lb = (cal["languages"].get(lang) or {}).get("lid_prob")
                labels = set(_L[lang].glotlid)
                if lb and labels:
                    tops = lid.top([r["hyp"] for r in fin], 1)
                    ok = [t[0][1] for t in tops if t and t[0][0] in labels]
                    m["lid"] = {"label_ok": len(ok) / max(len(fin), 1), "below_pmin": float(np.mean([p < lb["pmin"] for p in ok])) if ok else None, "pmin": lb["pmin"]}
            model[lang] = m
        res["model"] = model
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[anchors-eval] wrote {a.out} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
