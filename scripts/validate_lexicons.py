# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validate the dictionary-overlap signal per language (D-57, agent B3): is `lexicon_overlap` a measure of translation quality?

    python scripts/validate_lexicons.py --lexicon lexicon --corpus corpus --flores flores --samples samples.jsonl --out validation.json \
        [--lexicon-rel lexicon_rel --lexicon-nonrel lexicon_nonrel] [--iso kab fon ...]

For every language with a lexicon:
  * FLORES+ DEV (never devtest, D-24; `flores/<code>.jsonl` + `flores/eng_Latn.jsonl`): human reference vs (a) the reference of a different sentence (mean of 5 shifts),
    (b) a copy of the English source, (c) the first half of the reference words (as calibrate_shaping): pooled AUC (Mann-Whitney, ties 0.5) and the
    same-source accuracy, on the sources where the lexicon covers >= MIN_COVERED content words; coverage = mean covered / content words;
  * hold-out pairs of the training corpus (whole English sentences that were not trained on), split religious / non-religious: AUC vs mismatched;
  * biblical bias: the same AUC without the entries flagged `bib`; AUC of a lexicon trained on the religious pairs only / the non-religious pairs only;
  * model outputs (`samples.jsonl`: iso, id, k, src, ref, hyp from qe_choice_sample.py): within-prompt Spearman of the overlap with sentence chrF++
    (mean over prompts), the pooled Spearman, and the gaming tests: AUC of the overlap of the model's own output vs the output of ANOTHER source for the same
    source (attractor outputs), vs the English source copied, and the overlap of a stuffed output (top candidates of the commonest English words).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import unicodedata
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemma_mt.languages import LANGUAGES  # noqa: E402
from gemma_mt.rl.lexicon import MIN_COVERED, english_content_stems, load_lexicon, x_folded_tokens  # noqa: E402


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    from scipy.stats import rankdata
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    r = rankdata(np.concatenate([p, n]))
    return float((r[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def flores_pairs(flores_dir: str, code: str):
    en = {int(r["id"]): r["text"] for r in read_jsonl(os.path.join(flores_dir, "eng_Latn.jsonl"))}
    p = os.path.join(flores_dir, f"{code}.jsonl")
    if not os.path.exists(p):
        return None
    ref = {int(r["id"]): r["text"] for r in read_jsonl(p)}
    ids = sorted(i for i in en if i in ref and ref[i].strip())
    return [en[i] for i in ids], [unicodedata.normalize("NFC", ref[i]) for i in ids]


def ov(lex, src, mt, **kw):
    return lex.overlap(src, mt, **kw)


def separation(lex, src: List[str], ref: List[str], **kw) -> Dict:
    """human vs mismatched / copy / half-dropped on the sources the lexicon covers."""
    n = len(src)
    hum = [ov(lex, s, r, **kw) for s, r in zip(src, ref)]
    ok = [i for i in range(n) if hum[i].overlap is not None]
    out: Dict = {"n": n, "n_scored": len(ok), "share_scored": len(ok) / max(n, 1),
                 "coverage": float(np.mean([h.coverage for h in hum])) if hum else 0.0,
                 "coverage_scored": float(np.mean([hum[i].coverage for i in ok])) if ok else 0.0}
    if len(ok) < 20:
        return out
    h = np.array([hum[i].overlap for i in ok])
    out["mean_human"] = float(h.mean())
    dh = [hum[i].density for i in range(n) if hum[i].density is not None]
    out["density_human_mean"] = float(np.mean(dh)) if dh else float("nan")
    out["density_human_p95"] = float(np.percentile(dh, 95)) if dh else float("nan")
    out["mean_human_guarded"] = float(np.mean([lex.guarded(hum[i]) or 0.0 for i in ok]))
    aucs, accs, means = [], [], []
    for k in (1, 7, 31, 101, 307):
        k = k % n or 1
        mm = np.array([ov(lex, src[i], ref[(i + k) % n], **kw).overlap or 0.0 for i in ok])
        aucs.append(auc(h, mm))
        accs.append(float((h > mm).mean() + 0.5 * (h == mm).mean()))
        means.append(float(mm.mean()))
    out["auc_vs_mismatched"] = float(np.mean(aucs))
    out["auc_vs_mismatched_sd"] = float(np.std(aucs))
    out["acc_vs_mismatched"] = float(np.mean(accs))
    out["mean_mismatched"] = float(np.mean(means))
    cp = np.array([ov(lex, src[i], src[i], **kw).overlap or 0.0 for i in ok])
    out["auc_vs_copy"] = auc(h, cp)
    out["mean_copy"] = float(cp.mean())
    hf = np.array([ov(lex, src[i], " ".join(ref[i].split()[: max(1, len(ref[i].split()) // 2)]), **kw).overlap or 0.0 for i in ok])
    out["auc_vs_half_dropped"] = auc(h, hf)
    out["mean_half_dropped"] = float(hf.mean())
    return out


def heldout_pairs(corpus_dir: str, iso: str, rel: bool, cap: int = 600):
    import pandas as pd
    p = os.path.join(corpus_dir, f"{iso}.parquet")
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p)
    d = d[d["held"] & (d["rel"] == rel)].drop_duplicates("en").head(cap)
    if len(d) < 30:
        return None
    return list(d["en"]), [unicodedata.normalize("NFC", x) for x in d["x"]]


def stuffed_output(lex, flores_src: List[str], n_words: int) -> str:
    """The attack on a recall-only signal: the top candidate of the commonest English content words of the FLORES sources, as one output."""
    df = collections.Counter(w for s in flores_src for w in set(english_content_stems(s)))
    out = []
    for w, _ in df.most_common():
        e = lex.entries.get(w)
        if e and e[2]:
            out.append(e[2][0][0])
        if len(out) >= n_words:
            break
    return " ".join(out)


def model_tests(lex, rows: List[Dict], flores_src: List[str], **kw) -> Dict:
    from scipy.stats import spearmanr
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=2)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    rho, pooled_o, pooled_c, per_prompt_n = [], [], [], 0
    own, other, copy_, ids = [], [], [], sorted(by)
    rng = random.Random(1)
    for pid in ids:
        rs = by[pid]
        o = [ov(lex, r["src"], r["hyp"], **kw) for r in rs]
        c = [chrf.sentence_score(unicodedata.normalize("NFC", r["hyp"]), [unicodedata.normalize("NFC", r["ref"])]).score for r in rs]
        if all(x.overlap is not None for x in o):
            ov_ = [x.overlap for x in o]
            pooled_o += ov_
            pooled_c += c
            if len(set(ov_)) > 1 and len(set(c)) > 1:
                rho.append(spearmanr(ov_, c)[0])
            per_prompt_n += 1
        # attractor test: the same source against the output of another source
        if o[0].overlap is not None:
            j = rng.choice([x for x in ids if x != pid])
            own.append(np.mean([x.overlap for x in o if x.overlap is not None]))
            other.append(ov(lex, rs[0]["src"], by[j][0]["hyp"], **kw).overlap or 0.0)
            copy_.append(ov(lex, rs[0]["src"], rs[0]["src"], **kw).overlap or 0.0)
    out: Dict = {"n_prompts": len(ids), "n_prompts_scored": per_prompt_n}
    if rho:
        out["spearman_within_prompt"] = float(np.mean(rho))
        out["spearman_within_prompt_n"] = len(rho)
        out["spearman_within_prompt_share_pos"] = float(np.mean([r > 0 for r in rho]))
    if len(pooled_o) > 20:
        out["spearman_pooled"] = float(spearmanr(pooled_o, pooled_c)[0])
    if own:
        out["auc_own_vs_other_source_output"] = auc(own, other)
        out["mean_overlap_model_output"] = float(np.mean(own))
        out["mean_overlap_other_source_output"] = float(np.mean(other))
        out["mean_overlap_english_copy"] = float(np.mean(copy_))
    srcs = list({r["src"]: 1 for r in rows})
    for nw in (30, 80):
        st = stuffed_output(lex, flores_src or srcs, nw)
        rs = [ov(lex, s, st, **kw) for s in srcs]
        vals = [x.overlap for x in rs if x.overlap is not None]
        if vals:
            out[f"mean_overlap_stuffed_{nw}"] = float(np.mean(vals))
            out[f"mean_overlap_stuffed_{nw}_guarded"] = float(np.mean([lex.guarded(x) or 0.0 for x in rs if x.overlap is not None]))
            out[f"density_stuffed_{nw}"] = float(np.mean([x.density for x in rs if x.density is not None]))
    own_g = [lex.guarded(ov(lex, r["src"], r["hyp"], **kw)) for r in rows]
    own_g = [x for x in own_g if x is not None]
    if own_g:
        out["mean_overlap_model_output_guarded"] = float(np.mean(own_g))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", default="lexicon")
    ap.add_argument("--lexicon-rel", default="")
    ap.add_argument("--lexicon-nonrel", default="")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--flores", default="flores")
    ap.add_argument("--samples", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iso", nargs="*")
    ap.add_argument("--match", default="", help="force a match mode (exact|affix) instead of the lexicon's own")
    a = ap.parse_args(argv)
    os.environ["GEMMA_MT_LEXICON_DIR"] = a.lexicon
    samples = collections.defaultdict(list)
    if a.samples:
        for r in read_jsonl(a.samples):
            samples[r["iso"]].append(r)
    isos = a.iso or sorted(os.path.basename(p)[:-5] for p in os.listdir(a.lexicon) if p.endswith(".json") and not p.startswith("_"))
    res: Dict[str, Dict] = {}
    for iso in isos:
        lex = load_lexicon(iso, a.lexicon)
        if lex is None:
            continue
        e: Dict = {"entries": len(lex), "bib_entries": int(sum(v[1] for v in lex.entries.values())), "pairs": lex.meta.get("pairs"),
                   "pairs_nonrel": lex.meta.get("pairs_nonrel")}
        code = LANGUAGES[iso].flores
        fl = flores_pairs(a.flores, code) if code else None
        kw0 = {"match": a.match} if a.match else {}
        if fl:
            src, ref = fl
            e["flores"] = {}
            dens = [x.density for x in (ov(lex, s_, r_, match="affix") for s_, r_ in zip(src, ref)) if x.density is not None]
            if dens:
                lex.meta["density_max"] = float(np.percentile(dens, 95))
                e["density_max"] = lex.meta["density_max"]
            for mode in ("exact", "affix"):
                e["flores"][mode] = separation(lex, src, ref, match=mode)
            e["flores"]["exact_no_bib"] = separation(lex, src, ref, match="exact", exclude_biblical=True)
            e["flores"]["affix_no_bib"] = separation(lex, src, ref, match="affix", exclude_biblical=True)
            e["flores"]["exact_top1"] = separation(lex, src, ref, match="exact", topk=1)
            e["flores"]["exact_uniform"] = separation(lex, src, ref, match="exact", weighting="uniform")
            for tag, d in (("rel_only", a.lexicon_rel), ("nonrel_only", a.lexicon_nonrel)):
                if d and os.path.exists(os.path.join(d, f"{iso}.json")):
                    lx = load_lexicon(iso, d)
                    e["flores"][tag] = {"entries": len(lx), **{m: separation(lx, src, ref, match=m) for m in ("exact", "affix")}}
            if samples.get(iso):
                e["model"] = {m: model_tests(lex, samples[iso], src, match=m) for m in ("exact", "affix")}
        for rel in (False, True):
            hp = heldout_pairs(a.corpus, iso, rel)
            if hp:
                e["heldout_rel" if rel else "heldout_nonrel"] = {m: separation(lex, hp[0], hp[1], match=m) for m in ("exact", "affix")}
        res[iso] = e
        f = e.get("flores", {}).get("exact", {})
        print(iso, "entries", e["entries"], "flores cov %.2f auc %.3f" % (f.get("coverage", float("nan")), f.get("auc_vs_mismatched", float("nan"))),
              "affix auc %.3f" % e.get("flores", {}).get("affix", {}).get("auc_vs_mismatched", float("nan")), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
