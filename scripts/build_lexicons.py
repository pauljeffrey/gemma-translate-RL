# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the English -> X lexicons of the dictionary-overlap signal (D-57, agent B3). CPU only (numpy), no network except the optional `gather`.

    # 1. rows of the weak-signal languages out of the pivot shards of <HF_ORG_LEGACY>/interlaanguage-translation  (see --help of `gather`)
    # 2. clean pairs + religion flag + hold-out split + general-English statistics
    python scripts/build_lexicons.py prep   --raw corpus_raw --pivot dl/full_pairs/pivot_en_sentences.parquet --out corpus
    # 3. IBM-1 both directions -> lexicon/<iso>.json
    python scripts/build_lexicons.py train  --corpus corpus --out lexicon [--iso kab fon ...] [--subset all|rel|nonrel]

Data: English-pivot pairs (two translations of the same English sentence; `group_key` -> `pivot_en_sentences.parquet`), X as src OR tgt, plus the pairs
with a `pivot_en` column of `sft_pairs_*.parquet`. 65-99 % of the pairs of these languages are religious (Bible / JW), so:
  * every pair carries `rel` (provenance religious_jw OR a religious cue word in the English sentence); the cap (300k) takes the non-religious pairs first
    and the rest of the budget is religious pairs; religious pairs get the weight --w-rel (0.3) in the EM counts;
  * the English words have IDF over a GENERAL corpus (English of the non-religious pairs + FLORES+ dev English), a `bib` flag (>= 6x more frequent in the
    religious English than in the general English) and only content words (stop list, IDF floor, >= 5 sentences) get entries;
  * a candidate needs >= MIN_JOINT sentences with both words ("several independent contexts"; an English sentence is used at most twice).
Algorithm: IBM Model 1 with NULL word, both directions (e->f and f->e), 6 EM iterations on unique types per sentence; the candidate score is the geometric
mean sqrt(t(f|e) * t(e|f)) (the symmetrised posterior removes the "garbage collector" rare words of one-directional IBM-1); top-3 per English stem.
Normalisation is rl/lexicon.py (fold = NFC + lowercase + marks removed; the surface variants are kept in `v`).
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemma_mt.rl.lexicon import STOP, _EN_WORD, en_stem, fold, x_tokens  # noqa: E402

WEAK = ("efi bam fon kab kam kbp kmb mos nus umb dyu sag swc loz ven iso tiv kqn wal lun kua kon ibb urh nbl bin dag oke shi bbj kea ach tsc vmw seh nyn her yao sid ndc "
        "kri ish dua bas wes").split()

REL_WORDS = re.compile(r"\b(god|gods|lord|jesus|christ|christian|christians|jehovah|bible|biblical|prayer|prayers|pray|prayed|praying|holy|spirit|spirits|sin|sins|sinned|"
                       r"sinners|sinner|faith|faithful|church|churches|angel|angels|heaven|heavens|heavenly|salvation|saved|gospel|apostle|apostles|disciple|disciples|"
                       r"sabbath|israel|israelites|priest|priests|temple|worship|worshipped|righteous|righteousness|sacrifice|sacrifices|prophet|prophets|psalm|"
                       r"scripture|scriptures|kingdom|congregation|congregations|baptism|baptized|resurrection|verily|unto|thou|thee|thy|thine|hath|shalt|amen|"
                       r"satan|devil|demons?|pharaoh|moses|david|abraham|paul|peter|john|egypt|jerusalem|covenant|blessed|blessing|anoint\w*|glory|soul|souls|"
                       r"elders?|brothers|brothers and sisters|witnesses|witness|elohim|messiah|redeem\w*|repent\w*|forgiv\w*|sanctif\w*|lamb|altar|ark)\b", re.I)

MIN_JOINT = 3            # sentences with both words
MIN_DF_E = 5             # English stem frequency (sentences)
IDF_MIN = 2.3            # general-corpus IDF floor of a content word  (df <= ~10 % of the general sentences)
CAP = 300_000
HELDOUT = 600            # per language, kept out of the training pairs (hash of the English text)


def norm_key(s: str) -> str:
    return re.sub(r"\W+", " ", unicodedata.normalize("NFC", s).lower()).strip()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =============================================================================================== prep
def english_tokens(s: str) -> List[str]:
    return [en_stem(w.group(0).lower().strip("'-’")) for w in _EN_WORD.finditer(s)]


def prep_language(iso: str, raw_dir: str, en_map: Dict[str, str], flores_en: set, out_dir: str, seed: int = 0):
    import pandas as pd
    p = os.path.join(raw_dir, f"{iso}.parquet")
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p).reset_index(drop=True)
    d["en"] = [g[4:] if g.startswith("sft:") else en_map.get(g, "") for g in d["group_key"]]
    d["en"] = d["en"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    d["x"] = d["x"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    n0 = len(d)
    d = d[(d["en"] != "") & (d["x"] != "") & (d["lid"].fillna(-1) != 0)].copy()
    d["enk"] = d["en"].map(norm_key)
    d["xk"] = d["x"].map(norm_key)
    d = d[(d["enk"] != d["xk"]) & ~d["enk"].isin(flores_en)]
    lx, le = d["x"].str.len(), d["en"].str.len()
    d = d[(le >= 12) & (le <= 400) & (lx >= 6) & (lx / le > 0.4) & (lx / le < 2.5)]
    d = d.drop_duplicates(["enk", "xk"])
    d = d.groupby("enk", sort=False).head(2)           # an English sentence is used at most twice (independent contexts)
    d = d.groupby("xk", sort=False).head(2)
    d = d.reset_index(drop=True)
    d["rel"] = (d["prov"] == "religious_jw") | d["en"].map(lambda s: bool(REL_WORDS.search(s))) | d["source"].str.contains("bible", case=False)
    h = d["enk"].map(lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
    # hold-out = whole English sentences (hash order), taken separately from the rel and the nonrel pairs: up to HELDOUT/2 each, at most 10 % of the pairs
    d["held"] = False
    for flag in (False, True):
        sub = d[d["rel"] == flag]
        k = min(HELDOUT // 2, int(0.1 * sub["enk"].nunique()))
        if k > 0:
            first = sub.assign(_h=h.loc[sub.index]).drop_duplicates("enk").sort_values("_h")["enk"].iloc[:k]
            d.loc[sub.index[sub["enk"].isin(set(first))], "held"] = True
    keep = ["en", "x", "rel", "held", "source", "prov"]
    d[keep].to_parquet(os.path.join(out_dir, f"{iso}.parquet"))
    st = {"iso": iso, "raw": n0, "kept": len(d), "rel": int(d["rel"].sum()), "nonrel": int((~d["rel"]).sum()), "held": int(d["held"].sum())}
    return st


def cmd_prep(a):
    import pandas as pd
    import pyarrow.parquet as pq
    os.makedirs(a.out, exist_ok=True)
    en_map = dict(pq.read_table(a.pivot, columns=["group_key", "pivot_en"]).to_pandas().itertuples(index=False, name=None))
    flores_en = set()
    if a.flores_en:
        flores_en = {norm_key(json.loads(l)["text"]) for l in open(a.flores_en, encoding="utf-8") if l.strip()}
    stats = []
    for iso in (a.iso or WEAK):
        s = prep_language(iso, a.raw, en_map, flores_en, a.out)
        if s:
            stats.append(s)
            log(s)
    # general-English statistics: df over sentences (stems) for the non-religious English (+ FLORES dev English) and for the religious English
    gen, rel = collections.Counter(), collections.Counter()
    ngen = nrel = 0
    seen = set()
    for iso in (a.iso or WEAK):
        p = os.path.join(a.out, f"{iso}.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p, columns=["en", "rel"])
        for en, r in zip(d["en"], d["rel"]):
            k = norm_key(en)
            if k in seen:
                continue
            seen.add(k)
            st = set(english_tokens(en))
            if r:
                rel.update(st)
                nrel += 1
            else:
                gen.update(st)
                ngen += 1
    if a.flores_en:
        for l in open(a.flores_en, encoding="utf-8"):
            if l.strip():
                gen.update(set(english_tokens(json.loads(l)["text"])))
                ngen += 1
    json.dump({"n_gen": ngen, "n_rel": nrel, "df_gen": {k: v for k, v in gen.items() if v >= 2 or rel.get(k, 0) >= 5}, "df_rel": {k: v for k, v in rel.items() if v >= 5}},
              open(os.path.join(a.out, "_en_stats.json"), "w"))
    json.dump(stats, open(os.path.join(a.out, "_prep_stats.json"), "w"), indent=1)
    log("general English sentences", ngen, "religious English sentences", nrel)


# =============================================================================================== IBM-1
def build_side(sents: List[List[int]], null: bool):
    """CSR of the unique type ids per sentence (+ NULL id 0 first when null)."""
    flat, ptr = [], [0]
    for s in sents:
        u = sorted(set(s))
        if null:
            u = [0] + u
        flat.extend(u)
        ptr.append(len(flat))
    return np.asarray(flat, dtype=np.int32), np.asarray(ptr, dtype=np.int64)


def expansions(e_flat, e_ptr, f_flat, f_ptr, chunk_pairs=6_000_000):
    """Yield (sent, e_id, f_id, f_pos) arrays of the cross product of the types of each sentence, in chunks."""
    ne = np.diff(e_ptr)
    nf = np.diff(f_ptr)
    sz = ne * nf
    cs = np.concatenate([[0], np.cumsum(sz)])
    n = len(ne)
    s0 = 0
    while s0 < n:
        s1 = int(np.searchsorted(cs, cs[s0] + chunk_pairs, side="right"))
        s1 = max(s1, s0 + 1)
        s1 = min(s1, n)
        idx = np.arange(s0, s1)
        tot = int(cs[s1] - cs[s0])
        sent = np.repeat(idx, sz[idx])
        r = np.arange(tot, dtype=np.int64) - (cs[sent] - cs[s0])
        j = r % nf[sent]
        i = r // nf[sent]
        e_id = e_flat[e_ptr[sent] + i]
        f_pos = f_ptr[sent] + j
        yield sent.astype(np.int32), e_id.astype(np.int64), f_flat[f_pos].astype(np.int64), f_pos
        s0 = s1


def ibm1(src_sents, tgt_sents, n_src, n_tgt, weights, iters=6):
    """IBM Model 1 t(tgt|src) with a NULL source word (id 0). src ids in 1..n_src, tgt ids in 0..n_tgt-1. Returns (keys, t, doc): key = src * n_tgt + tgt
    (sorted, unique), t normalised per src, doc = number of sentences containing both words."""
    s_flat, s_ptr = build_side(src_sents, True)
    t_flat, t_ptr = build_side(tgt_sents, False)
    keys_all, zid_all, sent_all = [], [], []
    for sent, s_id, t_id, t_pos in expansions(s_flat, s_ptr, t_flat, t_ptr):
        keys_all.append(s_id * n_tgt + t_id)
        zid_all.append(t_pos.astype(np.int32))
        sent_all.append(sent)
    keys = np.concatenate(keys_all)
    zid = np.concatenate(zid_all)
    sent = np.concatenate(sent_all)
    del keys_all, zid_all, sent_all
    uk, pid = np.unique(keys, return_inverse=True)
    del keys
    pid = pid.astype(np.int32)
    U = len(uk)
    s_of_u = uk // n_tgt
    wexp = np.asarray(weights, dtype=np.float64)[sent]
    del sent
    nz = int(t_ptr[-1])
    doc = np.bincount(pid, minlength=U)
    t = np.ones(U)
    for it in range(iters):
        tot = np.bincount(s_of_u, weights=t, minlength=n_src + 1)
        t = t / tot[s_of_u]
        tv = t[pid]
        Z = np.bincount(zid, weights=tv, minlength=nz)
        post = tv / Z[zid]
        t = np.bincount(pid, weights=post * wexp, minlength=U)
        del tv, Z, post
    tot = np.bincount(s_of_u, weights=t, minlength=n_src + 1)
    t = t / np.maximum(tot[s_of_u], 1e-30)
    return uk, t, doc


def train_language(iso: str, corpus_dir: str, subset: str, w_rel: float, cap: int, iters: int, stats: dict, seed: int = 0):
    import pandas as pd
    d = pd.read_parquet(os.path.join(corpus_dir, f"{iso}.parquet"))
    d = d[~d["held"]]
    if subset == "rel":
        d = d[d["rel"]]
    elif subset == "nonrel":
        d = d[~d["rel"]]
    if len(d) < 200:
        return None
    nonrel = d[~d["rel"]]
    rel = d[d["rel"]]
    if len(nonrel) >= cap:
        d = nonrel.sample(cap, random_state=seed)
    else:
        d = pd.concat([nonrel, rel.sample(min(len(rel), cap - len(nonrel)), random_state=seed)])
    return train_from_pairs(iso, list(d["en"]), list(d["x"]), [bool(r) for r in d["rel"]], w_rel, iters, stats, extra_meta={"cap": cap, "subset": subset})


def train_from_pairs(iso: str, en_sents: List[str], x_sents: List[str], rel_flags: List[bool], w_rel: float, iters: int, stats: dict, extra_meta: Optional[dict] = None,
                     min_joint: int = MIN_JOINT, min_df_e: int = MIN_DF_E, idf_min: float = IDF_MIN):
    """Lexicon of one language from English / X sentence pairs (lists of equal length). Returns the JSON object (meta, e, v)."""
    info = {"pairs": len(en_sents), "pairs_nonrel": int(sum(not r for r in rel_flags)), "pairs_rel": int(sum(rel_flags))}
    d = {"en": en_sents, "x": x_sents, "rel": np.asarray(rel_flags, dtype=bool)}
    en_tok = [english_tokens(s) for s in d["en"]]
    x_tok = [[fold(t) for t in x_tokens(s)] for s in d["x"]]
    x_tok = [[t for t in s if t] for s in x_tok]
    ok = [(0 < len(e) <= 70 and 0 < len(x) <= 80) for e, x in zip(en_tok, x_tok)]
    keep = np.flatnonzero(ok)
    en_tok = [en_tok[i] for i in keep]
    x_tok = [x_tok[i] for i in keep]
    wts = np.where(d["rel"][keep], w_rel, 1.0)
    # vocabularies on sentence frequency
    dfe = collections.Counter(w for s in en_tok for w in set(s))
    dff = collections.Counter(w for s in x_tok for w in set(s))
    ev = [w for w, c in dfe.items() if c >= 2 and w not in ("",)]
    fv = [w for w, c in dff.items() if c >= 2]
    e2i = {w: i + 1 for i, w in enumerate(sorted(ev, key=lambda w: -dfe[w]))}      # 0 = NULL
    f2i = {w: i for i, w in enumerate(sorted(fv, key=lambda w: -dff[w]))}
    i2e = {i: w for w, i in e2i.items()}
    i2f = {i: w for w, i in f2i.items()}
    e_sents = [[e2i[w] for w in s if w in e2i] for s in en_tok]
    f_sents = [[f2i[w] for w in s if w in f2i] for s in x_tok]
    nonempty = [i for i in range(len(e_sents)) if e_sents[i] and f_sents[i]]
    e_sents = [e_sents[i] for i in nonempty]
    f_sents = [f_sents[i] for i in nonempty]
    wts = wts[nonempty]
    ve, vf = len(e2i), len(f2i)
    info.update({"n_sent": len(e_sents), "en_vocab": ve, "x_vocab": vf})
    log(iso, info)
    t0 = time.time()
    # direction 1: t(f|e) with e in {NULL} + English;   direction 2: t(e|f) with f in X (no NULL on the English side, NULL = id 0 of the "source" = X side)
    uk1, t1, doc1 = ibm1(e_sents, f_sents, ve, vf, wts, iters)
    log(iso, "dir1", len(uk1), f"{time.time()-t0:.0f}s")
    uk2, t2, _ = ibm1([[f + 1 for f in s] for s in f_sents], [[e - 1 for e in s] for s in e_sents], vf, ve, wts, iters)
    log(iso, "dir2", len(uk2), f"{time.time()-t0:.0f}s")
    e1, f1 = uk1 // vf, uk1 % vf
    f2 = uk2 // ve - 1                       # -1 = NULL
    e2 = uk2 % ve + 1
    m2 = f2 >= 0
    k2 = e2[m2] * vf + f2[m2]
    order = np.searchsorted(uk1, k2)
    good = (order < len(uk1)) & (uk1[np.minimum(order, len(uk1) - 1)] == k2)
    tb = np.zeros(len(uk1))
    tb[order[good]] = t2[m2][good]
    nn = e1 > 0
    score = np.sqrt(t1 * tb)
    cand = nn & (doc1 >= min_joint) & (score > 0.02)
    ce, cf, cs, cdoc = e1[cand], f1[cand], score[cand], doc1[cand]
    # top-3 per English type
    o = np.lexsort((-cs, ce))
    ce, cf, cs, cdoc = ce[o], cf[o], cs[o], cdoc[o]
    first = np.concatenate([[0], np.flatnonzero(np.diff(ce)) + 1])
    rank = np.arange(len(ce)) - np.repeat(first, np.diff(np.concatenate([first, [len(ce)]])))
    sel = rank < 3
    ce, cf, cs, cdoc = ce[sel], cf[sel], cs[sel], cdoc[sel]
    # entries
    n_gen, n_rel = stats["n_gen"], stats["n_rel"]
    dfg, dfr = stats["df_gen"], stats["df_rel"]
    ent: Dict[str, list] = {}
    for e, f, s in zip(ce, cf, cs):
        w = i2e[int(e)]
        if len(w) < 3 or w in STOP or not w.isalpha() or dfe[w] < min_df_e:
            continue
        g = dfg.get(w, 0)
        idf = math.log((n_gen + 5) / (g + 5))
        if idf < idf_min:
            continue
        r = ((dfr.get(w, 0) + 1) / (n_rel + 1)) / ((g + 1) / (n_gen + 1))
        bib = int(r >= 6 and dfr.get(w, 0) >= 20)
        ent.setdefault(w, [round(idf, 2), bib]).append([i2f[int(f)], round(float(s), 3)])
    # surface variants of the candidates
    want = {c[0] for v in ent.values() for c in v[2:]}
    var = collections.defaultdict(collections.Counter)
    for s in d["x"]:
        for t in x_tokens(s):
            ft = fold(t)
            if ft in want:
                var[ft][t] += 1
    v = {k: [t for t, _ in c.most_common(2)] for k, c in var.items()}
    info.update({"entries": len(ent), "bib_entries": int(sum(e[1] for e in ent.values())), "seconds": round(time.time() - t0)})
    return {"meta": {"iso": iso, "built": time.strftime("%Y-%m-%d"), "algo": "ibm1-sym-geo", "iters": iters, "w_rel": w_rel, "min_joint": min_joint,
                     "min_df_e": min_df_e, "idf_min": idf_min, **(extra_meta or {}), **info, "validated": False, "weight": 0.0, "match": "exact"},
            "e": ent, "v": v}


def cmd_train(a):
    stats = json.load(open(os.path.join(a.corpus, "_en_stats.json")))
    os.makedirs(a.out, exist_ok=True)
    for iso in (a.iso or WEAK):
        if not os.path.exists(os.path.join(a.corpus, f"{iso}.parquet")):
            continue
        res = train_language(iso, a.corpus, a.subset, a.w_rel, a.cap, a.iters, stats)
        if res is None:
            log(iso, "too little data")
            continue
        p = os.path.join(a.out, f"{iso}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, separators=(",", ":"))
        log(iso, "wrote", p, os.path.getsize(p) // 1024, "KB", {k: res["meta"][k] for k in ("entries", "bib_entries", "pairs", "pairs_nonrel")})


# =============================================================================================== finalize (validation record into the lexicon files)
MIN_AUC = 0.75           # FLORES+ dev, human reference vs reference of another sentence (affix match), pooled over 5 shifts
MIN_COVERAGE = 0.30      # mean (covered / content words) over the FLORES+ dev sources
MIN_ATTRACTOR = 0.70     # model outputs: AUC of own-source output vs output of ANOTHER source (the gaming check), when samples exist
MAX_WEIGHT = 0.25


def proposed_weight(auc: float, cov: float) -> float:
    return round(MAX_WEIGHT * min(max((auc - 0.70) / 0.20, 0.0), 1.0) * min(1.0, cov / 0.5), 2)


def cmd_finalize(a):
    val = json.load(open(a.validation))
    summary = {}
    for fn in sorted(os.listdir(a.lexicon)):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        iso = fn[:-5]
        p = os.path.join(a.lexicon, fn)
        obj = json.load(open(p, encoding="utf-8"))
        m = obj["meta"]
        e = val.get(iso, {})
        f = e.get("flores", {}).get("affix")
        m["match"] = "affix"
        m.update({"validated": False, "weight": 0.0})
        rec = {"criteria": {"min_auc_vs_mismatched": MIN_AUC, "min_coverage": MIN_COVERAGE, "min_attractor_auc": MIN_ATTRACTOR, "max_weight": MAX_WEIGHT}}
        if not f or "auc_vs_mismatched" not in f:
            rec.update({"status": "not_validated", "reason": "no FLORES+ dev human reference for this language" if not f else "too few scored FLORES+ sources"})
        else:
            auc, cov = f["auc_vs_mismatched"], f["coverage"]
            att = e.get("model", {}).get("affix", {}).get("auc_own_vs_other_source_output")
            rec.update({"flores_auc_vs_mismatched": round(auc, 3), "flores_auc_vs_copy": round(f["auc_vs_copy"], 3), "flores_auc_vs_half_dropped": round(f["auc_vs_half_dropped"], 3),
                        "flores_coverage": round(cov, 3), "flores_share_scored": round(f["share_scored"], 3), "attractor_auc": None if att is None else round(att, 3)})
            ok = auc >= MIN_AUC and cov >= MIN_COVERAGE and (att is None or att >= MIN_ATTRACTOR)
            rec["status"] = "validated" if ok else "failed"
            if not ok:
                rec["reason"] = "; ".join(r for r, c in (("auc %.3f < %.2f" % (auc, MIN_AUC), auc < MIN_AUC), ("coverage %.2f < %.2f" % (cov, MIN_COVERAGE), cov < MIN_COVERAGE),
                                                         ("attractor auc low", att is not None and att < MIN_ATTRACTOR)) if c)
            else:
                m["validated"] = True
                m["weight"] = proposed_weight(auc, cov)
                m["density_max"] = round(e.get("density_max", 0.5), 3)
        m["validation"] = rec
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))
        summary[iso] = {k: rec.get(k) for k in ("status", "flores_auc_vs_mismatched", "flores_coverage", "reason")} | {"weight": m["weight"]}
    for k, v in summary.items():
        print(k, v)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--raw", default="corpus_raw")
    p.add_argument("--pivot", default="dl/full_pairs/pivot_en_sentences.parquet")
    p.add_argument("--out", default="corpus")
    p.add_argument("--flores-en", default="", help="FLORES+ dev eng_Latn.jsonl: its sentences are removed from the training pairs and enter the general-English statistics")
    p.add_argument("--iso", nargs="*")
    p.set_defaults(fn=cmd_prep)
    p = sub.add_parser("train")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--out", default="lexicon")
    p.add_argument("--iso", nargs="*")
    p.add_argument("--subset", default="all", choices=("all", "rel", "nonrel"))
    p.add_argument("--w-rel", type=float, default=0.3)
    p.add_argument("--cap", type=int, default=CAP)
    p.add_argument("--iters", type=int, default=6)
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("finalize", help="write the validation record (validated / weight / match / density_max) of validate_lexicons.py's output into the lexicon files")
    p.add_argument("--lexicon", default="lexicon")
    p.add_argument("--validation", required=True)
    p.set_defaults(fn=cmd_finalize)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
