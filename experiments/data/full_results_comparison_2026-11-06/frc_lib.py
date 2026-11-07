"""Shared loaders and vectorised metrics for the full results comparison (2026-11-06).  CPU only, laptop, python with numpy (+ sacrebleu for the checks).

Data sources (all read-only, all on the laptop):
  ours      : CACHE/<system>.npz made by s01_rescore.py (per-sentence chrF++/spBLEU statistics recomputed from the stored predictions)
              + per-sentence held-out AfriCOMET-STL-1.1 (reference-based) from the evidence npz files of the 385M evaluation (stats joined by (benchmark, iso, id))
  NLLB/TPsy : research_archive_2026-10-23/box/jobs/e2/scores/<name>.pkl.gz (per-sentence chrF++/spBLEU statistics, FLORES+ devtest, 1,012 rows per language),
              <name>.comet.pkl (AfriCOMET-STL-1.1 reference-based per sentence, rows with pair_seen == False only), <name>.ssa.pkl (SSA-COMET-MTL reference-based)
  NLLB aggregates on NTREX / MAFAND / SmolSent / SALT: reports/data/nllb600m_per_language_benchmark.csv (corpus-level, per language, per row subset)
  row flags : research_archive_2026-11-03/eval_385m/old_flags (flags of the first research's 270M SFT text) and
              research_archive_2026-11-03/eval_385m/contamination_385m_2026-11-03 (flags of the 385M SFT text)
"""
import csv
import gzip
import json
import os
import pickle
import zlib

import numpy as np

REPO = "/path/to/repo/"
ARC = REPO + "research_archive_2026-11-03/"
ARC1 = REPO + "research_archive_2026-10-23/"
E2 = ARC1 + "box/jobs/e2/scores/"
EVID = ARC + "box_final_2026-11-04/"

# system -> evidence npz holding the stored per-sentence AfriCOMET (same predictions as the cache of s01_rescore.py)
OURS = {
    "sft385":  EVID + "shm_results/evid_e13/sft.npz",
    "stageA":  EVID + "shm_results/evid_e20/stageA.npz",
    "b1best":  EVID + "shm_results/evid_e20/b1.npz",
    "mixed":   EVID + "shm_results/evid_e20/mixed.npz",
    "e03c1":   EVID + "e03_control/run_records_and_eval/c1_s6000.npz",
    "old_sft": EVID + "shm_results/evid_e13/old_sft.npz",
    "old_rl":  EVID + "shm_results/evid_e13/old_rl2c.npz",
}
E2_SYSTEMS = {"nllb600m": "nllb600m", "nllb3b3": "nllb3b3", "tpsy08": "tpsy08", "tpsy2b": "tpsy2b"}

LABEL = {
    "sft385": "385M SFT",
    "e03c1": "385M SFT-only control (E-03, +TranslatePsy)",
    "stageA": "385M RL stage A",
    "b1best": "385M RL B1 best",
    "mixed": "385M RL (mixed best = Hub main)",
    "old_sft": "270M SFT (earlier, E-13 rerun)",
    "old_rl": "270M RL-final (earlier, E-13 rerun)",
    "nllb600m": "NLLB-200-distilled-600M",
    "nllb3b3": "NLLB-200-3.3B",
    "tpsy08": "TranslatePsy-AfriSLM-0.8B",
    "tpsy2b": "TranslatePsy-AfriSLM-2B",
}
SHORT = {
    "sft385": "SFT", "e03c1": "SFT-ctrl", "stageA": "RL-A", "b1best": "RL-B1", "mixed": "RL", "old_sft": "270M SFT", "old_rl": "270M RL",
    "nllb600m": "NLLB-600M", "nllb3b3": "NLLB-3.3B", "tpsy08": "TPsy-0.8B", "tpsy2b": "TPsy-2B",
}
EPS = 1e-16
_BLEU = None


# ------------------------------------------------------------------------------------------------ metrics
def fscore(stats, order):
    """sacrebleu CHRF._compute_f_score (beta 2, default non-eps smoothing), vectorised.  stats [..., 3*order] (hyp, ref, match) per order."""
    s = np.asarray(stats, dtype=np.float64)[..., :3 * order]
    h, r, m = s[..., 0::3], s[..., 1::3], s[..., 2::3]
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(h > 0, m / np.where(h > 0, h, 1), EPS)
        rec = np.where(r > 0, m / np.where(r > 0, r, 1), EPS)
    eff = (h > 0) & (r > 0)
    n = eff.sum(-1)
    nn = np.where(n > 0, n, 1)
    ap = (prec * eff).sum(-1) / nn
    ar = (rec * eff).sum(-1) / nn
    f = 4.0
    den = f * ap + ar
    sc = np.where(ap + ar > 0, 100.0 * (1 + f) * ap * ar / np.where(den > 0, den, 1), 0.0)
    return sc


def chrfpp(stats):          # word order 2: 8 orders
    return fscore(stats, 8)


def chrf(stats):            # word order 0: the first 6 char orders
    return fscore(stats[..., :18], 6)


def spbleu_from_stats(st):
    """sacrebleu BLEU score from summed segment statistics (10 numbers: hyp_len, ref_len, then 4 x (correct, total)); exp smoothing as the default."""
    global _BLEU
    if _BLEU is None:
        from sacrebleu.metrics import BLEU
        _BLEU = BLEU(tokenize="flores200")
    return _BLEU._compute_score_from_stats([int(x) for x in st]).score


# ------------------------------------------------------------------------------------------------ loading
class Sys:
    """Per-sentence data of one system; arrays keyed by (benchmark, iso) -> dict(ids, chrf, bleu, comet, idx)."""

    def __init__(self, name):
        self.name = name
        self.d = {}

    def add(self, bench, iso, ids, chrf_st, bleu_st, comet=None):
        ids = np.asarray(ids).astype(str)
        self.d[(bench, iso)] = dict(ids=ids, chrf=np.asarray(chrf_st), bleu=None if bleu_st is None else np.asarray(bleu_st),
                                    comet=np.full(len(ids), np.nan) if comet is None else np.asarray(comet, dtype=np.float64),
                                    idx={x: i for i, x in enumerate(ids)})

    def has(self, bench, iso):
        return (bench, iso) in self.d

    def langs(self, bench):
        return sorted({k[1] for k in self.d if k[0] == bench})

    def get(self, bench, iso, ids):
        e = self.d[(bench, iso)]
        try:
            ix = np.fromiter((e["idx"][i] for i in ids), dtype=np.int64, count=len(ids))
        except KeyError:
            return None
        return e["chrf"][ix], (None if e["bleu"] is None else e["bleu"][ix]), e["comet"][ix]


def load_ours(name, cache):
    z = np.load(os.path.join(cache, name + ".npz"), allow_pickle=True)
    ev = np.load(OURS[name], allow_pickle=True)
    e_bench, e_iso, e_rid, e_comet, e_stats = ev["bench"], ev["iso"], ev["rid"].astype(str), ev["comet"], ev["stats"]
    ek = {(b, i, r): c for b, i, r, c in zip(e_bench, e_iso, e_rid, e_comet)}
    s = Sys(name)
    bench, iso, rid = z["bench"], z["iso"], z["rid"].astype(str)
    chrf_st, bleu_st = z["chrf_st"], z["bleu_st"]
    comet = np.array([ek.get((b, i, r), np.nan) for b, i, r in zip(bench, iso, rid)], dtype=np.float64)
    keys = {}
    for j, (b, i) in enumerate(zip(bench, iso)):
        keys.setdefault((b, i), []).append(j)
    for (b, i), js in keys.items():
        js = np.array(js)
        s.add(b, i, rid[js], chrf_st[js], bleu_st[js], comet[js])
    # integrity check against the stored evidence statistics (same hypotheses): chrF++ stats of the evidence file are in ev["stats"]
    st_ev = {(b, i, r): k for k, (b, i, r) in enumerate(zip(e_bench, e_iso, e_rid))}
    chk = 0
    mism = 0
    for j in range(0, len(rid), 37):
        k = st_ev.get((bench[j], iso[j], rid[j]))
        if k is None:
            continue
        chk += 1
        mism += int(not np.array_equal(e_stats[k], chrf_st[j]))
    s.check = (chk, mism)
    return s


def load_e2(name):
    S = pickle.load(gzip.open(E2 + name + ".pkl.gz", "rb"))
    C = pickle.load(open(E2 + name + ".comet.pkl", "rb"))
    s = Sys(name)
    s.ssa = pickle.load(open(E2 + name + ".ssa.pkl", "rb"))
    s.meta = {}
    for iso, d in S.items():
        cm = C.get(iso, {})
        comet = np.array([cm.get(i, np.nan) for i in d["ids"]], dtype=np.float64)
        s.add("flores_plus", iso, d["ids"], d["chrf_st"].astype(np.int64), d["bleu_st"].astype(np.int64), comet)
        s.meta[iso] = dict(src=d["src"], ref=d["ref"], hyp=d["hyp"], pair_seen=d["pair_seen"], contaminated=d["contaminated"])
    return s


def load_flags(bench, which):
    """-> {(iso, id): True if CLEAN (old: English not in the 270M SFT text; 385: not flagged by the 385M audit)}, and {(iso, id): True if pair NOT seen}."""
    p = (ARC + "eval_385m/old_flags/contamination_%s.json" if which == "old" else ARC + "eval_385m/contamination_385m_2026-11-03/contamination_%s.json") % bench
    d = json.load(open(p))
    clean, nopair = {}, {}
    for k, v in d["exact"].items():
        iso, rid = k.split(":", 1)
        clean[(iso, rid)] = not v
        nopair[(iso, rid)] = not d["pair"].get(k, False)
    return clean, nopair


def nllb_aggregates():
    """Corpus-level per-language chrF++/spBLEU of NLLB-3.3B beam 4 (and 600M on FLORES+) on row subsets all / no_pair / clean (flags of the 270M SFT text)."""
    out = {}
    for r in csv.DictReader(open(REPO + "reports/data/nllb600m_per_language_benchmark.csv")):
        def f(x):
            return float(x) if x not in ("", None) else np.nan
        out[(r["benchmark"], r["subset"], r["iso"])] = dict(n=int(r["n"]), n33_chrfpp=f(r["nllb_n33_beam4_chrf"]), n33_spbleu=f(r["nllb_n33_beam4_spbleu"]),
                                                          n600_chrfpp=f(r["nllb_n600_beam4_chrf"]), n600_spbleu=f(r["nllb_n600_beam4_spbleu"]),
                                                          ours270_chrfpp=f(r["ours_chrf"]))
    return out


def weights(bench, iso, ids, B, seed, _cache={}):
    """Poisson(1) bootstrap weights per source sentence (n, B) uint8.  Key = (bench, id), shared by all languages for the parallel benchmarks
    (FLORES+, NTREX); MAFAND, SmolSent etc. are not parallel, so their key also holds the language."""
    par = bench in ("flores_plus", "ntrex")
    out = np.empty((len(ids), B), dtype=np.uint8)
    for j, i in enumerate(ids):
        k = (bench, "" if par else iso, str(i), B, seed)
        w = _cache.get(k)
        if w is None:
            rng = np.random.default_rng([seed, zlib.crc32(bench.encode()), zlib.crc32(("" if par else iso).encode()), zlib.crc32(str(i).encode())])
            w = rng.poisson(1.0, B).astype(np.uint8)
            _cache[k] = w
        out[j] = w
    return out
