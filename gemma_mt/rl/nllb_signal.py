# SPDX-License-Identifier: AGPL-3.0-or-later
"""NLLB-200 as a REWARD signal (R14, 2026-10-22; OWNER: reward use only here -- NLLB output is CC-BY-NC, files are marked `reward_only_nc`).

Two signals, both OFF by default (`reward.nllb_signal: false`), both per language and gated by a human-validated calibration:

  (A) PSEUDO-REFERENCE  chrF++(student chunk, NLLB translation of the chunk's source sentence), precomputed OFFLINE per RL prompt
      (scripts/precompute_nllb_pseudorefs.py -> one parquet per language, rows keyed like the RL archive: sha1(iso + "\\n" + NFC(source))[:16]).
      A consensus-type signal: it needs no model call at reward time (a dict lookup + a sentence chrF++). The value is shaped with the language's
      anchors (human chrF++ against the NLLB translation: P(mismatched, 90) -> 0, human P50 -> +1, human P90 -> +2), the same curve as every other content signal.
  (B) FORCED-DECODING SCORE  length-normalised log-probability of the student's chunk under NLLB given the source (a reference-free, Prism-style score). Needs
      one NLLB forward pass per reward call (`ForcedDecodingScorer`, CTranslate2 `score_batch`); cost is measured in reports/nllb600m_teacher_2026-10-22.md.

Where the teacher is not clearly better than the student on human references (calibration `teacher_vs_student`), or the signal does not separate human
translations from corrupted ones (AUC), the language's weight is 0: the signal is never applied there.

The reward integration is ONE clearly marked hook (see `reward_hook_patch.md` in reports/): after the content value `r` of a chunk is known,
    r = nllb.blend_chunk(lang, seg_src, seg_mt, r, mode)            # no-op unless cfg.nllb_signal and the language has a weight
Nothing else of rl/reward.py changes. Sentence segmentation is rl/segments.split_sentences (the reward's own), so a chunk (one sentence, or several merged
by the chunk plan) maps to the NLLB translations of ITS sentences.
"""

from __future__ import annotations

import json
import os
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from gemma_mt.rl import consensus as cons
from gemma_mt.rl.archive import make_key
from gemma_mt.rl.reward_shaping import auc as _auc, make_curve, reliability_from_auc
from gemma_mt.rl.segments import split_sentences

REWARD_ONLY_NC = "reward_only_nc"
TEACHER_DEFAULT = "nllb-200-3.3B"
SIGNAL_KEY = "nllb"                                   # per-language block of the calibration file: {"pseudoref": {...}, "forced": {...}, "teacher_vs_student": {...}}

# our tag -> NLLB / FLORES-200 code (exact language) ; proxies are NOT listed (a proxy is not a validated signal). Source of truth with notes:
# gemma_mt/scripts/r14/nllb_langmap.py (checked against the 202 language tokens of the NLLB tokenizer).
NLLB_CODE: Dict[str, str] = {
    "yor": "yor_Latn", "hau": "hau_Latn", "ibo": "ibo_Latn", "twi": "twi_Latn", "ewe": "ewe_Latn", "fon": "fon_Latn", "wol": "wol_Latn", "swh": "swh_Latn",
    "kin": "kin_Latn", "som": "som_Latn", "orm": "gaz_Latn", "amh": "amh_Ethi", "tir": "tir_Ethi", "lug": "lug_Latn", "lin": "lin_Latn", "zul": "zul_Latn",
    "xho": "xho_Latn", "afr": "afr_Latn", "sna": "sna_Latn", "nya": "nya_Latn", "sot": "sot_Latn", "tsn": "tsn_Latn", "mlg": "plt_Latn", "bem": "bem_Latn",
    "tum": "tum_Latn", "run": "run_Latn", "tso": "tso_Latn", "nso": "nso_Latn", "ssw": "ssw_Latn", "sag": "sag_Latn", "umb": "umb_Latn", "kmb": "kmb_Latn",
    "mos": "mos_Latn", "kik": "kik_Latn", "luo": "luo_Latn", "kam": "kam_Latn", "dyu": "dyu_Latn", "bam": "bam_Latn", "kbp": "kbp_Latn", "kon": "kon_Latn",
    "kea": "kea_Latn", "knc": "knc_Latn", "kab": "kab_Latn", "nus": "nus_Latn", "lua": "lua_Latn", "tzm": "tzm_Tfng",
}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def key_of(iso: str, source: str) -> str:
    """The RL archive key of a source (rl/archive.make_key)."""
    return make_key(iso, source)


def chrf01(hyp: str, ref: str) -> float:
    """Sentence chrF++ in [0, 1] (rl/consensus.chrf_sentence is verified equal to sacrebleu CHRF(word_order=2).sentence_score)."""
    return cons.chrf_sentence(hyp, ref) / 100.0


# ------------------------------------------------------------------------------------------------ configuration
@dataclass
class NllbSignalConfig:
    """Mirrors the `reward:` yaml keys (all optional, off by default). RewardConfig gets the same field names (see reports/nllb600m_teacher_2026-10-22.md, section 7)."""
    nllb_signal: bool = False                       # master switch of (A); False = reward unchanged
    nllb_pseudoref_dir: str = ""                    # directory with <iso>.parquet written by scripts/precompute_nllb_pseudorefs.py
    nllb_calibration_path: str = ""                 # JSON with a `nllb` block per language (scripts/calibrate_nllb_signal.py); no file -> no weights -> no effect
    nllb_modes: List[str] = field(default_factory=lambda: ["consensus", "blend"])     # content modes the signal applies to ("qe" languages only if listed in nllb_languages)
    nllb_languages: List[str] = field(default_factory=list)            # extra languages (any mode) the signal applies to
    nllb_weight_default: float = 0.35               # w = nllb_weight_default * reliability(lang), capped at nllb_weight_max
    nllb_weight_max: float = 0.5
    nllb_weights: Dict[str, float] = field(default_factory=dict)       # per-language override of w (0 disables the language)
    nllb_double_pull_languages: List[str] = field(default_factory=list)   # languages whose SFT mixture also contains NLLB teacher data (teacher_v1 `nc_teacher`): the student is already pulled toward NLLB there
    nllb_double_pull_factor: float = 0.5            # weight multiplier for those languages (applied to the calibrated weight, not to an explicit nllb_weights entry)
    nllb_min_teacher_delta: float = 2.0             # teacher chrF++ - student chrF++ on human references must be >= this (calibration teacher_vs_student.delta)
    nllb_min_auc: float = 0.80                      # AUC(human > mismatched) of the pseudo-reference chrF++ must be >= this
    nllb_forced: bool = False                       # (B) forced-decoding score; needs nllb_ct2_dir. Combined with (A) as min(shaped A, shaped B): B alone prefers copies of the English source
    nllb_forced_combine: str = "min"                # min | mean of the two shaped values (min is the copy-safe choice, see the hack study)
    nllb_forced_min_auc: float = 0.80               # the forced block of a language is used only if its dev AUC (human > mismatched) is at least this
    nllb_ct2_dir: str = ""
    nllb_hf_dir: str = ""
    nllb_ct2_compute_type: str = "int8_float16"
    nllb_forced_batch_tokens: int = 4096

    @classmethod
    def from_reward_cfg(cls, cfg: Any) -> "NllbSignalConfig":
        base = cls()
        kw = {k: getattr(cfg, k) for k in base.__dict__ if hasattr(cfg, k)}
        return cls(**{**base.__dict__, **kw})


# ------------------------------------------------------------------------------------------------ pseudo-reference store
class PseudoRefStore:
    """Per-language NLLB translations. Rows: key, iso, source (the RL prompt window), sentences, translations (aligned lists). `lookup(iso, text)` returns the pseudo-reference
    of ANY text made of sentences the store has seen (a whole window, a chunk, a single sentence); None when one of its sentences is unknown."""

    def __init__(self, directory: str, languages: Optional[Iterable[str]] = None):
        self.dir = directory
        self.languages = set(languages) if languages is not None else None
        self._win: Dict[str, Dict[str, List[str]]] = {}      # iso -> window key -> translations
        self._sent: Dict[str, Dict[str, str]] = {}           # iso -> sentence key -> translation
        self._loaded: set = set()

    def available(self) -> List[str]:
        if not os.path.isdir(self.dir):
            return []
        return sorted(f[:-8] for f in os.listdir(self.dir) if f.endswith(".parquet"))

    def load(self, iso: str) -> bool:
        if iso in self._loaded:
            return iso in self._sent
        self._loaded.add(iso)
        path = os.path.join(self.dir, f"{iso}.parquet")
        if not os.path.exists(path) or (self.languages is not None and iso not in self.languages):
            return False
        import pyarrow.parquet as pq

        t = pq.read_table(path, columns=["key", "sentences", "translations"]).to_pydict()
        win, sent = {}, {}
        for k, ss, tt in zip(t["key"], t["sentences"], t["translations"]):
            win[k] = list(tt)
            for s, x in zip(ss, tt):
                sent[key_of(iso, s)] = x
        self._win[iso], self._sent[iso] = win, sent
        return True

    def add(self, iso: str, source: str, sentences: Sequence[str], translations: Sequence[str]) -> None:
        """In-memory insertion (tests, on-the-fly use)."""
        self._loaded.add(iso)
        self._win.setdefault(iso, {})[key_of(iso, source)] = list(translations)
        d = self._sent.setdefault(iso, {})
        for s, x in zip(sentences, translations):
            d[key_of(iso, s)] = x

    def lookup(self, iso: str, text: str) -> Optional[str]:
        if not self.load(iso) and iso not in self._sent:
            return None
        w = self._win[iso].get(key_of(iso, text))
        if w is not None and all(w):
            return " ".join(w)
        d = self._sent[iso]
        out = []
        for s in split_sentences(nfc(text)):
            x = d.get(key_of(iso, s))
            if not x:                                          # unknown sentence, or one the precompute rejected (empty / copy / loop): no pseudo-reference for this chunk
                return None
            out.append(x)
        return " ".join(out) if out else None


# ------------------------------------------------------------------------------------------------ calibration (human-validated)
def calibrate_scores(human: Sequence[float], mismatched: Sequence[float], floor: float = 0.0) -> Dict[str, Any]:
    """Anchors (t_fail, t_mid, t_great) and reliability of a [0, 1]-scaled signal from scores of human references and of MISMATCHED translations (different sentence of the same
    language): t_fail = P90 of mismatched, t_mid = P50 and t_great = P90 of human; reliability = linear in AUC(human > mismatched) (0.65 -> 0, 0.90 -> 1)."""
    h, m = np.asarray(human, float), np.asarray(mismatched, float)
    t_fail = max(floor, float(np.quantile(m, 0.9))) if len(m) else floor
    q = np.quantile(h, [0.5, 0.9])
    t_mid = max(float(q[0]), t_fail + 0.02)
    t_great = max(float(q[1]), t_mid + 0.02)
    a = _auc(h, m)
    return {"anchors": [t_fail, t_mid, t_great], "auc_vs_mismatched": a, "reliability": reliability_from_auc(a), "n": int(len(h))}


def shape01(c: float, anchors: Sequence[float], neg_scale: float = 2.0, preset: str = "repo") -> float:
    """The reward curve (-2 .. +2.25) for a signal in [0, 1] with NO AfriCOMET floor (the 0.46 floor belongs to the AfriCOMET scale)."""
    xs, ys = make_curve(anchors, neg_scale, preset, floor=0.0)
    return float(np.interp(c, xs, ys))


# ------------------------------------------------------------------------------------------------ the signal used by the reward
class NllbSignal:
    def __init__(self, cfg: NllbSignalConfig, store: Optional[PseudoRefStore] = None, calibration: Optional[Dict[str, Any]] = None,
                 forced: Optional["ForcedDecodingScorer"] = None):
        self.cfg = cfg
        self.store = store if store is not None else PseudoRefStore(cfg.nllb_pseudoref_dir)
        self.calib: Dict[str, dict] = calibration if calibration is not None else self._read_calibration(cfg.nllb_calibration_path)
        self.forced = forced
        self._fd_cache: Dict[Tuple[str, str, str], float] = {}
        self.stats = {"chunks": 0, "applied": 0, "no_pseudoref": 0, "seconds": 0.0, "forced_pairs": 0}
        self.lang_counts: Dict[str, List[int]] = {}      # D-67: per language [chunks that had a weight, chunks with a pseudo-reference] since the last pop_stats

    @staticmethod
    def _read_calibration(path: str) -> Dict[str, dict]:
        if not path or not os.path.exists(path):
            return {}
        data = json.load(open(path, encoding="utf-8"))
        return {k.split(">")[-1].strip(): v[SIGNAL_KEY] for k, v in data.items() if isinstance(v, dict) and isinstance(v.get(SIGNAL_KEY), dict)}

    # -- which languages and with which weight
    def weight(self, lang: str, mode: str = "consensus") -> float:
        cfg = self.cfg
        if not cfg.nllb_signal:
            return 0.0
        if mode not in cfg.nllb_modes and lang not in cfg.nllb_languages:
            return 0.0
        if lang in cfg.nllb_weights:
            return max(0.0, float(cfg.nllb_weights[lang]))
        blk = self.calib.get(lang) or {}
        pr = blk.get("pseudoref") or {}
        tv = blk.get("teacher_vs_student") or {}
        if not pr.get("anchors") or not blk.get("use", True):
            return 0.0
        if float(tv.get("delta", -99.0)) < cfg.nllb_min_teacher_delta:
            return 0.0                                              # the teacher is not clearly better than the student here: never pull the student toward it
        if float(pr.get("auc_vs_mismatched", 0.0)) < cfg.nllb_min_auc:
            return 0.0
        # weight = default x AUC reliability x weight_factor (calibration: how far the teacher is above the student, and the hack-study failure rate of the language)
        w = min(cfg.nllb_weight_max, cfg.nllb_weight_default * float(pr.get("reliability", 0.0)) * float(pr.get("weight_factor", 1.0)))
        return w * cfg.nllb_double_pull_factor if lang in cfg.nllb_double_pull_languages else w

    def value(self, lang: str, src: str, mt: str) -> Optional[float]:
        """Shaped reward-scale value of the chunk, or None (no pseudo-reference for this source / no anchors)."""
        pr = (self.calib.get(lang) or {}).get("pseudoref") or {}
        if not pr.get("anchors"):
            return None
        ref = self.store.lookup(lang, src)
        if ref is None:
            return None
        return shape01(chrf01(mt, ref), pr["anchors"])

    def _forced_ok(self, lang: str) -> bool:
        fb = (self.calib.get(lang) or {}).get("forced") or {}
        return bool(self.cfg.nllb_forced and self.forced is not None and lang in NLLB_CODE and fb.get("anchors")
                    and float(fb.get("auc_vs_mismatched", 0.0)) >= self.cfg.nllb_forced_min_auc)

    def prepare(self, items: Sequence[Tuple[str, str, str, str]]) -> None:
        """Batch the forced-decoding forward passes of one reward call. items: (lang, src, mt, mode). Cheap no-op when (B) is off or nothing needs it."""
        if not (self.cfg.nllb_forced and self.forced is not None):
            return
        by: Dict[str, List[Tuple[str, str]]] = {}
        for lang, src, mt, mode in items:
            if self.weight(lang, mode) > 0 and self._forced_ok(lang) and self.store.lookup(lang, src) is not None and (lang, src, mt) not in self._fd_cache:
                by.setdefault(lang, []).append((src, mt))
        t0 = time.time()
        for lang, pairs in by.items():
            uniq = list(dict.fromkeys(pairs))
            for lp, (src, mt) in zip(self.forced.score(lang, [a for a, _ in uniq], [b or "." for _, b in uniq]), uniq):
                self._fd_cache[(lang, src, mt)] = lp
            self.stats["forced_pairs"] += len(uniq)
        self.stats["seconds"] += time.time() - t0
        if len(self._fd_cache) > 50000:
            self._fd_cache.clear()

    def forced_value(self, lang: str, src: str, mt: str) -> Optional[float]:
        if not self._forced_ok(lang):
            return None
        k = (lang, src, mt)
        lp = self._fd_cache.get(k)
        if lp is None:
            lp = self.forced.score(lang, [src], [mt or "."])[0]
            self._fd_cache[k] = lp
            self.stats["forced_pairs"] += 1
        return shape01(forced_to_unit(lp), self.calib[lang]["forced"]["anchors"])

    def blend_chunk(self, lang: str, src: str, mt: str, r: float, mode: str = "consensus") -> float:
        """THE HOOK: (1 - w) * r + w * nllb_value. Identity when the signal is off, the language has weight 0 or no pseudo-reference exists.
        nllb_value = shaped pseudo-reference chrF++ (A); with (B) enabled and calibrated for the language: min (or mean) of the shaped A and the shaped forced-decoding score."""
        w = self.weight(lang, mode)
        if w <= 0.0:
            return r
        t0 = time.time()
        self.stats["chunks"] += 1
        lc = self.lang_counts.setdefault(lang, [0, 0])
        lc[0] += 1
        v = self.value(lang, src, mt)
        if v is not None and self._forced_ok(lang):
            fv = self.forced_value(lang, src, mt)
            if fv is not None:
                v = min(v, fv) if self.cfg.nllb_forced_combine == "min" else 0.5 * (v + fv)
        self.stats["seconds"] += time.time() - t0
        if v is None:
            self.stats["no_pseudoref"] += 1
            return r
        self.stats["applied"] += 1
        lc[1] += 1
        return (1.0 - w) * r + w * v

    def pop_stats(self) -> Dict[str, float]:
        """D-67: coverage of the pseudo-references per language (share of the chunks of a weighted language that had one) for the step's reward call, then reset. Prompts without a pseudo-reference are skipped silently."""
        out: Dict[str, float] = {}
        tot = app = 0
        for lang, (n, a) in sorted(self.lang_counts.items()):
            out[f"nllb/coverage_{lang}"] = a / n if n else 0.0
            out[f"nllb/chunks_{lang}"] = float(n)
            tot += n; app += a
        out["nllb/applied_rate"] = app / tot if tot else 0.0
        out["nllb/chunks"] = float(tot)
        self.lang_counts = {}
        return out


def forced_to_unit(mean_logprob: float, lo: float = -6.0, hi: float = 0.0) -> float:
    """Mean per-token log-probability (nats) -> [0, 1] (linear on [lo, hi], clipped): -6 nats/token or worse -> 0, 0 -> 1."""
    return float(min(1.0, max(0.0, (mean_logprob - lo) / (hi - lo))))


# ------------------------------------------------------------------------------------------------ NLLB (CTranslate2) wrappers
class NllbCT2:
    """Sentence-level NLLB with CTranslate2. `translate` (beam search) for the precompute, `score` (forced decoding) for signal (B). Needs ctranslate2 + transformers (lazy)."""

    def __init__(self, ct2_dir: str, hf_dir: str, device: str = "cuda", compute_type: str = "int8_float16", intra_threads: int = 4):
        import ctranslate2
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(hf_dir, src_lang="eng_Latn")
        self.tr = ctranslate2.Translator(ct2_dir, device=device, compute_type=compute_type, intra_threads=intra_threads)

    def _enc(self, texts: Sequence[str], max_len: int = 256) -> List[List[str]]:
        return [self.tok.convert_ids_to_tokens(self.tok.encode(t, truncation=True, max_length=max_len)) for t in texts]

    def translate(self, texts: Sequence[str], iso: str, beam: int = 4, no_repeat_ngram: int = 4, max_batch_tokens: int = 4096, max_len: int = 256,
                  return_scores: bool = False):
        tgt = NLLB_CODE[iso]
        src = self._enc(texts)
        kw: Dict[str, Any] = dict(beam_size=beam)
        if no_repeat_ngram:
            kw["no_repeat_ngram_size"] = no_repeat_ngram
        res = self.tr.translate_batch(src, target_prefix=[[tgt]] * len(src), max_batch_size=max_batch_tokens, batch_type="tokens", max_decoding_length=max_len,
                                      return_scores=True, **kw)
        hyps, scores = [], []
        for r in res:
            toks = r.hypotheses[0]
            toks = toks[1:] if toks and toks[0] == tgt else toks
            hyps.append(self.tok.decode(self.tok.convert_tokens_to_ids(toks), skip_special_tokens=True).strip())
            scores.append(r.scores[0] if r.scores else float("nan"))
        return (hyps, scores) if return_scores else hyps

    def score(self, iso: str, srcs: Sequence[str], mts: Sequence[str], max_batch_tokens: int = 4096) -> List[float]:
        """Mean per-token log-probability of each `mt` given `src` under NLLB with the language prefix (forced decoding; the tokens scored are the translation tokens + </s>)."""
        tgt = NLLB_CODE[iso]
        s = self._enc(srcs)
        t = []
        for m in mts:
            ids = self.tok.encode(m, add_special_tokens=False, truncation=True, max_length=254)
            t.append([tgt] + self.tok.convert_ids_to_tokens(ids) + ["</s>"])
        res = self.tr.score_batch(s, t, max_batch_size=max_batch_tokens, batch_type="tokens")
        out = []
        for r in res:
            lps = list(r.log_probs)[1:]                  # the first scored token is the forced language code: not evidence about the translation
            out.append(float(np.mean(lps)) if len(lps) else -99.0)
        return out


class ForcedDecodingScorer:
    """Signal (B): forced-decoding mean log-probability under NLLB. Thin wrapper around `NllbCT2` with call/second counters (cost per reward call)."""

    def __init__(self, ct2_dir: str, hf_dir: str, device: str = "cuda", compute_type: str = "int8_float16", batch_tokens: int = 4096, model: Optional[NllbCT2] = None):
        self.model = model or NllbCT2(ct2_dir, hf_dir, device, compute_type)
        self.batch_tokens = batch_tokens
        self.calls = 0
        self.pairs = 0
        self.seconds = 0.0

    def score(self, iso: str, srcs: Sequence[str], mts: Sequence[str]) -> List[float]:
        t0 = time.time()
        out = self.model.score(iso, srcs, mts, self.batch_tokens)
        self.calls += 1
        self.pairs += len(out)
        self.seconds += time.time() - t0
        return out
