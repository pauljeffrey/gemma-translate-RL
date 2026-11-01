#!/usr/bin/env python3
"""Shared helpers of the clean science-stage rebuild (agent C2, 2026-10-28). Imported by s4_*.py.

* path discovery: the `gemma_mt` package (GEMMA_MT_PKG, /workspace/work/gemma_mt_pkg, the repo copy) and the OLD builder scripts `mix2_*.py` / `build_sft_science_mix.py`
  (S4_OLD_SCRIPTS, <repo>/gemma_mt/scripts), which are reused where their logic is unchanged (long-document scoring and selection);
* `Protected`   the protected-index check (decon.py logic, rules R1 exact sentence / R2 contained sentence / R3 shared word 8-gram, both normal forms) loaded STRICTLY: the FLORES+ dev
                train-half exemption is never attached. Reuses s3_common.Checker (fork pool). `names(texts)` -> per text the list of protected sets hit ([] = clean);
* licence tags  per-row licence class + the two UNRESOLVED licence notes (LibreTexts cc-by vs CC BY-NC-SA; AfriScience-MT paper licences) so that they can be disclosed;
* forbidden sources: rows whose source matches NLLB / TranslatePsy / teacher are refused everywhere (owner rule 2026-10-23);
* the replay budget functions of the old builder (UniMax water filling), the prompt parser and jsonl helpers.
Nothing here writes benchmark text anywhere.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
import time
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

# ---------------------------------------------------------------------------------------------------------------- the recipe (A2W, reports/clean_rl_science_data_2026-10-28.md sec. 3)
SCI_LANGS = ("amh", "hau", "lug", "nso", "yor", "zul")
COMPONENTS = ("science", "replay", "archive", "longdoc", "anchor")          # source.npy codes 0..4
A2W_SHARES = {"science": 0.317, "replay": 0.430, "archive": 0.050, "longdoc": 0.135, "anchor": 0.067}
A2W_TOKENS = {"science": 6_619_550, "replay": 8_985_911, "archive": 1_043_557, "longdoc": 2_825_610, "anchor": 1_406_138}      # total 20,880,766 tokens, 319 steps x 65,536
# 47 weak / Ge'ez languages of the A2W anchors (reports/sft_mixture_v2_2026-10-22/anchor_unfiltered_stats.json, VERIFIED); the 45 weak-signal languages are these minus amh, tir
ANCHOR_LANGS = ("ach amh bam bas bbj bin dag dua dyu efi fon her ibb ish iso kab kam kbp kea kmb kon kqn kri kua loz lun mos nbl ndc nus nyn oke sag seh shi sid swc tir tiv tsc umb urh "
                "ven vmw wal wes yao").split()
WEAK45 = tuple(x for x in ANCHOR_LANGS if x not in ("amh", "tir"))
SEQ_LEN = 8192

LICENCE_CLASSES = ("commercial_ok", "share_alike", "nc", "unknown", "nc_teacher", "owner")
LIBRETEXTS_NOTE = ("UNRESOLVED licence conflict: the pool row says cc-by, the data backup README says CC BY-NC-SA (S3 report 2026-10-28 sec. 5, open item 2). "
                   "Disclose; switch off with --exclude-source libretexts")
AFRISCIENCE_NOTE = "dsfsi/afriscience_mt card says Apache-2.0; the licence of the 230 underlying papers is not stated (S3 report open item 1). Disclose"
FORBIDDEN_SOURCE_RE = re.compile(r"nllb|translate[_ -]?psy|teacher|\br14\b", re.I)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------- paths
def gemma_pkg() -> str:
    """Directory that contains the python package gemma_mt (put on sys.path)."""
    for p in (os.environ.get("GEMMA_MT_PKG"), "/workspace/work/gemma_mt_pkg", os.path.join(REPO, "gemma_mt")):
        if p and os.path.isdir(os.path.join(p, "gemma_mt")):
            if p not in sys.path:
                sys.path.insert(0, p)
            return p
    raise SystemExit("gemma_mt package not found (set GEMMA_MT_PKG)")


def old_scripts() -> str:
    """Directory with the old builder scripts (mix2_*.py ...). S4_OLD_SCRIPTS or <repo>/gemma_mt/scripts."""
    for p in (os.environ.get("S4_OLD_SCRIPTS"), os.path.join(REPO, "gemma_mt", "scripts"), "/workspace/s4/code/old_scripts"):
        if p and os.path.exists(os.path.join(p, "mix2_common.py")):
            if p not in sys.path:
                sys.path.insert(0, p)
            return p
    raise SystemExit("old builder scripts (mix2_common.py ...) not found (set S4_OLD_SCRIPTS)")


def setup_judge(cal: Optional[str] = None, anchors: Optional[str] = None, glotlid: Optional[str] = None, scorer: Optional[str] = None):
    """Import the old `mix2_common` and point its module-level constants at the NEW calibration files (they are read when a Judge is built).
    `scorer` = path of a json with the SCORER_SPEC dict (type worker|fake, python, pythonpath, model ids, batch size). Returns the module."""
    gemma_pkg()
    old_scripts()
    import mix2_common as C

    if cal:
        C.CAL = cal
    if anchors:
        C.ANCHORS = anchors
    if glotlid:
        C.GLOTLID = glotlid
    if scorer:
        C.SCORER_SPEC = json.load(open(scorer))
    return C


# ---------------------------------------------------------------------------------------------------------------- jsonl
def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    n = 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(path + ".tmp", path)
    return n


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# ---------------------------------------------------------------------------------------------------------------- licence / provenance
def licence_class(licence: str, source: str = "") -> str:
    """Row label -> class (same rule as the old mix2_build.licence_class, with the NLLB/teacher class kept for completeness)."""
    l = (licence or "").strip().lower()
    if not l:
        return "unknown"
    if re.search(r"(^|[-_ ])nc($|[-_ ])|non[-_ ]?commercial", l):
        return "nc"
    if "share" in l or "-sa" in l or "gfdl" in l:
        return "share_alike"
    if l.startswith(("cc-by", "cc0", "odc-by", "apache", "mit", "bsd", "public", "cc_by", "pddl", "unlicense", "commercial")):
        return "commercial_ok"
    return "unknown"


def licence_note(source_dataset: str, licence: str = "") -> str:
    s = (source_dataset or "").lower()
    if "libretexts" in s:
        return LIBRETEXTS_NOTE
    if "afriscience" in s:
        return AFRISCIENCE_NOTE
    return ""


def is_forbidden_source(*names: str) -> bool:
    return any(FORBIDDEN_SOURCE_RE.search(n or "") for n in names)


def provenance(component: str, **kw) -> dict:
    """The per-row provenance/licence columns every component row carries (missing -> empty string)."""
    cols = {"component": component, "source_dataset": "", "licence": "", "licence_class": "", "licence_note": "", "licence_unresolved": False, "prov_repo": "", "prov_ref": "",
            "prov_index": "", "generator": "", "uses_nllb_or_translatepsy": False}
    cols.update(kw)
    if not cols["licence_class"]:
        cols["licence_class"] = licence_class(cols["licence"], cols["source_dataset"])
    if not cols["licence_note"]:
        cols["licence_note"] = licence_note(cols["source_dataset"], cols["licence"])
    cols["licence_unresolved"] = bool(cols["licence_note"])
    return cols


# ---------------------------------------------------------------------------------------------------------------- protected index (strict)
class Protected:
    """Strict protected-index checker. index/manifest: .npz + sealed manifest (sha256 verified by the loader). Never attaches the FLORES+ dev train-half exemption."""

    def __init__(self, index: str, manifest: str, nproc: int = 2):
        import s3_common as S3

        self.ix, self.exempt, self.manifest = S3.load_strict(index, manifest)
        self.version = self.manifest["version"]
        self.chk = S3.Checker(self.ix, None, None, nproc=nproc)

    def names(self, texts: Sequence[str]) -> List[List[str]]:
        """Per text: sorted list of "RULE|set" hits ([] = clean). Duplicated texts are checked once."""
        uniq = sorted(set(t for t in texts if t))
        res = dict(zip(uniq, self.chk.run(uniq))) if uniq else {}
        rules = ("R1", "R2", "R3")
        out = []
        for t in texts:
            r = res.get(t)
            hit = []
            if r:
                for rule, m in zip(rules, r[:3]):
                    hit += [f"{rule}|{nm}" for nm in self.ix.names(m)] if m else []
            out.append(sorted(set(hit)))
        return out

    def hit(self, texts: Sequence[str]) -> List[bool]:
        return [bool(x) for x in self.names(texts)]

    def close(self):
        self.chk.close()


# ---------------------------------------------------------------------------------------------------------------- replay budgets (the old builder's functions, unchanged)
def unimax_budgets(available: Sequence[float], total: float, max_epochs: float = 1.0, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """UniMax water-filling: b_l = min(w_l * c, max_epochs * available_l), the level c chosen so that sum(b) = total (or every language is at its cap)."""
    n = np.asarray(available, dtype=np.float64)
    w = np.ones_like(n) if weights is None else np.asarray(weights, dtype=np.float64)
    cap = max_epochs * n
    if total >= cap.sum():
        return cap
    lo, hi = 0.0, float(cap.max() / w.min()) + 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.minimum(w * mid, cap).sum() > total:
            hi = mid
        else:
            lo = mid
    return np.minimum(w * lo, cap)


_PROMPT_RE = re.compile(r"^<bos><start_of_turn>user\nTranslate the text to [^:\n]*:  (.*)<end_of_turn>\n<start_of_turn>model\n$", re.S)


def english_of_prompt(tok, ids: np.ndarray, prompt_len: int) -> Optional[str]:
    m = _PROMPT_RE.match(tok.decode(ids[:prompt_len].tolist(), skip_special_tokens=False))
    return m.group(1) if m else None


def target_of_sample(tok, ids: np.ndarray, prompt_len: int, target_len: int, iso: str) -> Optional[str]:
    """`{iso} {translation}<end_of_turn>` -> translation (None if the sample does not have that shape)."""
    t = tok.decode(ids[prompt_len:prompt_len + target_len].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not t.endswith("<end_of_turn>") or not t.startswith(iso + " "):
        return None
    return t[len(iso) + 1:-len("<end_of_turn>")]


def tokenizer(model: str):
    gemma_pkg()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    from gemma_mt import format as fmt

    fmt.assert_template_matches_reference(tok)
    return tok


def example(tok, english: str, iso: str, target: str):
    gemma_pkg()
    from gemma_mt import format as fmt

    return fmt.build_example(english, iso, target, tok)
