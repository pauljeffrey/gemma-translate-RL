# SPDX-License-Identifier: AGPL-3.0-or-later
"""QE-free calibration of a language from HUMAN reference translations (D-43): the consensus-only counterpart of `calibrate_shaping`.

`calibrate_shaping` needs the AfriCOMET-QE scorer (anchors, AUC, same-source accuracy). A language that is judged by the QE-free consensus signal
(D-38, content mode `consensus`) has no use for those numbers, but the SAME reference sets still give it everything else the reward reads from a
calibration entry:

  len_ratio    p01 / p02 / p05 / p50 / p95 / corpus / s2 of len(reference) / len(source) (characters): the length window and the omission model
  diacritics   density distribution, threshold, allowed inventory (rl/diacritics.py)
  lexical      human P10 / P50 of the numbers / names / content-word overlap with the source (rl/lexical.py)
  omission     per-language operating point of the dropped-sentence detector (rl/omission_eval.choose_omission_cost)

and the entry keeps the shape of the other consensus-only entries of v5: `content` {mode: consensus}, `lid` (measured separately), advantage weight in
`reliability` / `reliability_same_source`, NO `anchors` (the shaper falls back to the default anchors, which a consensus-mode language never uses).

Reference sets (English -> X, test_sets layout of eval/fetch_data.py): smol_sent (`src`, `trg`), smol_doc (`srcs`, `trgs`: sentence aligned, one row =
one document), ibom_mt (`src`, `trg`), salt (`src`, `trg`). Pairs are NFC-normalised, de-duplicated by English and split DETERMINISTICALLY by a hash of the
English sentence into a CALIBRATION half and a HOLD-OUT half; only the calibration half may feed an entry. The contamination flags of
`bench/test_sets/_contamination/` (English side seen in the SFT windows, reference pair seen) are attached to every pair.
"""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SPLIT_SALT = "calib-split-v1"
DEFAULT_SOURCES = ("smol_sent", "ibom_mt", "smol_doc")           # order = priority when the same English occurs in several sets


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()


def split_bucket(english: str, salt: str = SPLIT_SALT) -> float:
    """Deterministic value in [0, 1) from the NFC English sentence (case and spacing kept: the sentence is the key)."""
    h = hashlib.sha1((salt + "\n" + nfc(english)).encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def split_rows(rows: Sequence[dict], frac: float = 0.5, salt: str = SPLIT_SALT) -> Tuple[List[dict], List[dict]]:
    """(calibration, hold-out): a pair is in the calibration half when its English hashes below `frac`. Same English -> same half, always."""
    cal, hold = [], []
    for r in rows:
        (cal if split_bucket(r["english"], salt) < frac else hold).append(r)
    return cal, hold


def _jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _contamination(contam_dir: Optional[str], source: str) -> Dict[str, Dict[str, bool]]:
    if not contam_dir:
        return {}
    p = os.path.join(contam_dir, f"contamination_{source}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def _flags(cont: dict, key: str, doc: bool) -> Dict[str, Optional[bool]]:
    """english_seen: the English side occurs in the SFT windows (sentence rule; documents: exact OR near); pair_seen: the reference pair does."""
    if not cont:
        return {"english_seen": None, "pair_seen": None}
    g = lambda k: bool((cont.get(k) or {}).get(key, False))
    return {"english_seen": (g("exact") or g("near")) if doc else (g("sentence") or g("exact")), "pair_seen": g("pair")}


def load_pairs(test_sets: str, iso: str, spec: Dict[str, str], contam_dir: Optional[str] = None,
               sources: Sequence[str] = DEFAULT_SOURCES) -> List[dict]:
    """All English -> `iso` reference pairs of the sets in `spec` ({set name: file code}, e.g. {"smol_sent": "efi", "ibom_mt": "efi"}), NFC, de-duplicated by
    English (the first set in `sources` order wins, the contamination flags of the duplicates are OR-ed). Row: english, reference, iso, source, doc (document id
    or None), idx (sentence index in the document / row id), english_seen, pair_seen."""
    rows: List[dict] = []
    for source in sources:
        code = spec.get(source)
        path = os.path.join(test_sets, source, f"en_{code}.jsonl") if code else None
        if not path or not os.path.exists(path):
            continue
        cont = _contamination(contam_dir, source)
        for r in _jsonl(path):
            if source == "smol_doc":
                key = f"{iso}:{r['id']}"
                fl = _flags(cont, key, doc=True)
                for i, (s, t) in enumerate(zip(r["srcs"], r["trgs"])):
                    rows.append({"english": nfc(s), "reference": nfc(t), "iso": iso, "source": source, "doc": str(r["id"]), "idx": i, **fl})
            else:
                key = f"{iso}:{r['id']}"
                rows.append({"english": nfc(r["src"]), "reference": nfc(r["trg"]), "iso": iso, "source": source, "doc": None, "idx": str(r["id"]),
                             **_flags(cont, key, doc=False)})
    seen: Dict[str, dict] = {}
    out: List[dict] = []
    for r in rows:
        if not r["english"] or not r["reference"]:
            continue
        k = r["english"]
        if k in seen:
            for f in ("english_seen", "pair_seen"):
                if r[f]:
                    seen[k][f] = True
            continue
        seen[k] = r
        out.append(r)
    return out


def _order_key(r: dict):
    """Document order for the omission test: SmolDoc documents first (sentences contiguous), then the sentence sets in id order."""
    rank = {"smol_doc": 0, "ibom_mt": 1, "salt": 2, "smol_sent": 3}.get(r["source"], 4)
    idx = r["idx"]
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        pass
    return (rank, r["doc"] or "", 0 if isinstance(idx, int) else 1, idx if isinstance(idx, int) else str(idx))


def ordered_for_omission(rows: Iterable[dict]) -> List[dict]:
    """The rows as `omission_eval` wants them: sorted by a running `id`, so windows of consecutive rows are made of sentences of the same document (SmolDoc)
    or of the same set."""
    srt = sorted(rows, key=_order_key)
    return [dict(r, id=i) for i, r in enumerate(srt)]


def reference_stats(rows: Sequence[dict], omission: bool = True, omission_target: float = 0.02) -> Dict[str, Any]:
    """Everything a consensus-only calibration entry takes from human references (see the module doc). `rows`: calibration pairs only."""
    from gemma_mt.rl import diacritics, lexical
    from gemma_mt.rl.omission_eval import choose_omission_cost, length_stats

    src = [r["english"] for r in rows]
    ref = [r["reference"] for r in rows]
    ls = length_stats(src, ref)
    out: Dict[str, Any] = {"n": len(rows), "len_ratio": {k: ls[k] for k in ("p05", "p50", "p95", "p01", "p02", "corpus", "s2")}}
    lex = lexical.summarise(zip(src, ref))
    if lex:
        out["lexical"] = lex
    dia = diacritics.summarise(ref)
    if dia:
        out["diacritics"] = dia
    if omission:
        out["omission"] = choose_omission_cost(ordered_for_omission(rows), omission_target)
    return out


def consensus_entry(stats: Dict[str, Any], lid_block: Optional[dict], weight: float, provenance: Dict[str, Any]) -> Dict[str, Any]:
    """A consensus-only calibration entry with human-reference statistics: the keys of the 23 consensus-only entries of v5 (usable, n, reliability,
    reliability_same_source, content, weight_reason, lid) plus len_ratio / diacritics / lexical / omission. No `anchors` / QE statistics. weight 0 -> the
    language stays disabled (`usable` false, advantage weight 0)."""
    on = weight > 0
    e: Dict[str, Any] = {"usable": bool(on), "n": int(stats["n"]), "reliability": float(weight), "reliability_same_source": float(weight)}
    for k in ("len_ratio", "diacritics", "lexical", "omission"):
        if stats.get(k):
            e[k] = stats[k]
    e["content"] = {"mode": "consensus", "beta": 1.0,
                    "why": "QE-free consensus (D-38 / D-43): human-reference statistics from the calibration half of the reference sets, no QE call, no anchors"}
    e["weight_reason"] = provenance.get("weight_reason", "")
    if lid_block:
        e["lid"] = lid_block
    e["calibration_source"] = {k: v for k, v in provenance.items() if k != "weight_reason"}
    if not on:
        e["reason"] = "consensus signal not validated against references (D-43): advantage weight 0"
    return e


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation with average ranks for ties; None when either side is constant or there are < 3 points."""
    n = len(x)
    if n < 3 or n != len(y):
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r

    rx, ry = ranks(list(x)), ranks(list(y))
    mx, my = sum(rx) / n, sum(ry) / n
    sx = sum((a - mx) ** 2 for a in rx)
    sy = sum((b - my) ** 2 for b in ry)
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy) ** 0.5


def weight_from_within_prompt_spearman(rho: Optional[float]) -> float:
    """D-43 decision rule: within-prompt Spearman >= 0.25 -> 0.5 (like the other consensus-only languages); 0.10-0.25 -> 0.3; below 0.10 -> 0 (disabled)."""
    if rho is None or rho != rho:
        return 0.0
    return 0.5 if rho >= 0.25 else (0.3 if rho >= 0.10 else 0.0)
