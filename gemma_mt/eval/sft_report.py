# SPDX-License-Identifier: AGPL-3.0-or-later
"""Markdown report of the post-SFT benchmark evaluation (input: the dict written to results.json by `sft_benchmarks`).

    python -m gemma_mt.eval.sft_report --results /workspace/sft_build/eval/sft_best/results.json --out report.md

Every table shows ALL rows and the CLEAN subset (rows whose English sentence is not in the SFT training text); the clean numbers are the
headline. Macro averages are unweighted means over languages; the clean macro only includes languages with at least `min_subset_rows`
clean rows (the number of languages is printed next to every macro).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from gemma_mt.eval import benchmarks as B
from gemma_mt.languages import ALL_ISOS, LANGUAGES

METRICS = (("chrf", "chrF++", 2), ("spbleu", "spBLEU", 2), ("bleu", "BLEU", 2), ("comet", "AfriCOMET", 3), ("comet_qe", "AfriCOMET-QE", 3))
SUBSET_LABEL = {"all": "all rows", "clean": "clean (headline: English sentence not in the training text)",
                "clean_sentence": "clean, sentence rule only (the 2026-10-14 dev filter)", "clean_strict": "clean and no near duplicate (8-gram overlap < 50%)",
                "no_pair": "reference pair not seen (English may have been seen with another language)"}
SYSTEM_LABEL = {"sft": "SFT", "base": "base gemma-3-270m-it"}


def _f(x: Any, nd: int = 2) -> str:
    return "-" if x is None or x == "" or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def _pct(x: Any) -> str:
    return "-" if x is None or x == "" or (isinstance(x, float) and x != x) else f"{100 * x:.1f}%"


def _macro(sysres: Dict[str, Any], bench: str, subset: str) -> Dict[str, Any]:
    d = sysres.get("benchmarks", {}).get(bench, {})
    return d.get("macro" if subset == "all" else f"macro_{subset}", {}) or {}


def _lang(sysres: Dict[str, Any], bench: str, iso: str, subset: str = "all") -> Dict[str, Any]:
    rec = sysres.get("benchmarks", {}).get(bench, {}).get("languages", {}).get(iso, {})
    return rec if subset == "all" else (rec.get(subset) or {})


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]


EXACT_METRIC = ("exact", "exact match", 3)


def _metrics_for(res: Dict[str, Any]):
    """The default metric columns, plus exact match when a word-level benchmark (GATITOS) is in the results."""
    has = any("exact" in l for s in res["systems"].values() for b in s.get("benchmarks", {}).values() for l in b.get("languages", {}).values())
    return METRICS + (EXACT_METRIC,) if has else METRICS


def _benchmarks(res: Dict[str, Any]) -> List[str]:
    order = list(B.ALL_BENCHMARKS)
    seen = {b for s in res["systems"].values() for b in s.get("benchmarks", {})}
    return [b for b in order if b in seen] + sorted(seen - set(order))


def _systems(res: Dict[str, Any]) -> List[str]:
    return [n for n in ("sft", "base") if n in res["systems"]] + [n for n in res["systems"] if n not in ("sft", "base")]


def headline_tables(res: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    systems = _systems(res)
    metrics = _metrics_for(res)
    for subset in ("clean", "all"):
        out += [f"### Macro averages, {SUBSET_LABEL[subset]}", ""]
        rows = []
        for b in _benchmarks(res):
            for s in systems:
                m = _macro(res["systems"][s], b, subset)
                if not m:
                    continue
                rows.append([f"**{b}**" if s == systems[0] else b, SYSTEM_LABEL.get(s, s), str(m.get("n_languages", "")),
                             str(m.get("n_rows", "")) if subset != "all" else str(sum(l.get("n", 0) for l in
                             res["systems"][s]["benchmarks"][b]["languages"].values()))] +
                            [f"{_f(m.get(k), nd)} ({m.get(k + '_n_languages', '')})" if k in m else "-" for k, _, nd in metrics])
        out += _table(["benchmark", "system", "languages", "rows"] + [f"{lab} (n lang)" for _, lab, _ in metrics], rows)
    sft = res["systems"].get("sft", {})
    if sft.get("macro_all_clean"):
        out += ["### Macro over every (benchmark, language) pair", ""]
        rows = []
        for s in systems:
            r = res["systems"][s]
            c, a = r.get("macro_all_clean", {}), r.get("macro_all", {})
            rows.append([SYSTEM_LABEL.get(s, s), f"{_f(c.get('chrf'))} ({c.get('chrf_n_languages', '')})", f"{_f(a.get('chrf'))} ({a.get('chrf_n_languages', '')})",
                         _f(c.get("comet"), 3), _f(a.get("comet"), 3)])
        out += _table(["system", "chrF++ clean (n)", "chrF++ all rows (n)", "AfriCOMET clean", "AfriCOMET all"], rows)
    return out


def contamination_tables(res: Dict[str, Any]) -> List[str]:
    out = ["## Contamination of the benchmarks by the SFT training text", "",
           "A row is *contaminated* when the normalised English sentence (NFKC, case-folded, punctuation removed) of the benchmark row occurs in the SFT "
           "training windows (`gemma/windows_sft`, the input of the packed training set): as a whole sentence of a window or as a contiguous run inside one. "
           "*sentence rule* = the first half only (the filter that produced `bench_dev_clean.jsonl`; it reproduces that file exactly: 10,043 of 30,418 dev rows). "
           "*near* = not contaminated but >= 50% of its 8-token n-grams occur in the training English. *pair* = contaminated and the reference translation "
           "occurs in the target side of a window of the same language (the model saw the very pair).", ""]
    sft = res["systems"].get("sft") or next(iter(res["systems"].values()))
    rows = []
    for b in _benchmarks(res):
        langs = sft.get("benchmarks", {}).get(b, {}).get("languages", {})
        n = sum(l.get("n", 0) for l in langs.values())
        if not any("n_contaminated" in l for l in langs.values()):
            rows.append([b, str(n), "-", "-", "-", "-", "-"])
            continue
        tot = {k: sum(l.get(k, 0) for l in langs.values()) for k in ("n_contaminated", "n_sentence_rule", "n_near", "n_pair")}
        rows.append([b, str(n), f"{tot['n_contaminated']} ({_pct(tot['n_contaminated'] / max(n, 1))})", f"{tot['n_sentence_rule']} ({_pct(tot['n_sentence_rule'] / max(n, 1))})",
                     f"{tot['n_near']} ({_pct(tot['n_near'] / max(n, 1))})", f"{tot['n_pair']} ({_pct(tot['n_pair'] / max(n, 1))})",
                     f"{n - tot['n_contaminated']} ({_pct(1 - tot['n_contaminated'] / max(n, 1))})"])
    out += _table(["benchmark", "rows", "contaminated", "of which sentence-rule hits", "near", "pair seen", "clean rows"], rows)
    for b in _benchmarks(res):
        langs = sft.get("benchmarks", {}).get(b, {}).get("languages", {})
        if not langs or not any("n_contaminated" in l for l in langs.values()):
            continue
        shares = {i: l["n_contaminated"] / max(l["n"], 1) for i, l in langs.items()}
        if len({round(v, 4) for v in shares.values()}) == 1:
            n0 = next(iter(langs.values()))["n"]
            out += [f"**{b}**: the English side is the same for every language (multi-way parallel set: FLORES+, NTREX, SmolSent, IbomMT, FLORES-MOZ), so every language has the same "
                    f"{next(iter(langs.values()))['n_contaminated']} of {n0} rows contaminated ({_pct(next(iter(shares.values())))}); rows whose reference pair was seen, "
                    "per language: " + (", ".join(f"{i} {l.get('n_pair', 0)}" for i, l in langs.items() if l.get("n_pair")) or "none") + ".", ""]
        else:
            rows = [[f"{LANGUAGES[i].name} (`{i}`)", str(l["n"]), f"{l['n_contaminated']} ({_pct(shares[i])})", str(l.get("n_sentence_rule", "")),
                     str(l.get("n_near", "")), str(l.get("n_pair", "")), str(l["n"] - l["n_contaminated"])] for i, l in langs.items()]
            out += [f"**{b}** per language:", ""] + _table(["language", "rows", "contaminated", "sentence rule", "near", "pair", "clean"], rows)
    return out


def per_language_tables(res: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    systems = _systems(res)
    sft = res["systems"].get("sft")
    base = res["systems"].get("base")
    if not sft:
        return out
    for b in _benchmarks(res):
        langs = sft["benchmarks"].get(b, {}).get("languages", {})
        if not langs:
            continue
        if b in B.WORD_LEVEL_BENCHMARKS:
            out += [f"### {b}: SFT model per language (word level)", "",
                    "n = entries (English word or phrase); exact = the output equals ANY accepted translation (NFC, case-folded, surrounding punctuation removed); "
                    "chrF++ = against the best accepted translation. clean = the entry is not a whole sentence of the training windows.", ""]
            rows = []
            for i in [i for i in ALL_ISOS if i in langs]:
                a, c = langs[i], langs[i].get("clean") or {}
                bc = _lang(base, b, i, "clean") if base else {}
                ba = _lang(base, b, i, "all") if base else {}
                rows.append([f"{LANGUAGES[i].name} (`{i}`)", str(a.get("n", "")), str(c.get("n", "-")), _f(a.get("chrf")), _f(c.get("chrf")), _pct(a.get("exact")),
                             _pct(c.get("exact")), _pct(ba.get("exact")) if base else "-", _pct(bc.get("exact")) if base else "-", _pct(a.get("finished_rate")),
                             _pct(a.get("prefix_ok_rate"))])
            out += _table(["language", "n", "n clean", "chrF++ all", "chrF++ clean", "exact all", "exact clean", "base exact all", "base exact clean",
                           "finished", "prefix ok"], rows)
            continue
        out += [f"### {b}: SFT model per language", "",
                "n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy "
                "(the clean macro leaves out languages below the minimum number of clean rows).", ""]
        rows = []
        for i in [i for i in ALL_ISOS if i in langs]:
            a, c = langs[i], langs[i].get("clean") or {}
            bc = _lang(base, b, i, "clean") if base else {}
            rows.append([f"{LANGUAGES[i].name} (`{i}`)", str(a.get("n", "")), str(c.get("n", "-")), _f(a.get("chrf")), _f(c.get("chrf")), _f(a.get("spbleu")),
                         _f(c.get("spbleu")), _f(a.get("bleu")), _f(c.get("bleu")), _f(a.get("comet"), 3), _f(c.get("comet"), 3), _f(c.get("comet_qe"), 3),
                         _f(bc.get("chrf")) if base else "-", (f"{c['chrf'] - bc['chrf']:+.2f}" if c.get("chrf") is not None and bc.get("chrf") is not None else "-")])
        out += _table(["language", "n", "n clean", "chrF++ all", "chrF++ clean", "spBLEU all", "spBLEU clean", "BLEU all", "BLEU clean", "AfriCOMET all",
                       "AfriCOMET clean", "QE clean", "base chrF++ clean", "Δ chrF++ clean"], rows)
    return out


def failure_tables(res: Dict[str, Any]) -> List[str]:
    out = ["## Output format failures", "",
           "`prefix_ok` = the completion starts with the expected language tag (the base model does not know the convention: its whole output is scored, "
           "the tag stripped only if present); `finished` = `<end_of_turn>` produced inside the token budget; `empty` = nothing left after stripping. "
           "All rows, unweighted mean over languages; the worst languages follow.", ""]
    rows = []
    worst: List[str] = []
    for b in _benchmarks(res):
        for s in _systems(res):
            langs = res["systems"][s].get("benchmarks", {}).get(b, {}).get("languages", {})
            if not langs:
                continue
            tot = sum(l["n"] for l in langs.values())
            avg = lambda k: sum(l.get(k, 0) * l["n"] for l in langs.values()) / max(tot, 1)      # noqa: E731 - row-weighted
            rows.append([b, SYSTEM_LABEL.get(s, s), _pct(avg("prefix_ok_rate")), _pct(avg("finished_rate")), _pct(avg("empty_rate"))])
            if s == "sft":
                for key, label in (("prefix_ok_rate", "lowest prefix_ok"), ("finished_rate", "lowest finished"), ("empty_rate", "highest empty")):
                    srt = sorted(langs.items(), key=lambda kv: kv[1].get(key, 0), reverse=(key == "empty_rate"))[:4]
                    worst.append(f"- {b}, SFT, {label}: " + ", ".join(f"{i} {_pct(l.get(key))}" for i, l in srt))
    out += _table(["benchmark", "system", "prefix_ok", "finished", "empty"], rows)
    out += worst + [""]
    return out


def sensitivity_tables(res: Dict[str, Any]) -> List[str]:
    out = ["## How the contamination rule changes the macro chrF++", ""]
    rows = []
    for b in _benchmarks(res):
        for s in _systems(res):
            cells = []
            for sub in ("all", "clean", "clean_sentence", "clean_strict", "no_pair"):
                m = _macro(res["systems"][s], b, sub)
                cells.append(f"{_f(m.get('chrf'))} ({m.get('n_languages', '')}L)" if m else "-")
            rows.append([b, SYSTEM_LABEL.get(s, s)] + cells)
    out += _table(["benchmark", "system", "all", "clean", "clean, sentence rule", "clean, no near", "pair not seen"], rows)
    return out


def runtime_table(res: Dict[str, Any]) -> List[str]:
    rows = []
    for s in _systems(res):
        m = res["systems"][s]["meta"]
        for b, t in (m.get("timings") or {}).items():
            rows.append([SYSTEM_LABEL.get(s, s), b, str(res["meta"].get("rows", {}).get(b, "")), str(t.get("sampler", "")),
                         ("cached " if t.get("generation_cached") else "") + _f(t.get("generation_seconds"), 0), _f(t.get("scoring_seconds"), 0)])
    return ["## Runtime", ""] + _table(["system", "benchmark", "rows", "sampler", "generation s", "scoring s (incl. AfriCOMET)"], rows) + \
        [f"Whole run: {_f(res['meta'].get('seconds'), 0)} s.", ""]


def render_report(res: Dict[str, Any], title: str = "SFT benchmark evaluation (English -> African)") -> str:
    meta = res.get("meta", {})
    lines = [f"# {title}", ""]
    sft = res["systems"].get("sft", {}).get("meta", {})
    ts = sft.get("training_state", {})
    lines += [f"* SFT model: `{sft.get('model')}` revision `{sft.get('revision') or 'main'}` commit `{sft.get('sha')}`"
              + (f" (training step {ts.get('step')}, macro validation loss {_f(ts.get('val_loss_macro'), 4)})" if ts else ""),
              f"* base model: `{res['systems'].get('base', {}).get('meta', {}).get('model', '-')}`",
              "* decoding: greedy, same prompt for both (`Translate the text to {language}:  {english}`), max_new_tokens "
              f"{sft.get('max_new_tokens')}, sampler {sft.get('sampler')}; metrics on NFC-normalised text; spBLEU tokenizer {sft.get('spbleu_tokenizer')}",
              f"* started {meta.get('started')}, finished {meta.get('finished')}, {_f(meta.get('seconds'), 0)} s; environment {json.dumps(meta.get('environment', {}))}", ""]
    if res.get("failures"):
        lines += ["## FAILURES", ""] + [f"- {x}" for x in res["failures"]] + [""]
    if res.get("warnings"):
        lines += ["## Warnings", ""] + [f"- {x}" for x in res["warnings"]] + [""]
    lines += ["## Results", ""] + headline_tables(res)
    lines += contamination_tables(res)
    lines += per_language_tables(res)
    lines += failure_tables(res)
    lines += sensitivity_tables(res)
    lines += runtime_table(res)
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.results, encoding="utf-8") as f:
        res = json.load(f)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(render_report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
