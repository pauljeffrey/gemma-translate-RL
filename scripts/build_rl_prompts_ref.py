#!/usr/bin/env python3
"""Reference-anchored RL prompts (owner 2026-11-03, reward.reference_weight): English windows that come WITH a human translation, domain "ref".

    python scripts/build_rl_prompts_ref.py --index PROTECTED/protected_index.npz --manifest PROTECTED/protected_manifest.json --out artifacts/rl_prompts_ref [--max-per-lang 1500]

One parquet per language in --out (put them at <prompts_path>/ref/<iso>.parquet; `load_rl_prompts` walks the pool recursively, the `domain` column says "ref"). Columns = the news pool's columns plus
`reference`, `ref_source`, `ref_licence`, `seen_in_sft`. Sources (HUMAN translations only; every one documented in reports/rl_features_dynamic_ref_2026-11-03.md):
    afriscience   dsfsi/afriscience_mt via the clean train file of <HF_ORG_LEGACY>/sabiyarn-clean-rl-prompts-2026-10-28 (34,007 pairs; Apache-2.0; professional translators; amh hau lug nso yor zul)
    vukuzenzele   (OFF by default, --vukuzenzele DIR) dsfsi/vukuzenzele-sentence-aligned EVAL + TEST splits (CC-BY-4.0; South African government magazine, PDF-extracted, alignment score filtered; afr nbl nso sot ssw
                  tsn tso ven xho zul). The SFT build read the TRAIN split only. LEFT OUT of release 1: the corpus is lower-cased on both sides, a case-sensitive chrF++ would reward lower-cased output.
    ibom_train    howard-nlp/ibom-mt eng-<iso>/train.csv (CC-BY-NC-4.0, human; efi ibb ...). Seen in SFT (train split, owner decision 2026-10-28: the one deliberate NC source).
NOT used: MT560 / OPUS (mixes web-mined and religious text, no row level provenance), NLLB-derived or machine-translated pairs, MAFAND train (owner), benchmark dev/devtest/test sets of the protected index,
sudoping01 Wolof (provenance undocumented), the Vukuzenzele eval / test splits (lower-cased), the Bible (cap 25 % would apply; none in this release), Pontoon / Weblate (UI strings).
Every pair is checked against the protected benchmark index on BOTH sides (strict rule of decon.py: exact sentence, contained sentence, shared word 8-gram); hits are dropped and counted.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import unicodedata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "data_pipeline", "sft_clean"))

from gemma_mt.languages import LANGUAGES, SUPPORTED_RL, canonical_iso   # noqa: E402

POOL = "ref_v1_2026-11-03"
VUK_MIN_SCORE = 0.80
SPACED = re.compile(r"(?:\b\w\s){5,}\w\b")                   # PDF extraction artefact: "b a n g a b a n e z i m p a w u"
URLISH = re.compile(r"https?://|www\.|@\w+\.\w+")
MAX_CHARS = 1200


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def ok_pair(en: str, rf: str) -> bool:
    if not en or not rf or en.lower() == rf.lower():
        return False
    if len(en.split()) < 3 or len(en) > MAX_CHARS or len(rf) > 3 * MAX_CHARS:
        return False
    r = len(rf) / max(len(en), 1)
    if r < 0.3 or r > 4.0:
        return False
    if SPACED.search(en) or SPACED.search(rf) or URLISH.search(en) or URLISH.search(rf):
        return False
    return True


def load_afriscience(src):
    import pyarrow.parquet as pq

    t = pq.read_table(src).to_pylist()
    for i, r in enumerate(t):
        yield dict(english=r["source"], reference=r["target"], iso=r["target_lang"], source_dataset="dsfsi/afriscience_mt (clean train)", licence="apache-2.0", ref_source="afriscience_mt_train",
                   ref_licence="apache-2.0", seen_in_sft="no (science SFT stage not run yet)", prov_file="afriscience_clean_train.parquet", prov_row=i,
                   paper=int(r["paper_id"]), sent=int(r["sentence_id"]))


def load_vukuzenzele(root):
    import pyarrow.parquet as pq

    for d in sorted(glob.glob(os.path.join(root, "*-*"))):
        a, b = os.path.basename(d).split("-")
        if a == "eng":
            code = b
        elif (a, b) == ("afr", "eng"):
            code = "afr"
        else:
            continue                                                   # African-African pairs
        iso = canonical_iso(code)
        if iso is None or iso not in SUPPORTED_RL:
            continue
        for f in sorted(glob.glob(os.path.join(d, "*-0000*.parquet"))):
            split = os.path.basename(f).split("-")[0]
            if split not in ("eval", "test"):
                continue
            for i, r in enumerate(pq.read_table(f).to_pylist()):
                if r.get("score", 0.0) < VUK_MIN_SCORE:
                    continue
                yield dict(english=r["eng"], reference=r[code], iso=iso, source_dataset="dsfsi/vukuzenzele-sentence-aligned", licence="cc-by-4.0", ref_source=f"vukuzenzele_{split}",
                           ref_licence="cc-by-4.0", seen_in_sft="split not read by the SFT build (train only); may overlap through Mehgoss/sa-languages-translation (unverified)",
                           prov_file=f"{os.path.basename(d)}/{os.path.basename(f)}", prov_row=i)


def load_ibom(root):
    import csv

    for f in sorted(glob.glob(os.path.join(root, "eng-*", "train.csv"))):
        code = os.path.basename(os.path.dirname(f)).split("-")[1]
        iso = canonical_iso(code)
        if iso is None or iso not in SUPPORTED_RL:
            continue
        with open(f, encoding="utf-8", newline="") as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                yield dict(english=r.get("eng") or r.get("english"), reference=r.get(code), iso=iso, source_dataset="howard-nlp/ibom-mt (train)", licence="cc-by-nc-4.0", ref_source="ibom_mt_train",
                           ref_licence="cc-by-nc-4.0", seen_in_sft="yes (train split, owner exception 2026-10-28)", prov_file=f"{os.path.basename(os.path.dirname(f))}/train.csv", prov_row=i)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--afriscience", required=True, help="afriscience_clean_train.parquet")
    ap.add_argument("--vukuzenzele", default=None, help="local dir of dsfsi/vukuzenzele-sentence-aligned (eval/test parquet). OFF by default: the corpus is LOWERCASED on both sides (none of the rows contain an upper-case letter), "
                    "so a case-sensitive chrF++ against it would reward lower-cased output")
    ap.add_argument("--ibom", required=True, help="local dir of howard-nlp/ibom-mt (eng-*/train.csv)")
    ap.add_argument("--afro-tokenizer", required=True, help="tokenizer.json of Davlan/afro-xlmr-large-76L (src_afr_tokens = AfriCOMET encoder tokens)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-lang", type=int, default=1500)
    ap.add_argument("--max-per-source-lang", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--min-rows-per-lang", type=int, default=150, help="a language with fewer reference rows is left out (it would be drawn in a tiny loop)")
    ap.add_argument("--max-windows-per-lang", type=int, default=500, help="multi-sentence AfriScience windows (b256 stage)")
    a = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    import xxhash
    from tokenizers import Tokenizer

    from protected_index_loader import load as load_index

    tk = Tokenizer.from_file(a.afro_tokenizer)
    ix, man = load_index(a.index, a.manifest)
    version = man.get("version", "protected")
    rows, drop = [], collections.Counter()
    for gen in [load_afriscience(a.afriscience), load_ibom(a.ibom)] + ([load_vukuzenzele(a.vukuzenzele)] if a.vukuzenzele else []):
        for r in gen:
            iso = canonical_iso(r["iso"])
            if iso is None or iso not in SUPPORTED_RL:
                drop["language_not_rl"] += 1
                continue
            en, rf = nfc(str(r["english"] or "")), nfc(str(r["reference"] or ""))
            if not ok_pair(en, rf):
                drop["pair_rules"] += 1
                continue
            rows.append(dict(r, iso=iso, english=en, reference=rf))
    print("candidates", len(rows), "dropped", dict(drop), flush=True)
    # protected index, both sides
    keep = []
    B = 5000
    hits = collections.Counter()
    for s in range(0, len(rows), B):
        chunk = rows[s:s + B]
        he = ix.check_texts([c["english"] for c in chunk])
        hr = ix.check_texts([c["reference"] for c in chunk])
        for c, x, y in zip(chunk, he, hr):
            if any(x) or any(y):
                hits[c["ref_source"]] += 1
            else:
                keep.append(c)
    print("protected-index hits (dropped):", dict(hits), "kept", len(keep), flush=True)
    # b256 windows: consecutive AfriScience sentences of one paper (sentence ids n, n+1, ...) joined to 65..250 source tokens; the other sources are single sentences (no document order)
    def ntok(t):
        return len(tk.encode(t).ids)

    sci = collections.defaultdict(list)
    for c in keep:
        if c["ref_source"] == "afriscience_mt_train":
            sci[(c["iso"], c["paper"])].append(c)
    wins = []
    for (iso, paper), lst in sci.items():
        lst.sort(key=lambda c: c["sent"])
        cur = []
        def flush():
            if len(cur) >= 2:
                en, rf = " ".join(x["english"] for x in cur), " ".join(x["reference"] for x in cur)
                if 65 <= ntok(en) <= 250 and ok_pair(en, rf):
                    w = dict(cur[0], english=en, reference=rf, prov_row=cur[0]["prov_row"], window_sentences=len(cur))
                    wins.append(w)
        for c in lst:
            if cur and (c["sent"] != cur[-1]["sent"] + 1 or ntok(" ".join(x["english"] for x in cur) + " " + c["english"]) > 250):
                flush()
                cur = []
            cur.append(c)
        flush()
    rngw = np.random.default_rng(a.seed + 1)
    wby = collections.defaultdict(list)
    for w in wins:
        wby[w["iso"]].append(w)
    wins = [w for iso, l in wby.items() for w in [l[i] for i in rngw.permutation(len(l))[:a.max_windows_per_lang]]]
    hw = ix.check_texts([w["english"] for w in wins]) if wins else []
    hr = ix.check_texts([w["reference"] for w in wins]) if wins else []
    wins = [w for w, x, y in zip(wins, hw, hr) if not (any(x) or any(y))]
    for w in wins:
        w["ref_source"] = "afriscience_mt_train_window"
    print("afriscience b256 windows:", len(wins), dict(collections.Counter(w["iso"] for w in wins)), flush=True)
    keep = keep + wins
    # dedupe on English per language, tokens, bins
    seen, out = set(), []
    for c in keep:
        k = (c["iso"], c["english"].lower())
        if k in seen:
            drop["duplicate_english"] += 1
            continue
        seen.add(k)
        c["src_afr_tokens"] = len(tk.encode(c["english"]).ids)
        if c["src_afr_tokens"] > 256:
            drop["too_long"] += 1
            continue
        c["length_bin"] = "b64" if c["src_afr_tokens"] <= 64 else "b256"
        out.append(c)
    # per-source and per-language caps, both bins kept (b64 and b256 stages draw from their own bin)
    rng = np.random.default_rng(a.seed)
    by = collections.defaultdict(list)
    for c in out:
        by[(c["iso"], c["ref_source"], c["length_bin"])].append(c)
    sel = []
    per_src_lang = collections.defaultdict(list)
    for (iso, src, b), lst in by.items():
        per_src_lang[(iso, src)].append((b, lst))
    for (iso, src), parts in per_src_lang.items():
        total = sum(len(l) for _, l in parts)
        for b, lst in parts:
            take = min(len(lst), max(1, int(round(a.max_per_source_lang * len(lst) / max(total, 1)))) if total > a.max_per_source_lang else len(lst))
            sel += [lst[i] for i in rng.permutation(len(lst))[:take]]
    final = []
    by_lang = collections.defaultdict(list)
    for c in sel:
        by_lang[c["iso"]].append(c)
    for iso, lst in by_lang.items():
        for b, cap in (("b64", a.max_per_lang), ("b256", a.max_windows_per_lang)):          # the b64 and the b256 stage draw from their own bin: cap each bin separately
            part = [c for c in lst if c["length_bin"] == b]
            final += [part[i] for i in rng.permutation(len(part))[:cap]] if len(part) > cap else part
    small = {iso: len(l) for iso, l in by_lang.items() if len(l) < a.min_rows_per_lang}
    if small:
        print("languages left out (fewer than", a.min_rows_per_lang, "reference rows):", small, flush=True)
        final = [c for c in final if c["iso"] not in small]
    os.makedirs(a.out, exist_ok=True)
    schema_cols = ["english", "iso", "language", "src_afr_tokens", "length_bin", "n_sentences", "source_dataset", "licence", "domain", "reference", "ref_source", "ref_licence", "seen_in_sft",
                   "prov_pool", "prov_file", "prov_row", "prov_en_xxh64", "prov_index"]
    per = collections.defaultdict(list)
    for c in final:
        c["language"] = LANGUAGES[c["iso"]].name
        c["n_sentences"] = c.get("window_sentences") or len(re.findall(r"[.!?]+(?:\s|$)", c["english"])) or 1
        c["domain"] = "ref"
        c["prov_pool"] = POOL
        c["prov_en_xxh64"] = xxhash.xxh64(c["english"].encode("utf-8")).hexdigest()
        c["prov_index"] = version
        per[c["iso"]].append(c)
    stats = {}
    for iso, lst in sorted(per.items()):
        t = pa.table({k: [c[k] for c in lst] for k in schema_cols})
        pq.write_table(t, os.path.join(a.out, f"{iso}.parquet"), compression="zstd")
        stats[iso] = {"rows": len(lst), "b64": sum(c["length_bin"] == "b64" for c in lst), "b256": sum(c["length_bin"] == "b256" for c in lst),
                      "by_source": dict(collections.Counter(c["ref_source"] for c in lst)), "licences": sorted({c["ref_licence"] for c in lst})}
    json.dump({"pool": POOL, "index": version, "dropped": dict(drop), "protected_hits": dict(hits), "languages": stats, "rows": len(final), "max_per_lang": a.max_per_lang,
               "max_per_source_lang": a.max_per_source_lang, "languages_left_out": small}, open(os.path.join(a.out, "stats_ref.json"), "w"), indent=1)
    print(f"wrote {len(final)} rows, {len(per)} languages to {a.out}")
    for iso, d in sorted(stats.items()):
        print(f"  {iso}: {d['rows']} (b64 {d['b64']}, b256 {d['b256']}) {d['by_source']}")


if __name__ == "__main__":
    main()
