#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate README.md (the Hub dataset card) of the inter-African dataset from the builders' small outputs. Nothing is uploaded.

    python scripts/afri2afri_dataset_card.py --export-dir OUT/export --counts final_pair_counts.json --packed-meta packed/meta.json [--rl-report rl_report.json] --out README.md \
        --repo-id <HF_ORG_LEGACY>/interlaanguage-translation
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gemma_mt.languages import ALL_ISOS, LANGUAGES  # noqa: E402

SFT77 = [i for i in ALL_ISOS if i not in ("nde", "fat", "knc", "gaa", "men", "luy", "tzm", "ful")]
ISO1 = {"yor": "yo", "hau": "ha", "ibo": "ig", "swh": "sw", "kin": "rw", "som": "so", "orm": "om", "amh": "am", "tir": "ti", "lug": "lg", "lin": "ln", "zul": "zu", "xho": "xh",
        "afr": "af", "sna": "sn", "nya": "ny", "sot": "st", "tsn": "tn", "mlg": "mg", "ewe": "ee", "wol": "wo", "bam": "bm", "kik": "ki", "run": "rn", "ven": "ve", "tso": "ts"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--packed-meta", default=None)
    ap.add_argument("--rl-report", default=None)
    ap.add_argument("--full-dir", default=None, help="directory with full_pairs reports (_pivot_report.json, _direct_report.json, _fetch_log.json, _sizes.json)")
    ap.add_argument("--rl-v1-report", default=None, help="rl_report.json of the first RL build, for the before / after table")
    ap.add_argument("--repo-id", default=(os.environ['HF_ORG_LEGACY'] + "/interlaanguage-translation"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    man = json.load(open(os.path.join(a.export_dir, "MANIFEST.json")))
    counts = json.load(open(a.counts))
    import numpy as np

    v = np.array(list(counts.values()))
    files = man["files"]
    lic = man["licence_rows"].get("sft_pairs_train.parquet", {})
    srcs = man["source_dataset_rows"].get("sft_pairs_train.parquet", {})
    nl = sum(lic.values())
    lines = []
    A = lines.append
    A("---")
    A("license: other")
    A("license_name: per-row-source-licences")
    A("license_link: LICENSE")
    A("task_categories:\n- translation")
    A("pretty_name: Inter-African language translation (X to Y), SFT pairs and RL prompts")
    A("language:\n" + "\n".join(f"- {ISO1.get(i, i)}" for i in SFT77 if i in ISO1))
    A("size_categories:\n- 100M<n<1B")
    A("configs:\n- config_name: sft_pairs\n  data_files:\n  - split: train\n    path: sft_pairs_train.parquet\n  - split: validation\n    path: sft_pairs_val.parquet\n- config_name: rl_prompts\n  data_files:\n  - split: train\n    path: rl_prompts.parquet\n- config_name: full_pivot_en\n  data_files:\n  - split: train\n    path: full_pairs/pivot_en/*/*.parquet\n- config_name: full_direct\n  data_files:\n  - split: train\n    path: full_pairs/direct/*.parquet\n- config_name: full_nllb_mined\n  data_files:\n  - split: train\n    path: full_pairs/nllb_mined/*.parquet\n- config_name: full_pivot_en_sentences\n  data_files:\n  - split: train\n    path: full_pairs/pivot_en_sentences.parquet")
    A("---\n")
    A(f"# {a.repo_id.split('/')[-1]}: African language X -> African language Y\n")
    A("Training data for a **direct** (no English pivot) inter-African-language translation model: sentence-pair windows between 77 African languages (SFT) and monolingual "
      "source windows with a target-language tag (RL prompts). The 77 languages are listed in `direction_counts.csv` (ISO 639-3 style tags in the `src_iso` / `tgt_iso` columns; the YAML header lists only those with a two-letter code). Built from the data of the English -> African translation model (`gemma_mt`, D-46 in the project's DECISIONS.md).\n")
    A("## IMPORTANT: licences\n")
    A("**Every row keeps the licence of the source it was taken from.** The `licence` column holds the licence *declared on the source dataset's card* (`unknown` when the card "
      "declares none; we never guess one). A pair that joins two sources shows both, `a + b`. **Some rows are non-commercial, share-alike or of unknown licence; the dataset as a "
      "whole is therefore NOT under one licence. Check the `licence` and `source_dataset` columns and filter before you use the data** (for commercial use keep only the licences you "
      "may use; for CC-BY-SA material keep the share-alike obligation). `licence_class` is the coarser class used by the build (`cc-by`, `share_alike`, `unspecified`, ...). "
      "Reported licence distribution of the SFT training pairs:\n")
    unk = sum(n for k, n in lic.items() if "unknown" in k)
    oth = sum(n for k, n in lic.items() if "other" in k)
    A(f"**{100 * unk / nl:.1f} % of the SFT training rows have at least one side whose licence is `unknown`, {100 * oth / nl:.1f} % one side declared `other`.** No source in these rows declares a "
      "non-commercial licence (sources with NC licences were dropped by the en->X build, D-21), but the *original* texts behind some sources (JW / Bible material of the OPUS-MT560 sets) carry "
      "copyright of their own that the dataset cards do not clear.\n")
    A("| `licence` | rows | share |\n|---|---:|---:|")
    for k, n in list(lic.items())[:25]:
        A(f"| {k} | {n:,} | {100 * n / nl:.2f}% |")
    A("")
    A("## Files\n")
    A("| file | rows | content |\n|---|---:|---|")
    desc = {"sft_pairs_train.parquet": "SFT sentence pairs, one row per UNORDERED pair {src_iso, tgt_iso} (train a model on both directions)",
            "sft_pairs_val.parquet": "validation pairs (whole English sentences held out, never used with another partner in train)",
            "rl_prompts.parquet": "RL prompts: source-language window + target tag"}
    for k, d in sorted(files.items()):
        if k.startswith("stage/"):
            A(f"| {k.split('/')[-1]} | {d['rows']:,} | {desc.get(k.split('/')[-1], '')} |")
    A("")
    A("## What is in a row (SFT pairs)\n")
    A("`src_iso`, `tgt_iso` (ISO-639-3 style tags of `gemma_mt/languages.py`, `src_iso` sorts before `tgt_iso`), `src`, `tgt` (NFC text, windows of consecutive sentences are made at training time), "
      "`pivot_en` (the English sentence both sides translate, `null` for direct pairs), `origin` (`pivot_en` = two translations of the same English sentence; `direct` = a corpus that already "
      "pairs the two languages; `mined_nllb` = NLLB-v1 mined pair, filtered, **non-commercial upstream**), `provenance_kind` (`human_curated` | `mined_nllb` | `mixed` | `religious_jw` | `synthetic_mt`: a name-pattern heuristic, not an audit), `prompt_form`, `source_dataset`, `licence`, "
      "`licence_class`, `qe_src` / `qe_tgt` (AfriCOMET-QE of the English -> X translation of each side, NaN for direct pairs), `group_size` (number of languages that share the English "
      "sentence), `group_key`, `src_tok` / `tgt_tok` (Gemma tokens), `epochs` (how often the build planned to repeat the pair).\n")
    A("## Composition\n")
    A(f"* SFT train pairs: {files['stage/sft_pairs_train.parquet']['rows']:,}; ordered directions with data: {len(counts)} of {len(SFT77) * (len(SFT77) - 1)}; "
      f"with >= 1,000 pairs: {int((v >= 1000).sum())}; >= 100 pairs: {int((v >= 100).sum())}. Per-direction counts: `direction_counts.csv`.")
    A("* Sampling: UniMax over the unordered language pairs (uniform share of the budget, at most 3 passes over a pair's own data), so no rich pair dominates; the cap per language pair is a few thousand sentence pairs (<= 3.9 k).")
    A("* Domain warning: most multi-way material comes from Bible / JW-style parallel text (see `provenance_kind`); government text (South African languages), news and community corpora are a small share.")
    if a.packed_meta and os.path.exists(a.packed_meta):
        pm = json.load(open(a.packed_meta))
        t = pm["splits"]["train"]
        A(f"* The matching packed token dataset (not in this repo): {t['n_samples']:,} samples, {t['n_tokens']:,} Gemma tokens, seq_len {pm['seq_len']}; directions {t.get('directions', {})}; prompt forms {t.get('prompt_forms', {})}.")
    A("")
    if a.rl_report and os.path.exists(a.rl_report):
        rr = json.load(open(a.rl_report))
        rows = rr["rows_per_target"]
        A("## RL prompts\n")
        A(f"`rl_prompts.parquet`: {sum(rows.values()):,} prompts staged (before the private-source exclusion), one source-language window with a target tag, {rr['directions_with_prompts']} ordered directions "
          f"(median {rr['rows_per_direction_summary']['median']} prompts per direction, minimum {rr['rows_per_direction_summary']['min']}). Columns: `source`, `src_iso`, `iso` (target), `language`, `src_language`, "
          "`src_afr_tokens` (AfroXLM-R tokens), `length_bin` (b64 / b256 / b512 / b1024), `n_sentences`, `source_dataset`, `licence`, `licence_class`, `src_qe_valid`, `tgt_qe_valid`. Sources: real documents "
          "(HuggingFaceFW/fineweb-2, ODC-By-1.0; amh hau ibo pcm tir yor) and, for every language, text of the en->X corpus that is held out of the SFT pairs (neither the text nor its English sentence is in any SFT pair). "
          "Thin pools (kea, her, ish, wes, dua, oke, dyu, ibb) have only a handful of windows each and are reused many times.\n")
    if a.full_dir and os.path.exists(os.path.join(a.full_dir, "_pivot_report.json")):
        fp = json.load(open(os.path.join(a.full_dir, "_pivot_report.json")))
        fd = json.load(open(os.path.join(a.full_dir, "_direct_report.json"))) if os.path.exists(os.path.join(a.full_dir, "_direct_report.json")) else None
        flp = os.path.join(a.full_dir, "nllb_mined", "_fetch_log.json")
        fl = json.load(open(flp)) if os.path.exists(flp) else None
        mrp = os.path.join(a.full_dir, "_mined_rows_per_pair.json")
        mined_rows = json.load(open(mrp)) if os.path.exists(mrp) else None
        sz = json.load(open(os.path.join(a.full_dir, "_sizes.json"))) if os.path.exists(os.path.join(a.full_dir, "_sizes.json")) else {}
        A("## `full_pairs/`: the complete pair set (not a sample)\n")
        A("The files above are the SAMPLED training set (UniMax, a few thousand pairs per language pair). `full_pairs/` holds **everything we could derive**, sharded parquet (zstd), every row with `source_dataset`, "
          "`licence`, `licence_class`; the held-out evaluation sets and the rows of private source repositories are excluded exactly as above. One row per UNORDERED pair `{src_iso, tgt_iso}` "
          "(`src_iso` sorts before `tgt_iso`); a model uses both directions, so the number of ordered pairs is twice the number of rows. Nothing is sampled or thresholded: the quality signals are columns "
          "(`qe_*`, `lid_*`, `provenance_kind`, `score`), filter on them.\n")
        A("| folder | rows (unordered pairs) | ordered pairs | content |\n|---|---:|---:|---|")
        A(f"| `full_pairs/pivot_en/<src_iso>/part-*.parquet` | {fp['unordered_pairs_written']:,} | {fp['ordered_pairs']:,} | two translations of the same English sentence (join of the en->X corpus on the normalised English text); `group_key` links to `full_pairs/pivot_en_sentences.parquet` ({fp['english_sentences']:,} English sentences); `in_sft` marks the pairs selected for the sampled SFT files |")
        if fd:
            A(f"| `full_pairs/direct/direct_pairs.parquet` | {fd['written']:,} | {2 * fd['written']:,} | pairs of the SFT corpus that already pair two African languages (53 % machine translation, `provenance_kind` tells) |")
        if mined_rows:
            nrows = sum(mined_rows.values())
            A(f"| `full_pairs/nllb_mined/<src>-<tgt>.parquet` | {nrows:,} | {2 * nrows:,} | NLLB-v1 mined pairs republished as `michsethowusu/<lang>-<lang>_sentence-pairs` ({len(mined_rows)} shards; a few language pairs have a second shard `__<repo>` from a second repo). **No licence is declared on those cards; treated as non-commercial upstream**; raw mined data with a margin `score`, NOT filtered for quality or language |")
        A("")
        A("Held-out removal in `full_pairs`: " + f"{fp['held_out_rows_removed']:,} en->X rows (their pairs) matched an evaluation sentence by the 8-gram rule on top of the exact rule applied earlier; private-source rows removed: {fp['private_source_rows_removed']:,}." + (f" Mined pairs dropped for a held-out match: {sum(max(0, x['dropped_heldout']) for x in fl):,}." if fl else "") + (" Mined rows with an empty side are dropped (one repo, swahili-zulu, ships an empty Zulu column and is absent; `akan-twi` pairs a language with itself and is absent)." if fl else "") + "\n")
        A("Pivot pairs by provenance kind: " + ", ".join(f"{k} {v:,}" for k, v in fp["rows_by_provenance_kind"].items()) + ".\n")
        A("Licence distribution of `full_pairs/pivot_en` (declared `licence`, rows):\n")
        A("| licence | rows |\n|---|---:|")
        for k, v in list(fp["licence_rows"].items())[:15]:
            A(f"| {k} | {v:,} |")
        if fd:
            A("\n`full_pairs/direct` licences: " + ", ".join(f"{k} {v:,}" for k, v in list(fd["licence_rows"].items())[:8]) + ".\n")
        if sz:
            A("Sizes on disk: " + ", ".join(f"{k} {v / 1e9:.2f} GB" for k, v in sz.items()) + ".\n")
    if a.rl_v1_report and os.path.exists(a.rl_v1_report) and a.rl_report and os.path.exists(a.rl_report):
        r1, r2 = json.load(open(a.rl_v1_report)), json.load(open(a.rl_report))
        A("## RL prompts v2 (more real documents)\n")
        A("Source windows now also come from open web corpora streamed from the Hub (HuggingFaceFW/fineweb-2, cis-lmu/GlotCC-V1, HPLT/HPLT2.0_cleaned, wikimedia/wikipedia 20231101, allenai/c4 multilingual), "
          "filtered by GlotLID, de-duplicated and held out from the SFT pairs and the benchmarks; real documents are preferred over pseudo-paragraphs of translated sentences. Windows kept per source language (b64 + b256 + b512 + b1024) before -> after:\n")
        A("| language | before | after |\n|---|---:|---:|")
        for l, v in r2["per_source"].items():
            b = sum((r1["per_source"].get(l) or {}).values()) if r1["per_source"].get(l) else 0
            A(f"| {l} | {b:,} | {sum(v.values()) if v else 0:,} |")
        A("")
    A("## Prompt forms (SFT)\n")
    A("`prompt_form` (`explicit` | `implicit`) is a deterministic function of the pair (`blake2b` of `group_key|src_iso|tgt_iso`, 50 % each): `explicit` = `Translate the text from {source} to {target}:  {text}`, "
      "`implicit` = `Translate the text to {target}:  {text}` (the en->X-compatible form). The answer is always `{target tag} {translation}`.\n")
    A("## Filters\n")
    A("English sentence must be English and the African side must pass GlotLID for its language (where LID is reliable); per-language floor on the AfriCOMET-QE of the English -> X translation "
      "(30th percentile on top of the build's 15 % gate); length-ratio window per language pair; no copy of the source (identical, or word-set Jaccard >= 0.9); script check (Ethiopic languages); "
      "URL / markup debris removed; de-duplication on the text pair; validation = whole English-sentence groups.\n")
    A("## Held-out evaluation data (NOT in this dataset)\n")
    A("Nothing from these sets is in any file, by exact match after normalisation (any language) and by word-8-gram overlap: FLORES+ dev and devtest, NTREX-128, Google SMOL "
      "(SmolSent, SmolDoc, GATITOS), MAFAND-MT test, IbomMT, FLORES-MOZ, GlotStoryBook, SALT test, and the documents of the project's long-document benchmark. "
      "They are the benchmarks of the models trained on this data.\n")
    A("## Private / personal data\n")
    A("No personal data is knowingly included. Rows taken from source repositories that are flagged *private* in the build's licence scan were **not** staged for this repository "
      f"({sum(d['rows'] for k, d in files.items() if k.startswith('private_sources/')):,} rows).\n")
    A("## Sources (`source_dataset`, rows in the SFT training pairs)\n")
    A("| source_dataset | rows |\n|---|---:|")
    for k, n in list(srcs.items())[:60]:
        A(f"| {k} | {n:,} |")
    A("\nThe full list is in `MANIFEST.json`.\n")
    A("## Intended use and limits\n")
    A("Training and evaluating translation between African languages. Quality varies a lot by pair: pivot pairs inherit the noise of both English translations, `religious_jw` text is "
      "domain-narrow, `synthetic_mt` rows are machine translations (capped at 15 % per language pair in the build). No pair is human-verified at sentence level by us.\n")
    open(a.out, "w", encoding="utf-8").write("\n".join(lines))
    with open(os.path.join(os.path.dirname(a.out) or ".", "direction_counts.csv"), "w", encoding="utf-8") as f:
        f.write("src_iso,tgt_iso,pairs\n")
        for k, n in sorted(counts.items()):
            x, y = k.split(">")
            f.write(f"{x},{y},{n}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
