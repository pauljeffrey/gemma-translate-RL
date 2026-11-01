"""Source list for the RL prompt top-up, taken from the catalog worker's `rl/` folder (owner 2026-10-11: "use the rl/ catalogue list to augment
rl_translation_prompts.bin").

    python -m africomet_translation.catalog_rl_sources --catalog data-catalog/african_hf --langs pcm,fon,efi,ewe,twi,yor,ibo,tsn,wol,urh \
        --exclude-repos repos_already_in_registry.txt --out catalog_rl_sources.json

For every target language the folder lists the datasets that contain it (`rl/<iso3>_<language>.csv`).  A dataset is used when it is commercial-OK or of
unspecified licence (non-commercial, NLLB-derived, evaluation / benchmark sets and mixed multi-language collections are left out), is not already a
source of the registry, and has at least one plain text column.  The text columns of the dataset are read (every language of the row is tried; GlotLID
keeps only the sentences of the target language), so a pair set contributes its non-English side.
Output: a JSON list [{lang, repo, cols, license, rows}].
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

csv.field_size_limit(10 ** 9)

ALIASES = {"twi": ("twi", "aka", "fat")}
SKIP_COLS = re.compile(r"(^id$|_id$|^idx$|^index$|url|link|audio|image|path|file|hash|^lang|language|^split$|timestamp|date|duration|speaker|^source$|^dataset$|"
                       r"score|similarity|label|category|^name$)", re.I)
ENGLISH_COLS = {"english", "eng", "en", "source_en", "english_translation", "english_sentence", "translation_en", "en_text"}
NLLB = re.compile(r"nllb|no language left behind", re.I)
EVALISH = re.compile(r"(^|[-_ /.])(eval|evals|evaluation|benchmark|bench|test|dev|devtest|leaderboard)([-_ .]|$)|flores|ntrex|afrimmlu|afrixnli", re.I)
NLLB_PAIR = re.compile(r"^michsethowusu/[a-z-]+-[a-z-]+_sentence-pairs$", re.I)       # the NLLB-v1 mined pairs (the `_mt560` sets are OPUS and stay)


def string_cols(columns: str) -> list:
    out = []
    for part in (columns or "").split(";"):
        if ":" not in part:
            continue
        n, t = part.split(":", 1)
        n, t = n.strip(), t.strip()
        if t.startswith(("string", "large_string")) and not SKIP_COLS.search(n) and n.lower() not in ENGLISH_COLS:
            out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--langs", required=True)
    ap.add_argument("--exclude-repos", type=Path, default=None)
    ap.add_argument("--max-rows", type=int, default=150_000)
    ap.add_argument("--folder", default="rl", help="catalog sub-folder to read (rl | pretrain)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    excl = {l.strip().lower() for l in a.exclude_repos.read_text().splitlines()} if a.exclude_repos and a.exclude_repos.exists() else set()
    out, skipped = [], {}
    for lang in a.langs.split(","):
        seen = set()
        for iso in ALIASES.get(lang, (lang,)):
            for f in sorted(glob.glob(str(a.catalog / a.folder / f"{iso}_*.csv"))):
                with open(f, encoding="utf-8", newline="") as fh:
                    for r in csv.DictReader(fh):
                        repo = r["dataset_id"]
                        why = None
                        if repo in seen:
                            continue
                        seen.add(repo)
                        if repo.lower() in excl:
                            why = "already_in_registry"
                        elif r["commercial_use"] == "non_commercial":
                            why = "non_commercial"
                        elif r["scope"] == "mixed_subset":
                            why = "mixed_multilanguage_collection"
                        elif NLLB.search(" ".join((repo, r.get("synthetic_or_mt_hints", ""), r.get("provenance_hints", ""), r.get("readme_summary", "")))) or NLLB_PAIR.match(repo):
                            why = "nllb"
                        elif EVALISH.search(repo.split("/", 1)[1]):
                            why = "evaluation"
                        elif re.search(r"speech-to-text|tts", r["tasks"]) and not re.search(r"pretrain|translation|question|chat|instruction", r["tasks"]):
                            why = "audio_only"
                        cols = string_cols(r["columns"])
                        if why is None and not cols:
                            why = "no_text_column"
                        if why:
                            skipped[why] = skipped.get(why, 0) + 1
                            continue
                        try:
                            rows = int(float(r["num_rows"] or 0))
                        except ValueError:
                            rows = 0
                        out.append({"lang": lang, "repo": repo, "cols": cols[:3], "license": r["license"] or "unspecified", "rows": rows,
                                    "max_rows": a.max_rows})
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    import collections
    print("sources:", len(out), dict(collections.Counter(o["lang"] for o in out)))
    print("skipped:", skipped)


if __name__ == "__main__":
    main()
