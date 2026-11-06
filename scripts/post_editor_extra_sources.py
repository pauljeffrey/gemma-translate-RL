#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-editor, selection v3: extra HUMAN en->X pair sources with clear provenance that are not in the clean pair shards. Writes parquet shards in the shard schema
(english, target, iso, source_dataset, source_family, licence, kind) that `post_editor_select_data.py` reads like any other shard (`--input DIR --subdirs .`).

    python scripts/post_editor_extra_sources.py --out DIR [--efik-windows efik_windows.parquet]

Sources (all human-translated; the policy classes are in SOURCE_RULES of the selection script):
  salt-train      Sunbird/salt `text-all` TRAIN split (23,947 English sentences translated by Sunbird's translators into lug, ach, nyn, teo, lgg, ttj, xog, swa, ibo; CC-BY-SA-4.0, gated auto-accept).
                  The evaluation's SALT benchmark uses the TEST split (protected); train is disjoint but in the same domain: results on SALT are reported separately.
  infopankki      Helsinki-NLP/opus_infopankki en-so (47k pairs, municipal information for immigrants, human, CC-BY-4.0)
  chichewa-mt     dmatekenya/chichewa-machine-translation (14k English-Chichewa sentences, agriculture / health / education topics, Apache-2.0, human)
  efik-nt-sheet   the owner's Efik NT sheet (EFIK_7K+.xlsx, Efik-KJV verse pairs; provenance / licence unconfirmed, tag bible_derived): read from the packed windows of the clean pipeline
                  (`extra_efi` windows, single verses), class `rel` (religious caps apply). It was in the SFT for several epochs.
Not added (checked): opus_gnome / opus_ubuntu / kde4 (no African pair with English in the HF versions), opus_wikipedia (no African pair), MALIBA bambara-mt (contains google/smol rows = benchmark text),
dsfsi/za-mafoko-* (NOODL licence), NaathNLP Nuer-Dinka (templated, provenance unstated), nllb / aya / crawl-mined sets (excluded by rule).
"""
import argparse, csv, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))


def _tok():
    for f in ("/root/.hf_env",):
        if os.path.exists(f):
            for line in open(f):
                if "=" in line: k, v = line.strip().split("=", 1); os.environ.setdefault(k, v)
    return os.environ.get("HF_TOKEN")


def write(rows, out, name):
    import pyarrow as pa, pyarrow.parquet as pq
    if not rows: print(name, "no rows"); return 0
    os.makedirs(out, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(out, f"{name}.parquet"))
    print(name, len(rows), "rows,", len({r["iso"] for r in rows}), "languages", flush=True)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--efik-windows", default=None); ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from gemma_mt.languages import ALL_ISOS, canonical_iso
    tok = _tok(); counts = {}
    if "salt" not in a.skip:
        p = hf_hub_download("Sunbird/salt", "text-all/train-00000-of-00001.parquet", repo_type="dataset", token=tok)
        t = pq.read_table(p).to_pylist(); rows = []
        for r in t:
            en = (r.get("eng_source_text") or "").strip()
            for col in ("lug", "ach", "nyn", "teo", "lgg", "ttj", "xog", "swa", "ibo"):
                iso = canonical_iso({"swa": "swh"}.get(col, col))
                tg = (r.get(f"{col}_text") or "").strip()
                if iso in ALL_ISOS and en and tg:
                    rows.append({"english": en, "target": tg, "iso": iso, "source_dataset": "Sunbird/salt:text-all:train", "source_family": "salt-train", "licence": "share_alike", "kind": "human"})
        counts["salt-train"] = write(rows, a.out, "salt_train")
    if "infopankki" not in a.skip:
        p = hf_hub_download("Helsinki-NLP/opus_infopankki", "en-so/train-00000-of-00001.parquet", repo_type="dataset", token=tok)
        rows = [{"english": x["translation"]["en"].strip(), "target": x["translation"]["so"].strip(), "iso": "som", "source_dataset": "Helsinki-NLP/opus_infopankki:en-so", "source_family": "infopankki",
                 "licence": "cc-by", "kind": "human"} for x in pq.read_table(p).to_pylist()]
        counts["infopankki"] = write(rows, a.out, "infopankki")
    if "chichewa" not in a.skip:
        p = hf_hub_download("dmatekenya/chichewa-machine-translation", "chichewa-machine-translation.csv", repo_type="dataset", token=tok)
        rows = [{"english": r["english"].strip(), "target": r["chichewa"].strip(), "iso": "nya", "source_dataset": "dmatekenya/chichewa-machine-translation", "source_family": "chichewa-mt", "licence": "apache", "kind": "human"}
                for r in csv.DictReader(open(p, encoding="utf-8")) if r.get("english") and r.get("chichewa")]
        counts["chichewa-mt"] = write(rows, a.out, "chichewa_mt")
    if a.efik_windows and "efik" not in a.skip:
        t = pq.read_table(a.efik_windows).to_pylist()
        rows = [{"english": r["english"].strip(), "target": r["target"].strip(), "iso": "efi", "source_dataset": "extra-bible:owner-efik-nt-xlsx (bible_derived; provenance / licence unconfirmed)", "source_family": "efik-nt-sheet",
                 "licence": "unspecified", "kind": "bible_aligned"} for r in t if r["english"].strip() and r["target"].strip()]
        counts["efik-nt-sheet"] = write(rows, a.out, "efik_nt")
    json.dump(counts, open(os.path.join(a.out, "extra_counts.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
