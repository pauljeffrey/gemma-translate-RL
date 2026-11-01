#!/usr/bin/env python3
"""r4: raw input of scripts/build_lexicons.py (`prep` then `train`) from the CLEAN SFT pairs (agent D2 out/kept + out/specialist), English -> X rows of the lexicon languages.

    python r4_lex_raw.py --kept <D2 out/kept> <D2 out/specialist> --out lex_raw [--cap 600000]
    python build_lexicons.py prep  --raw lex_raw --pivot empty_pivot.parquet --out lex_corpus
    python build_lexicons.py train --corpus lex_corpus --out lexicon_clean [--iso kab fon ...]

Writes lex_raw/<iso>.parquet with the columns prep_language expects: group_key ("sft:" + English, which prep reads as the English sentence), x, lid (1: no LID filter), source, prov
(religious_jw for the mt560 Bible / JW sources, else empty). The kept corpus is already decontaminated against the protected index, so no benchmark English enters the pairs; the lexicons are
then VALIDATED on H_cal (scripts/validate_lexicons.py with the H_cal file) and only validated ones get a weight. Also writes empty_pivot.parquet (prep wants a pivot map; unused here).
"""
import argparse, collections, glob, hashlib, os
import pyarrow as pa, pyarrow.parquet as pq

WEAK = ("efi bam fon kab kam kbp kmb mos nus umb dyu sag swc loz ven iso tiv kqn wal lun kua kon ibb urh nbl bin dag oke shi bbj kea ach tsc vmw seh nyn her yao sid ndc "
        "kri ish dua bas wes").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--cap", type=int, default=600000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = collections.defaultdict(lambda: ([], [], []))
    for f in sorted(x for d in a.kept for x in glob.glob(os.path.join(d, "*.parquet"))):
        pf = pq.ParquetFile(f); names = pf.schema_arrow.names
        if "target" not in names or "english" not in names:
            continue
        cols = [c for c in ("english", "target", "iso", "source_dataset", "flipped") if c in names]
        for b in pf.iter_batches(batch_size=200000, columns=cols):
            d = b.to_pydict()
            for k in range(len(d["target"])):
                if d.get("flipped") and str(d["flipped"][k]) in ("True", "true", "1"):
                    continue
                iso = d["iso"][k]
                if iso not in WEAK:
                    continue
                e, x, s = rows[iso]
                if len(e) < a.cap:
                    e.append(d["english"][k]); x.append(d["target"][k]); s.append(d["source_dataset"][k] if "source_dataset" in d else os.path.basename(f))
    for iso, (e, x, s) in rows.items():
        pq.write_table(pa.table({"group_key": ["sft:" + t for t in e], "x": x, "lid": [1] * len(e), "source": s,
                                 "prov": ["religious_jw" if ("mt560" in z or "jw" in z.lower() or "bible" in z.lower()) else "" for z in s]}), os.path.join(a.out, iso + ".parquet"))
        print(iso, len(e), flush=True)
    pq.write_table(pa.table({"group_key": ["none"], "pivot_en": [""]}), os.path.join(os.path.dirname(os.path.abspath(a.out)), "empty_pivot.parquet"))


if __name__ == "__main__":
    main()
