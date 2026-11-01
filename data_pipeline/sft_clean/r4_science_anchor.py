#!/usr/bin/env python3
"""r4: science_anchor refit on the CLEAN AfriScience-MT train set (agent S3's afriscience_clean_train.parquet, protected-index filtered). Same builder as
scripts/build_science_anchor_calibration.py (rl/sci_anchor.build_calibration); dev and test are not read. Numbers only are written.

    python r4_science_anchor.py --parquet afriscience_clean_train.parquet --code <repo gemma_mt/> --out science_anchor_clean_v1.json
"""
import argparse, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True); ap.add_argument("--code", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    import pyarrow.parquet as pq
    from gemma_mt.rl import sci_anchor
    t = pq.read_table(a.parquet, columns=["target_lang", "source", "target"]).to_pydict()
    rows = [(i, s, r) for i, s, r in zip(t["target_lang"], t["source"], t["target"]) if s and r]
    blob = sci_anchor.build_calibration(rows, min_n=200, min_n_language=150)
    blob["source"] = "dsfsi/afriscience_mt TRAIN, protected-index filtered by agent S3 (afriscience_clean_train), human references, sentence level; p10 of each feature over the pairs that have it"
    blob["n_pairs"] = len(rows)
    json.dump(blob, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(blob["n_pairs"], {k: [round(v["p10"], 3), round(v["p50"], 3), v["n"]] for k, v in blob["default"].items()})


if __name__ == "__main__":
    main()
