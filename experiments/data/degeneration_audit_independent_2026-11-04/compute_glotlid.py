"""E-08 side check: run the REWARD's language identifier (GlotLID v3, cis-lmu/glotlid model_v3.bin, accepted labels = gemma_mt.languages glotlid, min 15 chars)
on the same rows, to compare its pass rates with the independent NLLB-218 LID. Output glot_<model>.parquet and glot_refs.parquet.
Usage: GLOT=/path/model_v3.bin python compute_glotlid.py [model ...]
"""
import os, sys
import numpy as np, pandas as pd
import fasttext
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "gemma_mt"))
import compute_features as CF
import gemma_mt.languages as GL

fasttext.FastText.eprint = lambda *a, **k: None
model = fasttext.load_model(os.environ["GLOT"])
models = sys.argv[1:] or ["sft", "stageA", "b1best", "mixedbest"]
for m in models:
    rows = CF.load(CF.FILES[m])
    labs, probs, _, _ = CF.lid_many(model, [CF.nfc(r["hyp"]) for r in rows])
    df = pd.DataFrame({"benchmark": [r["benchmark"] for r in rows], "iso": [r["iso"] for r in rows], "id": [str(r["id"]) for r in rows],
                       "glot_label": labs, "glot_prob": probs, "chars": [len(r["hyp"]) for r in rows]})
    acc = {i: set(L.glotlid) for i, L in GL.LANGUAGES.items()}
    df["glot_ok"] = [l in acc.get(i, set()) for l, i in zip(df.glot_label, df.iso)]
    df.to_parquet(os.path.join(HERE, f"glot_{m}.parquet"))
    if m == "sft":
        labs, probs, _, _ = CF.lid_many(model, [CF.nfc(r["reference"]) for r in rows])
        rdf = df[["benchmark", "iso", "id"]].copy()
        rdf["glot_label"], rdf["glot_prob"] = labs, probs
        rdf["glot_ok"] = [l in acc.get(i, set()) for l, i in zip(labs, rdf.iso)]
        rdf.to_parquet(os.path.join(HERE, "glot_refs.parquet"))
    print(m, len(df), flush=True)
