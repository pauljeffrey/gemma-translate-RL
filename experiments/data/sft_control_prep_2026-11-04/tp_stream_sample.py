#!/usr/bin/env python
"""Stream a deterministic sample of the English->X rows of qvac/TranslatePsy-AfriSLM-Synthetic-Mix (read-only, nothing stored but the extracted pairs).

Each of the 364 arrow shards holds ONE direction of ONE language pair (tp_shard_map.json, from the first-research scan R10). For every language we take SHARDS_PER_LANG en->xx
shards (first and middle of the language's shard list) and the first N_BATCHES record batches (1,000 rows each) of each. Output: one jsonl per language with
{source, target, shard, row}. Usage: HFT=<read token> python tp_stream_sample.py OUTDIR [N_BATCHES] [SHARDS_PER_LANG]
"""
import json, os, sys, time, collections
import pyarrow as pa
from huggingface_hub import HfFileSystem

NAME2ISO = {"Afrikaans": "afr", "Amharic": "amh", "Hausa": "hau", "Igbo": "ibo", "Kinyarwanda": "kin", "Lingala": "lin", "Luganda": "lug", "Malagasy": "mlg", "Nyanja": "nya",
            "Oromo": "orm", "Shona": "sna", "Somali": "som", "Southern Sotho": "sot", "Swahili": "swh", "Tswana": "tsn", "Wolof": "wol", "Xhosa": "xho", "Yoruba": "yor", "Zulu": "zul"}
HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1]; nb = int(sys.argv[2]) if len(sys.argv) > 2 else 25; spl = int(sys.argv[3]) if len(sys.argv) > 3 else 2
os.makedirs(out, exist_ok=True)
M = json.load(open(os.path.join(HERE, "tp_shard_map.json")))
en2x = collections.defaultdict(list)
for i, c, n in M:
    assert len(c) == 1
    s, t, p = c[0][0]
    if s == "English": en2x[t].append(i)
fs = HfFileSystem(token=os.environ["HFT"])
for name, iso in NAME2ISO.items():
    sh = sorted(en2x[name]); pick = sorted({sh[0], sh[len(sh) // 2], sh[-1]})[:spl] if spl >= 2 else [sh[0]]
    t0 = time.time(); n = 0
    with open(f"{out}/en2x_{iso}.jsonl", "w", encoding="utf-8") as f:
        for s in pick:
            with fs.open("datasets/qvac/TranslatePsy-AfriSLM-Synthetic-Mix/data-%05d-of-00364.arrow" % s, "rb") as fh:
                r = pa.ipc.open_stream(fh)
                row = 0
                for k, b in enumerate(r):
                    if k >= nb: break
                    cols = b.select(["source", "target", "source_lang", "target_lang"]).to_pydict()
                    for a, t, sl, tl in zip(cols["source"], cols["target"], cols["source_lang"], cols["target_lang"]):
                        assert sl == "English" and tl == name, (sl, tl, name, s)
                        f.write(json.dumps({"source": a, "target": t, "shard": s, "row": row}, ensure_ascii=False) + "\n"); row += 1; n += 1
    print(iso, name, "shards", pick, "rows", n, "%.1fs" % (time.time() - t0), flush=True)
