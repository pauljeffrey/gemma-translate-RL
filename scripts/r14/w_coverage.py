"""R14 widening: coverage of the rl_prompts_sci_v2 windows by a pseudo-reference directory (window level = every sentence translated; sentence level = chunk lookups), by length bin.
usage: w_coverage.py STORE_DIR OUT.json iso..."""
import collections, json, os, sys, unicodedata
import pyarrow.parquet as pq
from gemma_mt.rl.nllb_signal import key_of
from gemma_mt.rl.segments import split_sentences
nfc = lambda s: unicodedata.normalize("NFC", s)
store, out, langs = sys.argv[1], sys.argv[2], sys.argv[3:]
res = {}
for iso in langs:
    p = f"{store}/{iso}.parquet"
    if not os.path.exists(p): continue
    t = pq.read_table(p, columns=["key", "sentences", "translations"]).to_pydict()
    win = {k: all(tr) for k, tr in zip(t["key"], t["translations"])}
    sent = {key_of(iso, s) for ss, tr in zip(t["sentences"], t["translations"]) for s, x in zip(ss, tr) if x}
    pool = pq.read_table(f"/workspace/jobs/r7/assets2/rl_prompts_sci_v2/{iso}.parquet", columns=["english", "length_bin"]).to_pylist()
    c = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in pool:
        text = nfc(r["english"]); ss = split_sentences(text); b = r["length_bin"]
        c[b][0] += 1; c[b][1] += win.get(key_of(iso, text), False)
        c[b][2] += len(ss); c[b][3] += sum(key_of(iso, s) in sent for s in ss)
    tot = [sum(v[i] for v in c.values()) for i in range(4)]
    res[iso] = {"windows": tot[0], "window_cov": tot[1] / tot[0], "sentences": tot[2], "sentence_cov": tot[3] / tot[2],
                "by_bin": {b: {"windows": v[0], "window_cov": v[1] / v[0], "sentence_cov": v[3] / v[2]} for b, v in sorted(c.items())},
                "expected_draw_window_cov_uniform_bins": sum(v[1] / v[0] for v in c.values()) / len(c), "expected_draw_sentence_cov_uniform_bins": sum(v[3] / v[2] for v in c.values()) / len(c)}
    print(iso, round(res[iso]["window_cov"], 3), round(res[iso]["sentence_cov"], 3), round(res[iso]["expected_draw_sentence_cov_uniform_bins"], 3), flush=True)
json.dump(res, open(out, "w"), indent=1)
