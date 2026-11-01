# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where does the greedy member's loop start? (R8, greedy-in-group assessment.) The greedy completion gets ONE advantage for all its tokens, so the share of its tokens BEFORE
the loop (good text that is pushed down together with the loop) matters.   python greedy_onset.py groups"""
import gzip, json, os, statistics, sys
from collections import defaultdict
from tokenizers import Tokenizer
from gemma_mt.eval.longdoc import repetition_loop

tk = Tokenizer.from_file("/workspace/sft_build/runs/final_1433/tokenizer.json")
res = defaultdict(list)
for fn in sorted(os.listdir(sys.argv[1])):
    if not fn.endswith(".jsonl.gz"):
        continue
    for l in gzip.open(os.path.join(sys.argv[1], fn), "rt"):
        r = json.loads(l)
        if r["kind"] != "greedy":
            continue
        text = tk.decode(r["ids"], skip_special_tokens=True)
        n = len(text.split())
        lp = repetition_loop(text)
        res[r["bin"]].append((r["finish"] == "FinishReason.STOP" or r["finish"].endswith("stop") or "stop" in r["finish"], lp["loop"], (lp["start_word"] / max(n, 1)) if lp["loop"] else None, lp["covered_share"] if lp["loop"] else 0.0, len(r["ids"])))
print("bin  n  finished  loop  loop_with_finish  pre-loop share of words (median over looping)  loop covered share (median)  mean tokens")
for b, rows in sorted(res.items()):
    loops = [x for x in rows if x[1]]
    print(b, len(rows), f"{sum(x[0] for x in rows)/len(rows):.2f}", f"{len(loops)/len(rows):.2f}", f"{sum(1 for x in loops if x[0])/max(len(loops),1):.2f}",
          f"{statistics.median(x[2] for x in loops):.2f}" if loops else "-", f"{statistics.median(x[3] for x in loops):.2f}" if loops else "-", f"{statistics.mean(x[4] for x in rows):.0f}")
