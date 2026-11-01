#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-61: rows (iso, cls, src, mt) for the QE check of the padded / stuffed attacks: held-out FLORES+ dev prompts of agent A5's policy samples; classes human, policy (T=1 sample of rl2c), the attacks
of rl/anchors_attacks.py, en_copy. Score with /workspace/jobs/a5/qe_choice_score.py (AfriCOMET-QE-STL-1.1 and SSA-COMET-MTL reference-free) and summarise with scripts/qe_attack_summary.py."""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_anchors_v2 as E
from gemma_mt.rl import anchors_attacks as AT
from gemma_mt.rl import lexicon as L

ap = argparse.ArgumentParser()
ap.add_argument("--cal", required=True)
ap.add_argument("--funcwords", required=True)
ap.add_argument("--policy", required=True)
ap.add_argument("--lexicon-dir", required=True)
ap.add_argument("--langs", default="fon,sag,umb,kab,kam,mos,kmb,kbp,kea,vmw,yor,hau,swh")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--out", required=True)
a = ap.parse_args()
cal = json.load(open(a.cal))
fwd = json.load(open(a.funcwords))
pol = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(a.policy, encoding="utf-8"):
    r = json.loads(l)
    pol[r["iso"]][r["id"]].append(r)
out = []
for iso in a.langs.split(","):
    prompts = [rows for rows in pol[iso].values() if E.fold_of(iso, rows[0]["src"]) == 1][: a.n]
    units = [{"src": rows[0]["src"], "ref": rows[0]["ref"]} for rows in prompts]
    lex = L.load_lexicon(iso, a.lexicon_dir)
    lex = lex if lex is not None and lex.validated else None
    fb = cal["languages"][iso].get("func") or {}
    fw = fb.get("fw") or [t[0] for t in fwd[iso]["top"][:80]]
    atk = AT.make_attacks(units, fwd[iso]["top"], fw, [u["ref"] for u in units], random.Random(5), lex=lex, fw_share=0.25)
    for j, rows in enumerate(prompts):
        src = rows[0]["src"]
        out.append({"iso": iso, "cls": "human", "src": src, "mt": rows[0]["ref"]})
        out.append({"iso": iso, "cls": "policy", "src": src, "mt": rows[0]["hyp"]})
        for name in ("stuff30", "pad", "pad_fit", "stuff_fw", "en_copy") + (("gloss", "gloss_pad", "gloss_fw") if lex else ()):
            out.append({"iso": iso, "cls": name, "src": src, "mt": atk[name][j]})
with open(a.out, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(out), "rows")
