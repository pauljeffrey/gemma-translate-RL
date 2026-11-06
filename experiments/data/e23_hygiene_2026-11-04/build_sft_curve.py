"""E-23: copy the 385M SFT training/validation curve (stage A = g1a, stage B = g1b_s1..s6) into one CSV.
Source: research_archive_2026-11-03/clean_sft_small/run/runs/<run>/metrics.jsonl (laptop copy of the A100 run records).
Output: sft_385m_curve.csv (one row per logged step; val columns only at eval steps) and sft_385m_val_by_language_final.csv
Also checks the parameter count from the exported config."""
import json, csv, os, sys
ROOT = "research_archive_2026-11-03/clean_sft_small/run/runs"
OUT = os.path.dirname(os.path.abspath(__file__))
runs = ["g1a", "g1b_s1", "g1b_s2", "g1b_s3", "g1b_s4", "g1b_s5", "g1b_s6"]
rows = []
final_by_lang = None
for r in runs:
    for line in open(f"{ROOT}/{r}/metrics.jsonl"):
        d = json.loads(line)
        rows.append(dict(run=r, step=d["step"], epoch=d.get("epoch"), lr=d.get("lr"), train_loss=d.get("train_loss"),
                         grad_norm=d.get("grad_norm"), tokens_per_s=d.get("tokens_per_s"),
                         val_loss_micro=d.get("val_loss"), val_loss_macro=d.get("val_loss_macro"), val_tokens=d.get("val_tokens")))
with open(f"{OUT}/sft_385m_curve.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# final per-language validation loss from the last eval record of s6
last = None
for line in open(f"{ROOT}/g1b_s6/metrics.jsonl"):
    d = json.loads(line)
    if "val_loss_macro" in d: last = d
langs = {k.split("/", 1)[1]: v for k, v in last.items() if k.startswith("val_loss/")}
with open(f"{OUT}/sft_385m_val_by_language_final.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["iso", "val_loss"]); [w.writerow([k, v]) for k, v in sorted(langs.items())]
ev = [r for r in rows if r["val_loss_macro"] is not None]
print("rows", len(rows), "eval points", len(ev), "last step", last["step"], "macro", last["val_loss_macro"], "micro", last["val_loss"], "n_lang", len(langs))
print("first eval", ev[0]["run"], ev[0]["step"], ev[0]["val_loss_macro"], "| g1a last eval", [e for e in ev if e["run"]=="g1a"][-1])
for r in runs:
    e = [x for x in ev if x["run"] == r]
    lrs = [x["lr"] for x in rows if x["run"] == r and x["lr"] is not None]
    print(r, "steps", min(x["step"] for x in rows if x["run"]==r), max(x["step"] for x in rows if x["run"]==r), "max lr", max(lrs), "last lr", lrs[-1], "last val macro", e[-1]["val_loss_macro"] if e else None)
# parameter count from the exported config
c = json.load(open(f"{OUT}/model_config_stageA_best.json"))
h, i, L, V = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["vocab_size"]
nh, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
attn = h*nh*hd + 2*h*nkv*hd + nh*hd*h
mlp = 3*h*i
norms = 4*h + 2*hd
total = V*h + L*(attn+mlp+norms) + h
print("param count from config", total)
