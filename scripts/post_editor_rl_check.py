#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pre-launch check of the editor-RL reward: does each reward term (and the total) rank E1's edits the way the HUMAN REFERENCE does?

  gen    (venv_vllm)   python scripts/post_editor_rl_check.py gen --pool POOL.parquet --model E1_DIR --tokenizer TOK --out DIR [--per-language 20] [--samples 4]
                       E1 edits (greedy + N samples at T=1, top_p=1 = the RL sampling) of held-out reference prompts of the pool -> DIR/edits.jsonl, DIR/check_ids.txt
  score  (venv_train)  python scripts/post_editor_rl_check.py score --config configs/post_editor/pe_rl.yaml --edits DIR/edits.jsonl --out DIR
                       the real reward stack (QE ensemble + guards + EditorReward + reference-COMET worker) on groups [identity, greedy, samples...];
                       Spearman of every term and of the total with (a) the reference chrF++ gain and (b) the reference-COMET gain (21 languages); suggestions for tau.
                       Two totals: `total` (reference terms included: it contains the ground truth, so its correlation is partly circular) and `total_free`
                       (the same edits scored as a reference-FREE prompt: this is the honest figure for the ~45 % news prompts, and the launch rule uses it).
"""
import argparse, json, os, random, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))


def gen(a):
    import pandas as pd
    from transformers import AutoTokenizer
    from gemma_mt import post_edit as pe
    from gemma_mt.post_edit import build_pe_prompt, decode_gen

    df = pd.read_parquet(a.pool)
    df = df[df["reference"].astype(str).str.strip() != ""]
    rows = []
    rng = random.Random(a.seed)
    for iso, g in df.groupby("iso"):
        recs = g.to_dict("records"); rng.shuffle(recs)
        rows += recs[: a.per_language]
    print(f"[check] {len(rows)} reference prompts, {df['iso'].nunique()} languages", flush=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    backend = pe.make_backend("vllm", a.model, tok, "bfloat16", a.gpu_util, 2048, 100000)
    prompts = [build_pe_prompt(r["english"], r["draft"], r["iso"], tok, pe.MAX_DRAFT_TOKENS) for r in rows]
    gr = backend.greedy(prompts, 256)
    sm = backend.sample(prompts, n=a.samples, temperature=1.0, top_p=1.0, max_new=256, seed=a.seed)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "edits.jsonl"), "w", encoding="utf-8") as f, open(os.path.join(a.out, "check_ids.txt"), "w") as fi:
        for r, g, ss in zip(rows, gr, sm):
            edits = []
            for kind, o in [("greedy", g)] + [("sample", s) for s in ss]:
                text, fin, st = decode_gen(tok, o, r["iso"])
                edits.append({"kind": kind, "ids": o.ids, "text": text, "finished": fin, "status": st})
            f.write(json.dumps({"iso": r["iso"], "english": r["english"], "reference": r["reference"], "draft": r["draft"], "draft_system": r.get("draft_system", ""), "draft_category": r.get("draft_category", ""),
                                "pair_id": r.get("pair_id", ""), "edits": edits}, ensure_ascii=False) + "\n")
            fi.write(f"{r['iso']}\t{r['english']}\n")
    print("[check] done", flush=True)


def spearman(x, y):
    import numpy as np
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).correlation)


def score(a):
    import dataclasses
    import numpy as np
    from sacrebleu.metrics import CHRF
    from gemma_mt import format as fmt
    from gemma_mt.rl import post_edit_reward as per
    from gemma_mt.rl.comet import REF_MODEL_ID, WorkerScorer
    from gemma_mt.rl.post_edit_rl import EditorReward, editor_options, identity_completion
    from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
    from gemma_mt.rl.train_rl import RLConfig, build_scorer
    from gemma_mt.utils import dataclass_from_dict, load_yaml
    from transformers import AutoTokenizer

    d = load_yaml(a.config)
    cfg = dataclass_from_dict(RLConfig, d)
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    stop = fmt.resolve_stop_tokens(tok, None)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    scorer = build_scorer(cfg.scorer, "cuda")
    reward = TranslationReward(dataclass_from_dict(RewardConfig, cfg.reward), scorer, tok, stop.ids, lid=None, pad_id=pad_id)
    eo = editor_options(cfg.editor)
    rs = WorkerScorer(python=eo.ref_comet.get("python", sys.executable), model_id=eo.ref_comet.get("model_id", REF_MODEL_ID), batch_size=eo.ref_comet.get("batch_size", 32), fake=False,
                      pythonpath=eo.ref_comet.get("pythonpath"), reference_based=True)
    er = EditorReward(reward, eo, rs)
    chrf = CHRF(word_order=2)
    sc = lambda h, r: chrf.sentence_score(h, [r]).score

    recs = [json.loads(l) for l in open(a.edits, encoding="utf-8")]
    if a.limit:
        recs = recs[: a.limit]
    G = 1 + max(len(r["edits"]) for r in recs)
    samples, info, draft_ids, ident = [], [], [], []
    for g, r in enumerate(recs):
        did = identity_completion(tok, r["english"], r["draft"], r["iso"])
        draft_ids.append(did)
        row = [Sample(list(did), r["iso"], r["english"])] + [Sample(list(e["ids"]), r["iso"], r["english"]) for e in r["edits"]]
        while len(row) < G:
            row.append(Sample(list(did), r["iso"], r["english"]))
        samples += row
        ident.append(g * G)
        info.append({"english": r["english"], "draft": r["draft"], "iso": r["iso"], "reference": r["reference"]})
    t0 = time.time()
    out = reward(samples, group_size=G)                                   # consensus is off in the config: the QE ensemble, guards and failures
    print(f"[check] QE scored {len(samples)} samples in {time.time() - t0:.0f} s", flush=True)
    rew_ref, m = er(samples, out, info, G, draft_ids, identity_rows=ident)
    comps_ref = list(er.last_components)
    info_free = [dict(p, reference="") for p in info]
    rew_free, m2 = er(samples, out, info_free, G, draft_ids, identity_rows=ident)
    comps_free = list(er.last_components)
    # ground truth: human-reference chrF++ gain and reference-COMET gain of every edit over its draft
    rows = []
    items, pos = [], []
    for g, r in enumerate(recs):
        text_d = er.draft_scores([r["english"]], [r["draft"]], [r["iso"]], [draft_ids[g]])[0][1]
        cd = sc(text_d, r["reference"])
        for j in range(G):
            i = g * G + j
            ok = out.status[i] == "ok"
            text_e = out.texts[i] if ok else ""
            ce = sc(text_e, r["reference"]) if ok and text_e.strip() else 0.0
            rows.append({"g": g, "j": j, "iso": r["iso"], "kind": "identity" if j == 0 else (r["edits"][j - 1]["kind"] if j - 1 < len(r["edits"]) else "pad"), "ok": ok,
                         "dchrf": ce - cd, "same": text_e == text_d, "qe_gain": float(out.rewards[i]) - er._draft_cache[(r["iso"], r["english"], r["draft"])][0] if ok else float("nan")})
            if r["iso"] in er.covered and ok and text_e.strip():
                items.append((r["iso"], r["english"], text_e, r["reference"])); pos.append(i)
                items.append((r["iso"], r["english"], text_d, r["reference"])); pos.append(-1 - i)
    rc = er.ref_comet(items)
    rcd = {}
    for p, v in zip(pos, rc):
        rcd[p] = v
    for k, row in enumerate(rows):
        i = row["g"] * G + row["j"]
        e, dd = rcd.get(i), rcd.get(-1 - i)
        row["drc"] = (e - dd) if e is not None and dd is not None else float("nan")
        for name in ("gain", "ref", "refcomet", "ledger", "repair", "ungrounded", "churn", "copy", "total"):
            row[name] = getattr(comps_ref[k], name)
            row[name + "_free"] = getattr(comps_free[k], name)
        row["reward"] = rew_ref[k]; row["reward_free"] = rew_free[k]
    import pandas as pd
    df = pd.DataFrame(rows)
    df = df[(df["kind"] != "pad")]
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "check_rows.csv"), index=False)

    def demean(col, tgt):                                                  # within-group (what GRPO uses): both sides minus the group mean
        s = df.groupby("g")[col].transform("mean"); t = df.groupby("g")[tgt].transform("mean")
        return df[col] - s, df[tgt] - t

    res = {"n_prompts": len(recs), "n_samples": int(len(df)), "groups": G}
    terms = [("gain", "gain_free"), ("ref", None), ("refcomet", None), ("ledger", "ledger_free"), ("repair", "repair_free"), ("ungrounded", "ungrounded_free"), ("churn", "churn_free"), ("copy", "copy_free"),
             ("reward (with reference terms)", "reward"), ("reward_free (reference-free scoring)", "reward_free")]
    cov = df[df["drc"].notna()]
    table = []
    for label, col in terms:
        col = col or label
        if col not in df.columns:
            col = label
        entry = {"term": label, "nonzero_share": float((df[col].abs() > 1e-9).mean())}
        for tgt, base in (("dchrf", df), ("drc", cov)):
            b = base[base[col].notna() & base[tgt].notna()]
            entry[f"spearman_{tgt}"] = spearman(b[col], b[tgt])
            if len(b):
                x = b[col] - b.groupby("g")[col].transform("mean"); y = b[tgt] - b.groupby("g")[tgt].transform("mean")
                entry[f"spearman_within_{tgt}"] = spearman(x, y)
        table.append(entry)
    res["terms"] = table
    # by kind of edit and by draft category: how the free reward behaves on identity / greedy / samples
    res["mean_by_kind"] = {k: {"dchrf": float(g["dchrf"].mean()), "reward_free": float(g["reward_free"].mean()), "reward": float(g["reward"].mean()), "same": float(g["same"].mean())} for k, g in df.groupby("kind")}
    # noise: the QE gain of a sample that equals its draft is exactly 0 by construction; the informative number is how much chrF++ / COMET an edit with a small QE gain really gains
    e = df[(df["kind"] != "identity") & df["qe_gain"].notna() & ~df["same"]]
    bins = [-9, -0.5, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, 0.5, 9]
    qb = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        x = e[(e["qe_gain"] > lo) & (e["qe_gain"] <= hi)]
        if len(x):
            qb.append({"qe_gain_bin": f"({lo},{hi}]", "n": int(len(x)), "mean_dchrf": float(x["dchrf"].mean()), "share_dchrf_pos": float((x["dchrf"] > 0).mean()), "mean_drc": float(x["drc"].mean()) if x["drc"].notna().any() else None})
    res["qe_gain_bins"] = qb
    small = cov[(cov["kind"] != "identity") & (~cov["same"])]
    if len(small):
        res["refcomet_delta_abs_median"] = float(small["drc"].abs().median()); res["refcomet_delta_q75"] = float(small["drc"].abs().quantile(0.75))
    res["reward_free_vs_group"] = {"identity_mean_free": float(df[df.kind == "identity"]["reward_free"].mean()), "identity_mean": float(df[df.kind == "identity"]["reward"].mean())}
    res["metrics_reference_scoring"] = m; res["metrics_free_scoring"] = m2
    # top-1 agreement: the member with the highest reward per group vs the member with the highest reference chrF++ gain
    for col in ("reward", "reward_free"):
        wins = []
        for g, x in df.groupby("g"):
            if x["dchrf"].nunique() > 1:
                wins.append(float(x.loc[x[col].idxmax(), "dchrf"] >= x["dchrf"].median()))
        res[f"top1_above_median_{col}"] = float(np.mean(wins)) if wins else None
    json.dump(res, open(os.path.join(a.out, "check_summary.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--pool", required=True); g.add_argument("--model", required=True); g.add_argument("--tokenizer", required=True); g.add_argument("--out", required=True)
    g.add_argument("--per-language", type=int, default=20); g.add_argument("--samples", type=int, default=4); g.add_argument("--seed", type=int, default=7); g.add_argument("--gpu-util", type=float, default=0.8)
    s = sub.add_parser("score"); s.add_argument("--config", required=True); s.add_argument("--edits", required=True); s.add_argument("--out", required=True); s.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    gen(a) if a.cmd == "gen" else score(a)


if __name__ == "__main__":
    main()
