#!/usr/bin/env python3
"""Provisional per-language token table for the NEW translation data (hub/extra2k + hub/extra2_hold), before the final gate.

Per language: rows, Gemma tokens per pair (tokenizer sample), share already scored, share of the scored rows at or above the gate threshold (thr_cal; q15 of its own scores
for languages without a threshold), and the estimated kept rows and tokens (pass rate of the scored part applied to every row). The scored part is the first part of each file, so
the pass rates are provisional. The final numbers come from finish_all.sh (logs/delta_tokens_by_language.json).

    python provisional_tokens.py --root /workspace/work --out logs/provisional_tokens.csv
"""
import argparse, collections, glob, json, os, random, sys
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/workspace/work"); ap.add_argument("--out", default="logs/provisional_tokens.csv"); ap.add_argument("--sample", type=int, default=1500)
    a = ap.parse_args(); R = a.root; os.chdir(R)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it", token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    thr = json.load(open("work/gate/thresholds.json"))["languages"]
    ks, qs = [], []
    for f in glob.glob("work/scores_gpu/*.parquet") + glob.glob("work/scores_cpu/*.parquet"):
        t = pq.read_table(f, columns=["key", "qe"]); ks.append(t["key"].to_numpy()); qs.append(t["qe"].to_numpy().astype(np.float32))
    K = np.concatenate(ks); Q = np.concatenate(qs); o = np.argsort(K, kind="stable"); K, Q = K[o], Q[o]; del ks, qs
    print("scores", len(K), flush=True)
    old = np.sort(pq.read_table("work/gate_old/keep_final.parquet", columns=["key"])["key"].to_numpy())
    files = sorted(glob.glob("hub/extra2k/**/*.parquet", recursive=True)) + sorted(glob.glob("hub/extra2_hold/**/*.parquet", recursive=True))
    held = {f: ("extra2_hold" in f) for f in files}
    rows = collections.Counter(); scored = collections.Counter(); passed = collections.Counter(); chars = collections.Counter(); samples = collections.defaultdict(list); fil_rows = collections.Counter()
    rng = random.Random(7)
    for f in files:
        t = pq.read_table(f, columns=["english", "target", "iso", "key"])
        iso = np.array(t["iso"].to_pylist()); key = t["key"].to_numpy()
        keep_old = np.isin(key, old)                                  # already kept for A0 (no double counting)
        pos = np.clip(np.searchsorted(K, key), 0, len(K) - 1); has = K[pos] == key; q = np.where(has, Q[pos], np.nan)
        ch = np.array(pc.add(pc.utf8_length(t["english"]), pc.utf8_length(t["target"])).to_numpy(zero_copy_only=False), dtype=np.int64)
        for i in np.unique(iso):
            m = (iso == i) & ~keep_old; n = int(m.sum())
            if not n: continue
            L = thr.get(i, {}); th = L.get("thr_cal")
            rows[i] += n; chars[i] += int(ch[m].sum()); sc = m & has; scored[i] += int(sc.sum())
            if th is None:
                th = float(np.nanquantile(q[sc], 0.15)) if sc.sum() > 100 else -1e9
            passed[i] += int((q[sc] >= th).sum()); fil_rows[i] += 0
            if len(samples[i]) < a.sample:
                idx = np.flatnonzero(m); idx = rng.sample(list(idx), min(len(idx), 200))
                en = t["english"].take(pa.array(idx)).to_pylist(); tg = t["target"].take(pa.array(idx)).to_pylist(); samples[i] += list(zip(en, tg))
    out = []
    for i in sorted(rows, key=lambda x: -rows[x]):
        s = samples[i][:a.sample]
        tp = np.mean([len(tok(e, add_special_tokens=False)["input_ids"]) + len(tok(g, add_special_tokens=False)["input_ids"]) for e, g in s]) if s else 0.0
        tp += 12                                                        # instruction + tags + end of turn per pair (approximate)
        sh = scored[i] / rows[i]; pr = passed[i] / scored[i] if scored[i] else float("nan")
        out.append((i, rows[i], round(tp, 1), round(sh, 3), round(pr, 3) if scored[i] else "", int(rows[i] * (pr if scored[i] else 0.85)), round(rows[i] * (pr if scored[i] else 0.85) * tp / 1e6, 2), "hold" if any(held[f] for f in files if f.startswith("hub/extra2_hold") and i in f) else ""))
    with open(a.out, "w") as f:
        f.write("iso,new_rows,tokens_per_pair,scored_share,pass_rate_of_scored,est_kept_rows,est_kept_Mtokens,note\n")
        for r in out: f.write(",".join(map(str, r)) + "\n")
    print("languages", len(out), "new rows", sum(rows.values()), "est kept Mtokens", round(sum(r[6] for r in out), 1))


if __name__ == "__main__":
    main()
