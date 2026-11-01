#!/usr/bin/env python3
"""GlotLID precision / recall per language on the TRAINING targets of a stage-B shard (growth 1, 125 languages).

    python lid_measure.py --shard-prefix <S3_PREFIX> --per-lang 800 --out /tmp/lid_out

Texts: translation targets of the shard (direction en2x, first target token = the language's single-token tag), decoded with the Gemma tokenizer, tag removed, 30-500 characters, up to --per-lang
random samples per language (a class-balanced pool). This is the text the model learned, mostly web text (mined bitext, Bible, news), not clean human references: a lower bound for recall, noisy
labels included. Per language L with the accepted label set A(L) (calibration `lid.labels` for the 77 RL languages, the table's `glotlid` for the others):
    recall(L)    = share of L's texts whose GlotLID top-1 label is in A(L)
    precision(L) = of all pooled texts whose top-1 label is in A(L): the share that really are L (pool = all languages, equal weight)
    undet(L)     = share of L's texts labelled und_* / zxx_* (GlotLID's 'undetermined script' / 'non-linguistic' classes)
    recall@p     = recall when only predictions with probability >= p count (0.3 / 0.5 / 0.7), the open-set variant of the guard
and the three most frequent wrong labels. Output: lid_precision_recall.csv + summary.txt.
"""
import argparse, collections, csv, json, os, sys
import numpy as np


def env_keys():
    for l in open("/path/to/repo/.env"):
        if l.startswith("S3_ACCESS_KEY_ID="): ak = l.strip().split("=", 1)[1]
        if l.startswith("S3_SECRET_ACCESS_KEY="): sk = l.strip().split("=", 1)[1]
        if l.startswith("HF_API_KEY="): os.environ["HF_TOKEN"] = l.strip().split("=", 1)[1]
    return ak, sk


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shard-prefix", required=True); ap.add_argument("--per-lang", type=int, default=800); ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="/path/to/repo/gemma_mt"); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--dump-only", action="store_true", help="write texts.tsv (iso<TAB>text) and stop before GlotLID")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); sys.path.insert(0, a.repo)
    import boto3, fasttext
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    from gemma_mt import languages as L, tags
    ak, sk = env_keys()
    s3 = boto3.client("s3", endpoint_url=os.environ['S3_ENDPOINT_URL'], aws_access_key_id=ak, aws_secret_access_key=sk, region_name=os.environ.get("S3_REGION", "us-east-1"))
    d = os.path.join(a.out, "shard"); os.makedirs(d + "/train", exist_ok=True)
    for n in ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "direction", "pack_start", "tokens.u32.bin"):
        fn = n if n.endswith(".bin") else n + ".npy"; dst = f"{d}/train/{fn}"
        key = f"{a.shard_prefix}/train/{fn}"
        if not os.path.exists(dst) or os.path.getsize(dst) != s3.head_object(Bucket=os.environ['S3_BUCKET'], Key=key)["ContentLength"]:
            print("download", key, flush=True); s3.download_file(os.environ['S3_BUCKET'], key, dst)
    A = {n: np.load(f"{d}/train/{n}.npy", mmap_mode="r") for n in ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "direction")}
    tok = np.memmap(f"{d}/train/tokens.u32.bin", dtype=np.uint32, mode="r")
    tk = Tokenizer.from_file(hf_hub_download("google/gemma-3-270m-it", "tokenizer.json", cache_dir="/tmp/lid_cache"))
    iso_table = L.ALL_ISOS; tag_id = {i: tk.token_to_id(tags.tag_token(i)) for i in iso_table}
    eot = tk.token_to_id("<end_of_turn>")
    iso_id = np.asarray(A["iso_id"]); direction = np.asarray(A["direction"]); so = np.asarray(A["sample_offset"]).astype(np.int64)
    pl = np.asarray(A["prompt_len"]).astype(np.int64); tl = np.asarray(A["target_len"]).astype(np.int64)
    first = np.where(tl > 0, tok[np.minimum(so + pl, len(tok) - 1)], 0)
    rng = np.random.default_rng(a.seed); texts, truth = [], []
    for li, iso in enumerate(iso_table):
        cand = np.flatnonzero((iso_id == li) & (direction == 0) & (first == tag_id[iso]))
        if len(cand) == 0: continue
        cand = cand[rng.permutation(len(cand))][: a.per_lang * 4]; got = 0
        for s in cand:
            ids = tok[so[s] + pl[s] + 1: so[s] + pl[s] + tl[s]].tolist()
            if ids and ids[-1] == eot: ids = ids[:-1]
            t = tk.decode(ids, skip_special_tokens=True).replace("\n", " ").strip()
            if 30 <= len(t) <= 500: texts.append(t); truth.append(iso); got += 1
            if got >= a.per_lang: break
    print("texts", len(texts), "languages", len(set(truth)), flush=True)
    with open(os.path.join(a.out, "texts.tsv"), "w") as f:
        for i, t in zip(truth, texts): f.write(f"{i}\t{t}\n")
    if a.dump_only: return
    model = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model_v3.bin", cache_dir="/tmp/lid_cache"))
    pred, prob = [], []
    for t in texts:
        lab, p = model.predict(t, k=1); pred.append(lab[0].replace("__label__", "")); prob.append(float(p[0]))
    cal = json.load(open(os.path.join(a.repo, "calibration", "shaping_flores_dev_v5.json")))
    acc = {}
    for iso in iso_table:
        lab = (cal.get(iso, {}).get("lid", {}) or {}).get("labels")
        acc[iso] = set(lab) if lab else set(L.LANGUAGES[iso].glotlid)
    v6 = os.path.join(a.repo, "calibration", "lid_labels_v6.json")
    if os.path.exists(v6):
        for iso, d in json.load(open(v6)).items():
            if not iso.startswith("_") and d["labels"]: acc[iso] = set(d["labels"])             # fixed label sets of 2026-11-01
    truth = np.array(truth); pred = np.array(pred); prob = np.array(prob)
    rows = []
    for iso in sorted(set(truth)):
        m = truth == iso; A_ = acc[iso]
        hit = np.array([p in A_ for p in pred])
        rec = hit[m].mean(); n_pred = hit.sum(); prec = (hit & m).sum() / n_pred if n_pred else float("nan")
        und = np.mean([p.startswith(("und_", "zxx_")) for p in pred[m]])
        rp = {th: float((hit[m] & (prob[m] >= th)).mean()) for th in (0.3, 0.5, 0.7)}
        wrong = collections.Counter(pred[m][~hit[m]]).most_common(3)
        exists = any(x in set(open("/tmp/glotlid_labels.txt").read().replace("__label__", "").split("\n")) for x in A_)
        rows.append(dict(iso=iso, name=L.LANGUAGES[iso].name, n=int(m.sum()), accepted=" ".join(sorted(A_)), label_exists=exists, recall=round(float(rec), 3), precision=round(float(prec), 3), undetermined=round(float(und), 3),
                         recall_p30=round(rp[0.3], 3), recall_p50=round(rp[0.5], 3), recall_p70=round(rp[0.7], 3), top_wrong=" ".join(f"{k}:{v}" for k, v in wrong), is_new=iso_table.index(iso) >= 85))
    with open(os.path.join(a.out, "lid_precision_recall.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    new = [r for r in rows if r["is_new"]]
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(f"texts {len(texts)} languages {len(rows)}\nmean recall all {np.mean([r['recall'] for r in rows]):.3f} | new {np.mean([r['recall'] for r in new]):.3f}\n")
        f.write("new languages: iso recall precision undet r@0.5 top_wrong\n")
        for r in sorted(new, key=lambda r: r["recall"]): f.write(f"{r['iso']} {r['recall']} {r['precision']} {r['undetermined']} {r['recall_p50']} {r['top_wrong']}\n")
    print(open(os.path.join(a.out, "summary.txt")).read())


if __name__ == "__main__":
    main()
