#!/usr/bin/env python3
"""Small fastText language-ID classifier for the languages GlotLID cannot label (Nupe, Kongo, Fulah, Afar, Berber ...).

    python tools/language_id/train_fasttext_lid.py --train texts_train.tsv --test texts_test.tsv --out models/fasttext_lid125.ftz

Input files are `iso<TAB>text` lines (target sides of training pairs; see data_pipeline/sft_clean/fresh/lid_measure.py --dump-only). A dense model (dim 24, 100k hash buckets) is trained and then
product-quantised: 749 MB (dim 64, 2M buckets) -> about 1 MB at the same overall accuracy (0.864 against 0.870 on 77k held-out texts). Per-language recall and precision go to --report.
Limits: close siblings collapse (Zulu / Xhosa / Ndebele; Dyula / Bambara), so use it only for languages without a GlotLID label.
"""
import argparse, collections, csv, os
import numpy as np


def load(path):
    return [l.rstrip("\n").split("\t", 1) for l in open(path, encoding="utf-8") if "\t" in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="fasttext_lid_per_language.csv"); ap.add_argument("--dim", type=int, default=24); ap.add_argument("--bucket", type=int, default=100000)
    ap.add_argument("--epoch", type=int, default=12); ap.add_argument("--cutoff", type=int, default=50000); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import fasttext
    tr, te = load(a.train), load(a.test)
    tmp = a.out + ".train.txt"
    with open(tmp, "w", encoding="utf-8") as f:
        for iso, t in tr: f.write(f"__label__{iso} {t}\n")
    m = fasttext.train_supervised(tmp, dim=a.dim, bucket=a.bucket, epoch=a.epoch, lr=0.4, minn=2, maxn=5, loss="softmax", thread=a.threads, verbose=0)
    m.quantize(input=tmp, retrain=True, qnorm=True, cutoff=a.cutoff, dsub=2, epoch=4, lr=0.1, thread=a.threads, verbose=0)
    m.save_model(a.out); os.remove(tmp)
    pred = [m.predict(t.replace("\n", " "), k=1)[0][0].replace("__label__", "") for _, t in te]; truth = [i for i, _ in te]
    rows = []
    for iso in sorted(set(truth)):
        idx = [k for k, t in enumerate(truth) if t == iso]; npred = sum(p == iso for p in pred)
        rec = float(np.mean([pred[k] == iso for k in idx])); prec = sum(pred[k] == iso and truth[k] == iso for k in range(len(pred))) / max(1, npred)
        conf = collections.Counter(pred[k] for k in idx if pred[k] != iso).most_common(3)
        rows.append([iso, len(idx), round(rec, 3), round(prec, 3), " ".join(f"{x}:{y}" for x, y in conf)])
    with open(a.report, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "n_test", "recall", "precision", "top_confusions"]); w.writerows(rows)
    print("overall accuracy", round(float(np.mean([p == t for p, t in zip(pred, truth)])), 3), "| model MB", round(os.path.getsize(a.out) / 1e6, 1))


if __name__ == "__main__":
    main()
