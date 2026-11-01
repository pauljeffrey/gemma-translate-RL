#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU probe of reference-free scorers for X -> Y translation (D-46, deliverable 4): is a score of (African source, African translation) usable as a reward?

For each language pair of a small panel the FLORES+ DEV sentences (multi-way, human translations; dev is the calibration split, never trained on) are used as
true pairs; the contrast sets are
    mismatch   (src_i, hyp_j)  hyp of another sentence       -> AUC = P(score(true) > score(mismatch))     the discrimination the reward needs most
    neighbour  (src_i, hyp_{i+-1..3})  a neighbouring sentence of the same document (topic and style kept)  -> the hard contrast
    copy       (src_i, src_i)  the source copied as the answer                                          -> failure rate = P(score(copy) >= score(true))
Scorers (all embed each unique text ONCE and score any pair from the embeddings, so a panel of L languages costs L x N embeddings, not L^2 x N):
    africomet_qe   masakhane/africomet-qe-stl-1.1  (estimator on the two sentence embeddings)
    ssacomet_qe    McGill-NLP/ssa-comet-qe         (same code path)
    cos_africomet  cosine of the two AfriCOMET sentence embeddings (no estimator)
    labse          sentence-transformers/LaBSE via transformers (CLS pooler output, L2-normalised), cosine; only for the languages LaBSE was trained on
The English-source baseline (eng_i -> hyp_i) is measured on the same sentences, so the loss of validity when the source is African is a difference of AUCs.

    python scripts/afri2afri_reward_probe.py --flores /workspace/sft_build/ssa_cmp/flores --bench-dev /workspace/sft_build/bench/bench_dev.jsonl --out probe.json \
        --langs yor hau ibo swh zul xho afr amh som kin --n 120 --threads 8 [--labse]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

MODELS = {"africomet_qe": "masakhane/africomet-qe-stl-1.1", "ssacomet_qe": "McGill-NLP/ssa-comet-qe"}
LABSE_LANGS = {"afr", "amh", "hau", "ibo", "mlg", "nya", "kin", "sna", "som", "sot", "swh", "wol", "xho", "yor", "zul"}      # LaBSE's 109 languages that are also ours
FLORES_FILE = {"yor": "yor_Latn", "hau": "hau_Latn", "ibo": "ibo_Latn", "swh": "swh_Latn", "zul": "zul_Latn", "xho": "xho_Latn", "afr": "afr_Latn", "amh": "amh_Ethi",
               "som": "som_Latn", "kin": "kin_Latn", "nya": "nya_Latn", "sna": "sna_Latn", "sot": "sot_Latn", "tsn": "tsn_Latn", "wol": "wol_Latn", "ewe": "ewe_Latn",
               "lin": "lin_Latn", "lug": "lug_Latn", "tir": "tir_Ethi", "orm": "gaz_Latn"}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(tie), all pairs (rank formulation, O(n log n))."""
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_multiway(flores_dir: str, bench_dev: str, langs, n: int, seed: int):
    """{iso: {id: text}} for English and the panel (FLORES+ dev ids are shared by all languages)."""
    def read(path):
        out = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[int(r["id"])] = r["text"]
        return out

    data = {"eng": read(os.path.join(flores_dir, "eng_Latn.jsonl"))}
    bd = {}
    if bench_dev and os.path.exists(bench_dev):
        with open(bench_dev, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("benchmark") == "flores_plus_dev":
                    bd.setdefault(r["iso"], {})[int(str(r["id"]).split("-")[-1])] = r["reference"]
    for l in langs:
        p = os.path.join(flores_dir, FLORES_FILE[l] + ".jsonl")
        if os.path.exists(p):
            data[l] = read(p)
        elif l in bd:
            data[l] = bd[l]
        else:
            raise FileNotFoundError(f"no FLORES+ dev text for {l}")
    ids = sorted(set.intersection(*[set(v) for v in data.values()]))
    rng = random.Random(seed)
    rng.shuffle(ids)
    ids = sorted(ids[:n])
    return {k: [v[i] for i in ids] for k, v in data.items()}


def comet_embed(key: str, texts, threads: int, batch: int = 16):
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(threads)
    from comet import download_model, load_from_checkpoint

    m = load_from_checkpoint(download_model(MODELS[key]))
    m.eval()
    m.to("cpu")
    uniq = sorted(set(texts), key=len)
    emb = {}
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(uniq), batch):
            b = uniq[s:s + batch]
            enc = m.encoder.prepare_sample(b)
            e = m.get_sentence_embedding(enc["input_ids"], enc["attention_mask"])
            for t, v in zip(b, e):
                emb[t] = v.detach().clone()
            if (s // batch) % 20 == 0:
                print(f"  {key}: {s + len(b)}/{len(uniq)} texts {time.time() - t0:.0f}s", flush=True)
    return m, emb


def comet_scores(m, emb, srcs, mts):
    import torch

    out = []
    with torch.no_grad():
        for s in range(0, len(srcs), 256):
            S = torch.stack([emb[x] for x in srcs[s:s + 256]])
            M = torch.stack([emb[x] for x in mts[s:s + 256]])
            x = torch.cat((M, S, M * S, torch.abs(M - S)), dim=1)
            out += m.estimator(x).view(-1).float().tolist()
    return np.array(out)


def cos_scores(emb, srcs, mts):
    import torch

    S = torch.stack([emb[x] for x in srcs])
    M = torch.stack([emb[x] for x in mts])
    return torch.nn.functional.cosine_similarity(S, M, dim=1).numpy()


def labse_embed(texts, threads: int):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    name = "setu4993/LaBSE"
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
    uniq = sorted(set(texts), key=len)
    emb = {}
    with torch.no_grad():
        for s in range(0, len(uniq), 32):
            b = uniq[s:s + 32]
            x = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=256)
            e = torch.nn.functional.normalize(model(**x).pooler_output, dim=-1)
            for t, v in zip(b, e):
                emb[t] = v
    return emb


def contrast_sets(src, hyp, seed):
    K = len(src)
    rng = random.Random(seed)
    mism = [hyp[(i + rng.randrange(7, K - 7)) % K] for i in range(K)]
    neigh = [hyp[(i + rng.choice([-3, -2, -1, 1, 2, 3])) % K] for i in range(K)]
    return mism, neigh


def _omit(text: str) -> str:
    """A translation that stops half way (the failure the omission detector of the en->X reward looks for): the first half of the words."""
    w = text.split()
    return " ".join(w[: max(1, len(w) // 2)])


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(1, len(sa | sb))


def guard_stats(data, langs):
    """Do the surface guards of the en->X reward (copy: word-set Jaccard >= 0.9; length-ratio window) fire on TRUE human pairs of the panel?"""
    out = {}
    for x in langs:
        for y in langs:
            if x == y:
                continue
            j = np.array([_jaccard(a, b) for a, b in zip(data[x], data[y])])
            lr = np.array([max(1, len(b)) / max(1, len(a)) for a, b in zip(data[x], data[y])])
            out[f"{x}>{y}"] = {"jaccard_mean": float(j.mean()), "jaccard_ge_0.9": float((j >= 0.9).mean()), "jaccard_ge_0.5": float((j >= 0.5).mean()),
                               "char_ratio_p05_p50_p95": [float(v) for v in np.quantile(lr, [0.05, 0.5, 0.95])]}
    return out


def evaluate(score_fn, data, langs, seed, with_en=True, with_rev=True):
    """score_fn(srcs, mts) -> np.array. Returns {'X>Y': {...}} for every ordered pair of the panel, 'eng>Y' (English source baseline) and 'X>eng'
    (African source, English translation: the orientation AfriCOMET documents)."""
    res = {}
    pairs = [(x, y) for x in langs for y in langs if x != y] + ([("eng", y) for y in langs] if with_en else []) + ([(x, "eng") for x in langs] if with_rev else [])
    for x, y in pairs:
        src, hyp = data[x], data[y]
        if len(set(src)) < 8:
            continue
        K = len(src)
        mism, neigh = contrast_sets(src, hyp, zlib.crc32(f"{seed}{x}{y}".encode()))
        T = score_fn(src, hyp)
        M, N = score_fn(src, mism), score_fn(src, neigh)
        O = score_fn(src, [_omit(h) for h in hyp])
        C = score_fn(src, src) if x != "eng" and y != "eng" else score_fn(src, hyp[::-1])      # copy of the source as the answer (not defined with English on a side)
        res[f"{x}>{y}"] = {"auc_mismatch": auc(T, M), "auc_neighbour": auc(T, N), "auc_half_omitted": auc(T, O), "copy_ge_true": float((C >= T).mean()), "mean_true": float(T.mean()),
                           "p10_p50_p90": [float(v) for v in np.quantile(T, [0.1, 0.5, 0.9])]}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flores", required=True)
    ap.add_argument("--bench-dev", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="+", default=["yor", "hau", "ibo", "swh", "zul", "xho", "afr", "amh", "som", "kin"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--models", nargs="*", default=["africomet_qe", "ssacomet_qe"])
    ap.add_argument("--labse", action="store_true")
    ap.add_argument("--save-emb", default=None, help="save the AfriCOMET sentence embeddings (fp16) for later calibration")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    data = load_multiway(a.flores, a.bench_dev, a.langs, a.n, a.seed)
    texts = [t for v in data.values() for t in v]
    texts = texts + [_omit(t) for t in texts]                    # the half-omitted contrast texts are embedded too
    out = {"n_sentences": a.n, "languages": a.langs, "seed": a.seed, "results": {}, "guards": guard_stats(data, a.langs)}
    for key in a.models:
        m, emb = comet_embed(key, texts, a.threads)
        if a.save_emb and key == "africomet_qe":
            import torch

            torch.save({k: v.half() for k, v in emb.items()}, a.save_emb)
        out["results"][key] = evaluate(lambda s, h: comet_scores(m, emb, s, h), data, a.langs, a.seed)
        if key == "africomet_qe":
            out["results"]["cos_africomet"] = evaluate(lambda s, h: cos_scores(emb, s, h), data, a.langs, a.seed)
        json.dump(out, open(a.out, "w"), indent=1)
    if a.labse:
        ls = [l for l in a.langs if l in LABSE_LANGS]
        emb = labse_embed([t for l in ["eng"] + ls for t in data[l]], a.threads)
        out["results"]["labse"] = evaluate(lambda s, h: cos_scores(emb, s, h), {k: data[k] for k in ["eng"] + ls}, ls, a.seed)
        out["labse_languages"] = ls
    json.dump(out, open(a.out, "w"), indent=1)
    print("done", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
