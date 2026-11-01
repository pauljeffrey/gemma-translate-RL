#!/usr/bin/env python3
"""Clean AfriScience-MT TRAIN set for the science stage (agent S3, 2026-10-28).

dsfsi/afriscience_mt (Hub card: Apache-2.0; corpus config: train / dev / test, 6 target languages amh hau lug nso yor zul, multi-parallel English source). Only the TRAIN split is a
training resource; dev and test are protected sets (they are inside the protected index as afriscience_mt:dev / afriscience_mt:test and are only counted here, never written).
Every train pair is tested on BOTH sides (English source and African-language target) against the protected index (strict: no FLORES+ dev train-half exemption); pairs with any hit are
removed. Output: clean train parquet with provenance, a removal log without text, per-language / per-split statistics (pairs, papers, Gemma tokens).

    HF_TOKEN=... python s3_afriscience.py --index protected/protected_index.npz --manifest protected/protected_manifest.json --tok code/tok_gemma.json --out out_afs --work work_afs
"""
import argparse
import collections
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import xxhash  # noqa: E402

import s3_common as C  # noqa: E402

REPO = "dsfsi/afriscience_mt"
LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]


def h64(s):
    return format(xxhash.xxh64_intdigest((s or "").encode("utf-8", "surrogatepass")), "016x")


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--nproc", type=int, default=1)
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    from tokenizers import Tokenizer
    tokn = os.environ["HF_TOKEN"]
    api = HfApi(token=tokn)
    info = api.dataset_info(REPO)
    os.makedirs(a.out, exist_ok=True)
    tk = Tokenizer.from_file(a.tok)
    splits = {}
    for s in ("train", "dev", "test"):
        p = hf_hub_download(REPO, f"corpus/{s}-00000.parquet", repo_type="dataset", token=tokn, local_dir=a.work, revision=info.sha)
        splits[s] = pq.read_table(p).to_pydict()
    ix, ex, man = C.load_strict(a.index, a.manifest)
    chk = C.Checker(ix, ex, None, nproc=a.nproc)

    def ntok(texts):
        return [len(e.ids) for e in tk.encode_batch(list(texts), add_special_tokens=False)]

    # ---- 1. raw statistics of the three splits (dev / test only counted)
    rows_stats = []
    toks = {}
    for s, d in splits.items():
        en_t = ntok(d["source"])
        tg_t = ntok(d["target"])
        toks[s] = (en_t, tg_t)
        per = collections.defaultdict(lambda: dict(pairs=0, papers=set(), en_tok=0, tg_tok=0, en_chars=0, tg_chars=0, en_words=0, tg_words=0))
        for i in range(len(d["source"])):
            L = d["target_lang"][i]
            r = per[L]
            r["pairs"] += 1
            r["papers"].add(d["paper_id"][i])
            r["en_tok"] += en_t[i]
            r["tg_tok"] += tg_t[i]
            r["en_chars"] += len(d["source"][i])
            r["tg_chars"] += len(d["target"][i])
            r["en_words"] += len(d["source"][i].split())
            r["tg_words"] += len(d["target"][i].split())
        for L in LANGS:
            r = per[L]
            rows_stats.append(dict(split=s, iso=L, pairs=r["pairs"], papers=len(r["papers"]), en_gemma_tokens=r["en_tok"], target_gemma_tokens=r["tg_tok"], en_chars=r["en_chars"],
                                   target_chars=r["tg_chars"], en_words=r["en_words"], target_words=r["tg_words"], role=("TRAIN (training resource)" if s == "train" else "PROTECTED (never used; dev = selection, test = reported)")))
    # ---- 2. filter the train split on both sides
    d = splits["train"]
    n = len(d["source"])
    texts = sorted(set(d["source"]) | set(d["target"]))
    res = chk.run(texts)
    R = dict(zip(texts, res))
    en_t, tg_t = toks["train"]
    removed = np.zeros(n, bool)
    by_rule_set = collections.Counter()
    by_side_set = collections.Counter()
    by_lang_set = collections.Counter()
    removed_log = []
    exempt_only = 0
    for i in range(n):
        rs = R[d["source"][i]]
        rt = R[d["target"][i]]
        hs, ht = bool(rs[0] | rs[1] | rs[2]), bool(rt[0] | rt[1] | rt[2])
        if not (hs or ht):
            continue
        removed[i] = True
        names = set()
        for side, r in (("en", rs), ("target", rt)):
            for rule, m in zip(C.RULES, r[:3]):
                for nm in ix.names(m):
                    by_rule_set[f"{rule}|{nm}|{side}"] += 1
                    names.add(nm)
                    by_side_set[f"{nm}|{side}"] += 1
        for nm in names:
            by_lang_set[f"{d['target_lang'][i]}|{nm}"] += 1
        eo = all((r[3] is True) for r in (rs, rt) if (r[0] | r[1] | r[2]))
        exempt_only += int(eo)
        removed_log.append((d["paper_id"][i], d["sentence_id"][i], d["target_lang"][i], hs, ht, ";".join(sorted(names)), eo))
    # length ratio (flag only; the old mix_v1 dropped one Zulu pair with ratio outside [0.3, 3] x the language median)
    ratio = [len(d["target"][i]) / max(1, len(d["source"][i])) for i in range(n)]
    med = {L: statistics.median(r for r, l in zip(ratio, d["target_lang"]) if l == L) for L in LANGS}
    outlier = np.array([not (0.3 * med[d["target_lang"][i]] <= ratio[i] <= 3 * med[d["target_lang"][i]]) for i in range(n)])
    # English sentences with different outcomes across the 6 languages
    key_lang = collections.defaultdict(dict)
    for i in range(n):
        key_lang[(d["paper_id"][i], d["sentence_id"][i])][d["target_lang"][i]] = bool(removed[i])
    mixed = sum(1 for v in key_lang.values() if len(set(v.values())) > 1)
    all_removed = sum(1 for v in key_lang.values() if all(v.values()))
    # contiguity (previous kept sentence of the same paper and language has sentence_id - 1)
    order = sorted(range(n), key=lambda i: (d["target_lang"][i], d["paper_id"][i], d["sentence_id"][i]))
    contig = np.zeros(n, bool)
    last = {}
    for i in order:
        k = (d["target_lang"][i], d["paper_id"][i])
        if not removed[i]:
            contig[i] = (k in last and last[k] == d["sentence_id"][i] - 1)
            last[k] = d["sentence_id"][i]
        else:
            last[k] = None
    keep = ~removed
    kidx = np.nonzero(keep)[0]
    cols = {c: pa.array([d[c][i] for i in kidx]) for c in ("paper_id", "sentence_id", "domain", "lang_pair", "source_lang", "target_lang", "source", "target")}
    cols["en_gemma_tokens"] = pa.array([en_t[i] for i in kidx], pa.int32())
    cols["target_gemma_tokens"] = pa.array([tg_t[i] for i in kidx], pa.int32())
    cols["flag_len_ratio_outlier"] = pa.array(outlier[kidx], pa.bool_())
    cols["contiguous_with_previous_kept"] = pa.array(contig[kidx], pa.bool_())
    cols["prov_repo"] = pa.array([REPO + "@" + info.sha[:10] + ":corpus/train-00000.parquet"] * len(kidx))
    cols["prov_row"] = pa.array(kidx.astype(np.int64))
    cols["prov_en_xxh64"] = pa.array([h64(d["source"][i]) for i in kidx])
    cols["prov_tgt_xxh64"] = pa.array([h64(d["target"][i]) for i in kidx])
    cols["prov_index"] = pa.array([man["version"]] * len(kidx))
    cols["licence"] = pa.array(["apache-2.0"] * len(kidx))
    pq.write_table(pa.table(cols), os.path.join(a.out, "afriscience_clean_train.parquet"), compression="zstd")
    rl = list(zip(*removed_log)) if removed_log else [[]] * 7
    pq.write_table(pa.table({"paper_id": pa.array(rl[0], pa.int64()), "sentence_id": pa.array(rl[1], pa.int64()), "target_lang": pa.array(rl[2]), "hit_en_side": pa.array(rl[3], pa.bool_()),
                             "hit_target_side": pa.array(rl[4], pa.bool_()), "protected_sets": pa.array(rl[5]), "only_flores_dev_train_half": pa.array(rl[6], pa.bool_())}),
                   os.path.join(a.out, "afriscience_train_removed_log.parquet"), compression="zstd")
    # ---- 3. per-language clean-train statistics + document length distribution
    clean_stats = []
    for L in LANGS:
        idx = [i for i in range(n) if d["target_lang"][i] == L]
        kept = [i for i in idx if keep[i]]
        papers_b = {d["paper_id"][i] for i in idx}
        papers_a = {d["paper_id"][i] for i in kept}
        # whole-paper lengths (English + target + 24 tokens for prompt, tag and end-of-turn), before and after
        def paper_len(sel):
            tot = collections.Counter()
            for i in sel:
                tot[d["paper_id"][i]] += en_t[i] + tg_t[i]
            return [v + 24 for v in tot.values()]
        pl = paper_len(kept)
        clean_stats.append(dict(
            iso=L, pairs_before=len(idx), pairs_after=len(kept), removed=len(idx) - len(kept), removed_pct=round(100 * (len(idx) - len(kept)) / len(idx), 3),
            papers_before=len(papers_b), papers_after=len(papers_a),
            en_gemma_tokens_after=sum(en_t[i] for i in kept), target_gemma_tokens_after=sum(tg_t[i] for i in kept),
            en_gemma_tokens_before=sum(en_t[i] for i in idx), target_gemma_tokens_before=sum(tg_t[i] for i in idx),
            flag_len_ratio_outlier_kept=int(sum(outlier[i] for i in kept)),
            paper_len_tokens_median=int(statistics.median(pl)), paper_len_p95=pct(pl, 0.95), paper_len_max=max(pl),
            papers_over_1024=sum(1 for x in pl if x > 1024), papers_over_4096=sum(1 for x in pl if x > 4096), papers_over_8192=sum(1 for x in pl if x > 8192),
            kept_contiguous_with_previous=int(sum(contig[i] for i in kept))))
    summary = dict(created=C.now(), repo=REPO, commit=info.sha, licence_card="apache-2.0 (Hub card_data, VERIFIED)", index_version=man["version"], index_sha256=man["index_file_sha256"],
                   mode="strict: FLORES+ dev train-half exemption NOT attached", train_pairs=n, train_pairs_removed=int(removed.sum()), train_pairs_removed_only_by_flores_dev_train_half=exempt_only,
                   english_sentences_with_mixed_outcome_across_languages=mixed, english_sentences_removed_in_all_six=all_removed, english_sentences=len(key_lang),
                   by_rule_set_side=dict(by_rule_set), by_set_side=dict(by_side_set), by_lang_set=dict(by_lang_set), length_ratio_median_by_lang=med)
    json.dump(dict(summary=summary, split_stats=rows_stats, clean_stats=clean_stats), open(os.path.join(a.out, "afriscience_stats.json"), "w"), indent=1)
    import csv
    for name, rows in (("afriscience_split_stats.csv", rows_stats), ("afriscience_clean_train_stats.csv", clean_stats)):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    # protected dev / test: paper ids and counts only (no text)
    with open(os.path.join(a.out, "afriscience_protected_papers.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "paper_id", "domain", "sentences_per_language"])
        for s in ("dev", "test"):
            dd = splits[s]
            cnt = collections.Counter()
            dom = {}
            for i in range(len(dd["source"])):
                if dd["target_lang"][i] == "amh":
                    cnt[dd["paper_id"][i]] += 1
                    dom[dd["paper_id"][i]] = dd["domain"][i]
            for pid in sorted(cnt):
                w.writerow([s, pid, dom[pid], cnt[pid]])
    chk.close()
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("by_")}, indent=1))
    print(json.dumps(clean_stats, indent=0)[:1500])


if __name__ == "__main__":
    main()
