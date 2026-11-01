#!/usr/bin/env python3
"""STEP C: specialist dataset for 19 weak languages (amh tir fon dyu kbp nus mos kam bam umb kea kmb vmw kab seh wol ewe luo tum).
Sources, in the priority used for the exact dedupe (first occurrence keeps its licence class):
  main        this build's decontaminated output (out/final), rows of the 19 languages
  weak_perm   <HF_ORG_LEGACY>/sabiyarn-weak-language-parallel-2026-10-26 train_ready/permissive            (the flores_plus_dev rows of that repo are NOT taken: replaced by data_devhalf)
  r10_comm    r10 backup sft_extra_v1/commercial_ok (GoURMET CC0, GlobalVoices, Tatoeba, FineTranslations = real target + synthetic English)
  weak_sa     weak repo train_ready/share_alike  (OLDI Seed, Wikimedia, Travis block ...)
  r10_sa      r10 backup sft_extra_v1/share_alike
  devhalf     data_devhalf (FLORES+ dev train half, share_alike)
  weak_syn    weak repo synthetic_english_side   (kind synthetic_english_side)
  r10_nc      r10 backup sft_extra_v1/NC_noncommercial (tagged non_commercial; kept only with that tag)
  gemma4      r10 backup sft_extra_v1/owner_code_switch  (Gemma-4-31B synthetic targets, kind synthetic_gemma4, owner-private)
NOT taken (listed in excluded.csv): weak repo quarantine/, r10 EXCLUDED_unknown_licence (CCAligned web-mined, bible-uedin unspecified), anything whose source name mentions NLLB / TranslatePsy.
Every row goes through the v.2 protected-set filter (both sides, strict rule, FLORES+ dev train-half exemption) and the exact dedupe. The old ok() length filter is applied only to rows that
come from the main output (already filtered); external rows keep the builder's filters (document-level FineTranslations chunks are longer than ok() allows).
Output specialist/pairs-0000.parquet and specialist/stats/*.csv ; Gemma target tokens are exact (tokenizer of google/gemma-3-270m-it).
"""
import argparse, collections, csv, glob, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import decon as D
import protected_index_loader as PL
import build_pairs as BP

L19 = "amh tir fon dyu kbp nus mos kam bam umb kea kmb vmw kab seh wol ewe luo tum yor".split()      # 20 languages since pre-registration Amendment 1 (yor added); name kept for diff-ability
ACTIVE = list(L19)
EXCL = re.compile(r"nllb|translatepsy|translate-psy", re.I)
WEAK = (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-weak-language-parallel-2026-10-26")
R10 = (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-box-backup-r10-weak-language-data")


def lic_norm(raw):
    s = (raw or "").lower()
    if "cc0" in s: return "cc0"
    if "odc-by" in s: return "odc-by"
    if "-sa" in s or "share" in s: return "share_alike"
    if "non-commercial" in s or "-nc" in s or "tanzil" in s or "ted talks" in s: return "non_commercial"
    if "cc-by" in s or "cc by" in s: return "cc-by"
    if s.startswith("owner"): return "owner_private"
    return BP.lic_class(raw)


def write_stats(kept, out, sub="specialist"):
    os.makedirs(out + "/" + sub + "/stats", exist_ok=True)
    cols = ["english", "target", "iso", "source_dataset", "licence", "licence_raw", "kind", "source_family", "flipped", "tgt_tokens"]
    pq.write_table(pa.table({**{c: [r[c] for r in kept] for c in cols}, "decon_flag": [""] * len(kept)}), out + "/" + sub + "/pairs-0000.parquet", compression="zstd")
    S = out + "/" + sub + "/stats/"
    agg = collections.defaultdict(lambda: [0, 0])
    for r in kept:
        k = (r["iso"], r["licence"], r["kind"]); agg[k][0] += 1; agg[k][1] += r["tgt_tokens"]
    with open(S + "per_language_class_kind.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "licence_class", "data_kind", "pairs", "gemma_target_tokens"])
        for k, v in sorted(agg.items()): w.writerow([*k, *v])
    ag2 = collections.defaultdict(lambda: [0, 0])
    for r in kept:
        k = (r["iso"], r["source_family"], r["licence"], r["kind"]); ag2[k][0] += 1; ag2[k][1] += r["tgt_tokens"]
    with open(S + "per_language_source.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "source", "licence_class", "data_kind", "pairs", "gemma_target_tokens"])
        for k, v in sorted(ag2.items(), key=lambda kv: (kv[0][0], -kv[1][1])): w.writerow([*k, *v])
    summ = {}
    for r in kept:
        s = summ.setdefault(r["iso"], collections.Counter())
        s["pairs"] += 1; s["tokens"] += r["tgt_tokens"]
        s["human_pairs" if r["kind"] == "human" else "synthetic_pairs"] += 1
        s["human_tokens" if r["kind"] == "human" else "synthetic_tokens"] += r["tgt_tokens"]
        s["commercial_ok_tokens"] += r["tgt_tokens"] if r["licence"] in ("cc0", "cc-by", "odc-by", "apache", "mit", "mpl") else 0
    with open(S + "per_language_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "pairs", "gemma_target_tokens", "human_pairs", "human_tokens", "synthetic_pairs", "synthetic_tokens", "permissive_commercial_tokens", "below_2000_pairs", "human_below_2000_pairs"])
        for iso in sorted(summ, key=lambda k: (k not in L19, L19.index(k) if k in L19 else 0)):
            s = summ.get(iso, collections.Counter())
            w.writerow([iso, s["pairs"], s["tokens"], s["human_pairs"], s["human_tokens"], s["synthetic_pairs"], s["synthetic_tokens"], s["commercial_ok_tokens"], int(s["pairs"] < 2000), int(s["human_pairs"] < 2000)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-langs", nargs="*", default=None, help="restrict to these languages (e.g. yor) and write to specialist_<sub>"); ap.add_argument("--sub", default="specialist")
    ap.add_argument("--out", required=True); ap.add_argument("--index", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--restats", action="store_true", help="only relabel weak-repo licences of an existing specialist/pairs-0000.parquet and rewrite the per-language tables"); ap.add_argument("--tmp", default="/workspace/work/dl_spec")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download, HfApi
    from tokenizers import Tokenizer
    tok = os.environ["HF_TOKEN"]; api = HfApi(token=tok)
    ix, man = PL.load(a.index, a.manifest)
    held = set(man["devhalf"]["heldout_ids"])
    global ACTIVE
    if a.only_langs:
        ACTIVE = list(a.only_langs)
    if a.restats:
        pth = a.out + "/" + a.sub + "/pairs-0000.parquet"
        t = pq.read_table(pth).to_pydict(); n = len(t["iso"])
        for i in range(n):
            if t["source_dataset"][i].startswith("weak-repo:") and t["licence"][i] == "other":
                nl = lic_norm(t["licence_raw"][i]); t["licence"][i] = nl if nl not in ("other", "unspecified") else "cc-by"
        pq.write_table(pa.table(t), pth, compression="zstd")
        write_stats([{k: t[k][i] for k in t} for i in range(n)], a.out, a.sub)
        print("restats done"); return
    rows, excluded = [], collections.Counter()

    def add(prio, src, d):
        d["prio"] = prio; d["src_group"] = src; rows.append(d)

    # main
    for f in sorted(glob.glob(a.out + "/final/*.parquet")):
        t = pq.read_table(f, columns=["english", "target", "iso", "source_dataset", "licence", "flipped", "source_family"]).to_pydict()
        for i in range(len(t["iso"])):
            if t["iso"][i] in ACTIVE:
                fm = t["source_family"][i]
                rows.append(dict(english=t["english"][i], target=t["target"][i], iso=t["iso"][i], source_dataset=t["source_dataset"][i], licence=t["licence"][i], licence_raw=t["licence"][i],
                                 kind="synthetic_english_side" if fm == "finetranslations" else "human", source_family=fm, flipped=bool(t["flipped"][i]), prio=0, src_group="main"))
    # weak repo
    fl = [s.rfilename for s in api.dataset_info(WEAK).siblings if s.rfilename.endswith(".jsonl")]
    for f in fl:
        iso = os.path.basename(f)[:-6]; sec = f.split("/")[0] + "/" + f.split("/")[1] if f.startswith("train_ready") else f.split("/")[0]
        if iso not in ACTIVE:
            if iso not in ("knc", "knc_Arab", "efi", "ibb", "run", "yor"):
                pass
            excluded[(WEAK + ":" + f, "language not in the 19")] += 0
            continue
        p = hf_hub_download(WEAK, f, repo_type="dataset", token=tok, local_dir=a.tmp)
        pr = {"train_ready/permissive": 1, "train_ready/share_alike": 3, "synthetic_english_side": 6, "quarantine": -1}.get(sec, -1)
        for line in open(p, encoding="utf-8"):
            if not line.strip(): continue
            r = json.loads(line)
            sid = r.get("source_id", "")
            if pr == -1:
                excluded[(WEAK + ":quarantine/" + iso, "weak repo quarantine (licence doubt / NLLB-derived kea)")] += 1; continue
            if sid == "flores_plus_dev":
                excluded[(WEAK + ":flores_plus_dev", "replaced by data_devhalf (held-out half stays protected)")] += 1; continue
            if EXCL.search(sid + " " + r.get("licence", "") + " " + r.get("split_note", "")):
                excluded[(WEAK + ":" + sid, "NLLB / TranslatePsy mention")] += 1; continue
            add(pr, "weak", dict(english=r["english"], target=r["target"], iso=iso, source_dataset=f"weak-repo:{sid}", licence=lic_norm(r.get("licence")) if lic_norm(r.get("licence")) not in ("other", "unspecified") else {"permissive": "cc-by", "share_alike": "share_alike"}.get(r.get("licence_class"), "unspecified"),
                                 licence_raw=r.get("licence", ""), kind="synthetic_english_side" if (pr == 6 or r.get("english_machine_generated")) else "human",
                                 source_family=sid, flipped=False))
        os.remove(p)
    # r10 classes
    cls = {"commercial_ok": 2, "share_alike": 4, "NC_noncommercial": 7, "owner_code_switch": 8, "EXCLUDED_unknown_licence": -1}
    for c, pr in cls.items():
        p = hf_hub_download(R10, f"r10/sft_extra_v1/{c}/rows.parquet", repo_type="dataset", token=tok, local_dir=a.tmp)
        t = pq.read_table(p).to_pydict(); n = len(t["iso"])
        syn = t.get("synthetic_en") or t.get("synthetic_target") or [False] * n
        for i in range(n):
            iso = t["iso"][i]
            if iso not in ACTIVE: continue
            sd = t["source_dataset"][i]
            if pr == -1:
                excluded[("r10:EXCLUDED_unknown_licence:" + sd[:50], "unknown / unspecified licence class of the R10 audit")] += 1; continue
            if EXCL.search(sd + " " + t["licence"][i]):
                excluded[("r10:" + sd[:50], "NLLB / TranslatePsy mention")] += 1; continue
            if c == "owner_code_switch":
                kind, lic = "synthetic_gemma4", "owner_private"
            else:
                kind = "synthetic_english_side" if syn[i] else "human"
                lic = "non_commercial" if c == "NC_noncommercial" else lic_norm(t["licence"][i])
            add(pr, "r10", dict(english=t["english"][i], target=t["target"][i], iso=iso, source_dataset="r10:" + sd, licence=lic, licence_raw=t["licence"][i], kind=kind,
                                source_family=sd.split(":")[0].split(" ")[0], flipped=False))
        os.remove(p)
    # devhalf
    d = pq.read_table(a.out + "/data_devhalf/pairs-0000.parquet").to_pydict()
    for i in range(len(d["iso"])):
        if d["iso"][i] not in ACTIVE:
            continue
        add(5, "devhalf", dict(english=d["english"][i], target=d["target"][i], iso=d["iso"][i], source_dataset=d["source_dataset"][i], licence="share_alike", licence_raw="CC-BY-SA-4.0 (FLORES+)",
                               kind="human", source_family="flores-plus-dev-half", flipped=False))
    print("candidate rows", len(rows), flush=True)
    rows.sort(key=lambda r: r["prio"])
    # exact dedupe, then filter v.2
    seen, uniq, dups = set(), [], collections.Counter()
    for r in rows:
        k = BP.key_of(r["english"], r["iso"], r["target"])
        if k in seen:
            dups[r["src_group"] + ":" + r["source_family"]] += 1; continue
        seen.add(k); r["key"] = k; uniq.append(r)
    print("after dedupe", len(uniq), flush=True)
    kept, removed = [], collections.Counter()
    for i in range(0, len(uniq), 5000):
        ch = uniq[i:i + 5000]
        ra = ix.check_texts([r["english"] for r in ch]); rb = ix.check_texts([r["target"] for r in ch])
        for r, x, y in zip(ch, ra, rb):
            if any(x[:3]) or any(y[:3]):
                removed[(r["iso"], r["src_group"] + ":" + r["source_family"])] += 1
            else:
                kept.append(r)
    print("kept", len(kept), "removed by filter", sum(removed.values()), flush=True)
    # tokens
    tk = Tokenizer.from_file(a.tokenizer)
    for i in range(0, len(kept), 20000):
        enc = tk.encode_batch([r["target"] for r in kept[i:i + 20000]], add_special_tokens=False)
        for r, e in zip(kept[i:i + 20000], enc):
            r["tgt_tokens"] = len(e.ids)
    write_stats(kept, a.out, a.sub)
    S = a.out + "/" + a.sub + "/stats/"
    with open(S + "excluded.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source", "reason", "rows"])
        for (s, why), n in sorted(excluded.items(), key=lambda kv: -kv[1]):
            if n: w.writerow([s, why, n])
    with open(S + "removed_by_filter_and_duplicates.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["what", "iso", "source", "rows"])
        for (i, s), n in sorted(removed.items(), key=lambda kv: -kv[1]): w.writerow(["decon_v2", i, s, n])
        for s, n in dups.most_common(): w.writerow(["duplicate_of_earlier_source", "", s, n])
    json.dump({"rows": len(kept), "candidates": len(rows), "after_dedupe": len(uniq), "removed_by_filter": sum(removed.values()), "version": man["version"]}, open(S + "summary.json", "w"))
    print("done", len(kept))


if __name__ == "__main__":
    main()
