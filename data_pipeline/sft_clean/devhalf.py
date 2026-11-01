#!/usr/bin/env python3
"""STEP B (lead decision 2026-10-28): FLORES+ dev train half for the weak languages. Version protected-2026-10-28.2.

Rule (fixed, applied once to the 997 FLORES+ dev sentence ids for ALL languages):
    held-out  if int(sha256(str(id)).hexdigest(), 16) is odd, or id is one of the 100 ids of reports/data/weak_selection_2026-10-26/sample_ids.json (forced held-out)
    train half otherwise
Held-out half: stays fully protected in every language, both sides (reserved for reward calibration T2 and selection S). FLORES+ DEVTEST: unchanged, fully protected.
Train half: exempt from the filter ONLY as text of the 19 listed languages (target) and as English; the train-half targets of every other language stay protected.
Exemption = an extra hash index (protected_exemption_devhalf.npz, the train-half sentences and their 8-grams) attached to the v.1 index: a hit on an exempt sentence or an exempt
8-gram is not a hit (decon.Index.exempt). Output: data_devhalf/pairs-0000.parquet with the exact ids; licence class share_alike (FLORES+ is CC-BY-SA-4.0).
    python devhalf.py --index protected/protected_index.npz --manifest protected/protected_manifest.json --sample-ids sample_ids.json --out out
"""
import argparse, hashlib, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decon as D
import pyarrow as pa, pyarrow.parquet as pq

LANGS = {"amh": "amh_Ethi", "tir": "tir_Ethi", "fon": "fon_Latn", "dyu": "dyu_Latn", "kbp": "kbp_Latn", "nus": "nus_Latn", "mos": "mos_Latn", "kam": "kam_Latn", "bam": "bam_Latn",
         "umb": "umb_Latn", "kea": "kea_Latn", "kmb": "kmb_Latn", "vmw": "vmw_Latn", "kab": "kab_Latn", "seh": "seh_Latn_MZ", "wol": "wol_Latn", "ewe": "ewe_Latn", "luo": "luo_Latn",
         "tum": "tum_Latn", "yor": "yor_Latn"}      # yor added 2026-10-28 (pre-registration Amendment 1): 20 languages, protected-2026-10-28.3
RULE = "held-out if int(sha256(str(id).encode()).hexdigest(), 16) % 2 == 1 or id in forced; train half otherwise"
URL = re.compile(r"https?://|www\.", re.I)


def ok(en, tg):
    if not (3 <= len(en) <= 1500 and 3 <= len(tg) <= 2500):
        return False
    r = len(tg) / max(1, len(en))
    if not (0.25 <= r <= 4.5) or en.casefold() == tg.casefold() or URL.search(en) or URL.search(tg):
        return False
    return sum(c.isalpha() for c in en) / len(en) >= 0.4 and re.search(r"[A-Za-z]{2}", en) and sum(c.isalpha() for c in tg) / len(tg) >= 0.35


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--sample-ids", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default="/workspace/work/dl_devhalf")
    ap.add_argument("--version", default="protected-2026-10-28.3"); ap.add_argument("--exemption-name", default="protected_exemption_devhalf_v3.npz"); ap.add_argument("--manifest-out", default=None)
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ["HF_TOKEN"]
    commit = HfApi(token=tok).dataset_info("openlanguagedata/flores_plus").sha

    def load(code):
        p = hf_hub_download("openlanguagedata/flores_plus", f"dev/{code}.jsonl", repo_type="dataset", token=tok, local_dir=a.tmp)
        d = {}
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if (r.get("text") or "").strip():
                    d[int(r["id"])] = r["text"].strip()
        os.remove(p)
        return d
    eng = load("eng_Latn")
    forced = sorted(set(json.load(open(a.sample_ids))["flores_dev_ids"]))
    ids = sorted(eng)
    held = sorted(i for i in ids if i in forced or int(hashlib.sha256(str(i).encode()).hexdigest(), 16) % 2 == 1)
    train = sorted(i for i in ids if i not in set(held))
    print(f"dev ids {len(ids)}: held-out {len(held)} (forced {len(forced)}), train half {len(train)}", flush=True)
    tset = set(train)
    texts = {c: load(c) for c in LANGS.values()}
    # exemption index: train-half English + train-half targets of the listed languages
    E = D.Builder()
    for i in train:
        E.add("devhalf_train_exempt", eng[i])
    for iso, c in LANGS.items():
        for i in train:
            if i in texts[c]:
                E.add("devhalf_train_exempt", texts[c][i])
    ex = E.finish()
    # the train half must not make ANY protected text unprotected: remove from the exemption every sentence hash and 8-gram that also occurs in FLORES+ devtest or the held-out dev half
    # (FLORES sentences of one article share phrases, and some sentences are repeated across ids)
    import numpy as np
    Hb = D.Builder()
    fl_files = [x.rfilename for x in HfApi(token=tok).dataset_info("openlanguagedata/flores_plus").siblings if x.rfilename.startswith(("dev/", "devtest/")) and x.rfilename.endswith(".jsonl")]
    for ff in fl_files:
        pth = hf_hub_download("openlanguagedata/flores_plus", ff, repo_type="dataset", token=tok, local_dir=a.tmp)
        for line in open(pth, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if ff.startswith("devtest/") or int(r["id"]) in held:
                Hb.add("heldout_or_devtest", r["text"])
        os.remove(pth)
    hx = Hb.finish()
    ms = ~np.isin(ex.s_keys, hx.s_keys); mg = ~np.isin(ex.g_keys, hx.g_keys)
    removed_s, removed_g = int((~ms).sum()), int((~mg).sum())
    ex.s_keys, ex.s_bits, ex.g_keys, ex.g_bits = ex.s_keys[ms], ex.s_bits[ms], ex.g_keys[mg], ex.g_bits[mg]
    print(f"exemption trimmed: {removed_s} sentence hashes and {removed_g} 8-grams also occur in devtest / held-out dev and are not exempt", flush=True)
    exp = os.path.join(os.path.dirname(a.index), a.exemption_name)
    ex.save(exp)
    ix = D.Index.load(a.index)
    ix.exempt = ex
    # rows
    rows, stats = [], {}
    import collections
    hit_sets = collections.Counter()
    for iso, c in LANGS.items():
        cand = [(i, re.sub(r"\s+", " ", eng[i]).strip(), re.sub(r"\s+", " ", texts[c][i]).strip()) for i in train if i in texts[c]]
        okc = [x for x in cand if ok(x[1], x[2])]
        ra = ix.check_texts([x[1] for x in okc]); rb = ix.check_texts([x[2] for x in okc])
        n_keep = 0
        for x, u, v in zip(okc, ra, rb):
            if any(u[:3]) or any(v[:3]):
                for m in u[:3] + v[:3]:
                    for nm in ix.names(m):
                        hit_sets[nm] += 1
                continue
            rows.append({"english": x[1], "target": x[2], "iso": iso, "source_dataset": f"openlanguagedata/flores_plus:dev:{c}", "licence": "share_alike", "flipped": False,
                         "decon_flag": "", "source_family": "flores-plus-dev-half", "flores_id": x[0], "split": "dev_train_half"})
            n_keep += 1
        stats[iso] = {"code": c, "train_half_ids_with_text": len(cand), "pass_ok": len(okc), "kept_after_filter": n_keep}
    os.makedirs(f"{a.out}/data_devhalf", exist_ok=True)
    cols = ["english", "target", "iso", "source_dataset", "licence", "flipped", "decon_flag", "source_family", "flores_id", "split"]
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in cols}), f"{a.out}/data_devhalf/pairs-0000.parquet", compression="zstd")
    man = json.load(open(a.manifest))
    man["version"] = a.version
    man["devhalf"] = {"rule": RULE, "forced_heldout_ids_source": "reports/data/weak_selection_2026-10-26/sample_ids.json (flores_dev_ids, 100 ids)",
                      "flores_plus_commit": commit, "listed_languages_train_half_allowed": LANGS,
                      "heldout_ids": held, "train_ids": train, "n_heldout": len(held), "n_train": len(train), "forced_ids": forced,
                      "heldout_ids_sha256": sha(held), "train_ids_sha256": sha(train), "forced_ids_sha256": sha(forced),
                      "exemption_file": os.path.basename(exp), "exemption_file_sha256": hashlib.sha256(open(exp, "rb").read()).hexdigest(),
                      "exemption_digest": ex.digest(), "exemption_trimmed_sentence_hashes": removed_s, "exemption_trimmed_8grams": removed_g,
                      "semantics": "a hit on an exempt sentence or an exempt 8-gram is not a hit; held-out half, devtest and every other protected text are unchanged; the English side of the train half is exempt in all sets (xsim++, IbomMT dev, SIB-200 ... contain the same English)",
                      "per_language": stats, "rows_written": len(rows), "rows_blocked_by_other_protected_sets": dict(hit_sets)}
    json.dump(man, open(a.manifest_out or a.manifest.replace(".json", "_new.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"rows": len(rows), "stats": stats, "blocked": dict(hit_sets)}, indent=1))
    json.dump({"rule": RULE, "heldout_ids": held, "train_ids": train, "forced_ids": forced, "heldout_ids_sha256": sha(held), "train_ids_sha256": sha(train)},
              open(f"{a.out}/stats/devhalf_split.json", "w"))


if __name__ == "__main__":
    main()
