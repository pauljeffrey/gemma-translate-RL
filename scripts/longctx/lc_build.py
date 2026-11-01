"""Build the LC1 long-context benchmark definition on the box.  usage: lc_build.py
Writes WORK/docs.jsonl (English documents), items.jsonl (document x language), refs_human.jsonl / refs_nllb.jsonl (reference sentences per item), bench_summary.json."""
import sys, json, random, collections, unicodedata, statistics
from lc_common import *
from transformers import AutoTokenizer
from gemma_mt.eval.longdoc import join_sentences, norm_text

tok = AutoTokenizer.from_pretrained(TOK_DIR)
def ntok(t): return len(tok(t, add_special_tokens=False)["input_ids"])

N_CORE = {256: 20, 512: 20, 1024: 20, 2048: 20, 4096: 10}
N_HUM = {"sm": 30, "st512": 20, "st1024": 20, "st2048": 10}
docs, items, refs_h, refs_n = {}, [], {}, {}

# ------------------------------------------------------------------ A. core documents with NLLB-3.3B pseudo-references (12 human-reference languages)
core = {d["doc_id"]: d for d in rd(CORE + "/docs.jsonl")}
nl = {}
for iso in HUMAN_LANGS:
    nl[iso] = {r["doc_id"]: r for r in rd(f"{NLLB_REFS}/{iso}.refs.jsonl")}
ok = {did for did, d in core.items() if d["contam"]["flag"] == "clean" and all(did in nl[i] and nl[i][did].get("valid", True) and nl[i][did]["flagged_share"] <= 0.05
                                                                         and len(nl[i][did]["ref_sents"]) == len(d["sents"]) for i in HUMAN_LANGS)}
print("core docs clean+valid for all 12 languages:", collections.Counter(core[d]["bin"] for d in ok))
rng = random.Random(20261023)
for b in CORE_BINS + CORE_BINS_EXTRA:
    cand = sorted(d for d in ok if core[d]["bin"] == b)
    rng.shuffle(cand)
    for did in cand[:N_CORE[b]]:
        d = core[did]
        docs[did] = {"doc_id": did, "set": "core", "bin": f"b{b}", "text": d["text"], "sents": d["sents"], "para_starts": d["para_starts"], "domain": d.get("domain", ""),
                     "source": d.get("source", ""), "licence": d.get("licence", ""), "src_tokens": ntok(d["text"]), "contam": d["contam"]["flag"]}
        for iso in HUMAN_LANGS:
            iid = f"core|{did}|{iso}"
            items.append({"iid": iid, "doc_id": did, "iso": iso, "set": "core", "bin": f"b{b}"})
            refs_n[iid] = {"iid": iid, "ref_sents": nl[iso][did]["ref_sents"]}

# ------------------------------------------------------------------ B. SmolDoc human-reference documents: clean short documents + stitched (own stitching of CLEAN documents)
sm = {d["doc_id"]: d for d in rd(CORE + "/human_smoldoc/docs.jsonl")}
smref = {}
for iso in HUMAN_LANGS:
    smref[iso] = {r["doc_id"]: r["ref_sents"] for r in rd(f"{CORE}/human_smoldoc/refs/{iso}.jsonl")
                  if (not r["contaminated"]) and len(r["ref_sents"]) == len(sm[r["doc_id"]]["sents"])}
stats = {}
for gname, grp in (("G1", G1), ("G2", G2)):
    common = sorted(set.intersection(*[set(smref[i]) for i in grp]))
    lens = {d: ntok(sm[d]["text"]) for d in common}
    stats[gname] = {"languages": grp, "clean_common_sm_docs": len(common), "tokens": sum(lens.values())}
    rng = random.Random(f"hum-{gname}")
    # short bin
    cand = common[:]; rng.shuffle(cand)
    for did in cand[:N_HUM["sm"]]:
        s = sm[did]
        docs[f"{gname}:{did}"] = {"doc_id": f"{gname}:{did}", "set": "human", "bin": "sm", "group": gname, "parts": [did], "text": s["text"], "sents": s["sents"], "para_starts": s["para_starts"],
                                  "domain": s.get("domain", ""), "source": "smoldoc", "licence": "CC-BY-4.0", "src_tokens": lens[did]}
    for tgt, name in ((512, "st512"), (1024, "st1024"), (2048, "st2048")):
        pool = common[:]; rng.shuffle(pool)
        used = set(); made = 0
        while made < N_HUM[name]:
            parts, tot = [], 0
            for d in pool:
                if d in used:
                    continue
                if parts and tot + lens[d] > tgt * 1.1:
                    continue
                parts.append(d); tot += lens[d]
                if tot >= tgt * 0.93:
                    break
            if tot < tgt * 0.8:
                break
            used.update(parts)
            sents, ps = [], []
            for d in parts:
                ps += [len(sents) + p for p in sm[d]["para_starts"]]
                sents += sm[d]["sents"]
            text = join_sentences(sents, ps)
            did = f"{gname}:{name}:" + "+".join(p.split(":")[-1][:12] for p in parts[:2]) + f"+{len(parts)}"
            docs[did] = {"doc_id": did, "set": "human", "bin": name, "group": gname, "parts": parts, "text": text, "sents": sents, "para_starts": ps, "domain": "stitched_clean_smoldoc",
                         "source": "smoldoc", "licence": "CC-BY-4.0", "src_tokens": ntok(text)}
            made += 1
        stats[gname][name] = made
for did, d in list(docs.items()):
    if d["set"] != "human":
        continue
    grp = G1 if d["group"] == "G1" else G2
    for iso in grp:
        rs = []
        for p in d["parts"]:
            rs += smref[iso][p]
        iid = f"hum|{did}|{iso}"
        items.append({"iid": iid, "doc_id": did, "iso": iso, "set": "human", "bin": d["bin"]})
        refs_h[iid] = {"iid": iid, "ref_sents": rs}

# ------------------------------------------------------------------ C. AfriScience-MT dev / test papers (human references, 6 languages)
import pandas as pd, glob
for split in ("dev", "test"):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{AFS}/{split}-*.parquet"))], ignore_index=True)
    nf = unicodedata.normalize
    for iso in SCI_LANGS:
        d = df[df.lang_pair == f"eng-{iso}"].sort_values(["paper_id", "sentence_id"])
        for pid, g in d.groupby("paper_id", sort=True):
            did = f"sci_{split}:{pid}"
            src = [nf("NFC", s) for s in g.source]
            if did not in docs:
                docs[did] = {"doc_id": did, "set": "sci", "bin": f"sci_{split}", "split": split, "text": " ".join(src), "sents": src, "para_starts": [0], "domain": str(g.domain.iloc[0]),
                             "source": "afriscience_mt", "licence": "Apache-2.0", "src_tokens": ntok(" ".join(src))}
            else:
                assert docs[did]["sents"] == src, "English side differs between languages"
            iid = f"sci_{split}|{did}|{iso}"
            items.append({"iid": iid, "doc_id": did, "iso": iso, "set": "sci", "bin": f"sci_{split}"})
            refs_h[iid] = {"iid": iid, "ref_sents": [nf("NFC", s) for s in g.target]}

wr(WORK + "/docs.jsonl", list(docs.values()))
wr(WORK + "/items.jsonl", items)
wr(WORK + "/refs_human.jsonl", list(refs_h.values()))
wr(WORK + "/refs_nllb_core.jsonl", list(refs_n.values()))
summ = {"groups": stats, "docs_per_bin": collections.Counter(d["bin"] for d in docs.values()), "items_per_bin": collections.Counter(i["bin"] for i in items)}
tokb = collections.defaultdict(list)
for d in docs.values():
    tokb[d["bin"]].append(d["src_tokens"])
summ["gemma_tokens_per_bin"] = {k: {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v), "sum": sum(v)} for k, v in tokb.items()}
json.dump(summ, open(WORK + "/bench_summary.json", "w"), indent=1, default=dict)
print(json.dumps(summ, indent=1, default=dict))
