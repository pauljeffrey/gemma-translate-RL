"""R15 stage 4b (CPU): filter Gemma synthetic en->X pairs, pack in R4's SFT window format. usage: filter_pack.py <out_dir> <iso> [<iso>...]"""
import json, os, sys, re, glob, hashlib, collections, time, unicodedata
import numpy as np, pandas as pd
sys.path.insert(0, "/workspace/jobs/r10"); sys.path.insert(0, "/workspace/jobs/b1/gemma_mt"); sys.path.insert(0, "/workspace/jobs/r4/gemma_mt_sft/scripts"); sys.path.insert(0, "/workspace/jobs/r4/gemma_mt_sft")
import build_extra as B
import build_sft_science_mix as R4
from sacrebleu.metrics import CHRF
from transformers import AutoTokenizer
from gemma_mt import format as fmt
from gemma_mt.data.prepare_sft import tokenizer_fingerprint
from gemma_mt.languages import ALL_ISOS, LANGUAGES, iso_to_id
from gemma_mt.rl.lid import GlotLIDClassifier
OUTD = sys.argv[1]; ISOS = sys.argv[2:]
S = "/workspace/jobs/r15/synth"; TAU = float(os.environ.get("TAU", "60")); VALFRAC = 0.03; SEQ = 8192
W = re.compile(r"\w+", re.U); words = lambda s: W.findall(unicodedata.normalize("NFC", s).lower())
nfc = lambda s: unicodedata.normalize("NFC", s)
chrf = CHRF(word_order=2)
lid = GlotLIDClassifier(path=glob.glob("/workspace/sft_build/hf/hub/models--cis-lmu--glotlid/snapshots/*/model_v3.bin")[0])
tok = AutoTokenizer.from_pretrained("/workspace/sft_build/runs/final_1433"); fmt.assert_template_matches_reference(tok)
fp = tokenizer_fingerprint(tok); base_meta = json.load(open("/workspace/jobs/r4/mix_v1/meta.json")); assert base_meta["tokenizer"]["fingerprint"] == fp and base_meta["iso_table"] == list(ALL_ISOS)
META = re.compile(r"(^|\n)\s*(note|translation|here is|here's|english)\s*[:\-]|\(note|\[note|\*\*|```|<[a-z/]+>", re.I)
def grams(w, n): return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
def distinct(texts, n):
    g = []
    for t in texts: g += grams(words(t), n)
    return len(set(g)) / max(1, len(g))
all_rows, all_wins, items_tr, items_va = [], [], [], []
manifest = {}
rng = np.random.default_rng(20261022)
for iso in ISOS:
    acc = set(LANGUAGES[iso].glotlid)
    pool = [json.loads(x) for x in open(f"/workspace/jobs/r10/pool/{iso}.jsonl")]
    ratios = np.array([len(words(p["target"])) / max(1, len(words(p["english"]))) for p in pool]); lo, hi = np.percentile(ratios, 1) / 1.25, np.percentile(ratios, 99) * 1.25
    pool_t = [p["target"] for p in pool]; pool_g4 = set(g for p in pool_t for g in grams(words(p), 4))
    rows = []
    for f in sorted(glob.glob(f"{S}/out/{iso}/shard_*.jsonl")): rows += [json.loads(x) for x in open(f)]
    st = collections.Counter(); st["generated"] = len(rows)
    cand = []
    for r in rows:
        h, s, en = nfc(r["hyp_g"]).strip(), nfc(r["hyp_s"]).strip(), r["english"]
        if not h or "\n" in h: st["empty_or_multiline"] += 1; continue
        if "length" in r["fin_g"].lower(): st["truncated"] += 1; continue
        if META.search(h): st["meta_text"] += 1; continue
        hw, sw = words(h), words(en)
        if len(hw) < 3 and len(sw) >= 6: st["too_short"] += 1; continue
        if (np.mean([w in set(sw) for w in hw]) >= 0.8 and len(hw) >= 3) or h.lower() == en.lower(): st["copy_en"] += 1; continue
        ratio = len(hw) / max(1, len(sw))
        if not (lo <= ratio <= hi): st["length_ratio"] += 1; continue
        g3 = grams(hw, 3)
        if g3 and 1 - len(set(g3)) / len(g3) > 0.25: st["repetition"] += 1; continue
        dig = set(re.findall(r"\d+", en))
        if dig and sum(d in h for d in dig) / len(dig) < 0.8: st["numbers_lost"] += 1; continue
        sh = [words(t) for t in r["shot_targets"]]; hg = set(grams(hw, 4))
        if any(hw == x for x in sh) or (hg and len(hg & set(g for x in sh for g in grams(x, 4))) / len(hg) > 0.5): st["copies_shots"] += 1; continue
        cand.append((r, h, s))
    labs = lid.top([c[1] for c in cand], 1) if cand else []
    keep1 = []
    for c, l in zip(cand, labs):
        if (l[0][0] if l else "") not in acc: st["lid_wrong"] += 1; continue
        keep1.append(c)
    keep2 = []
    for r, h, s in keep1:
        cs = chrf.sentence_score(h, [s]).score if s else 0.0
        if cs < TAU: st["inconsistent_samples"] += 1; continue
        keep2.append((r, h, cs))
    seen = set(); final = []
    for r, h, cs in keep2:
        k = " ".join(words(h))
        if k in seen: st["dup_target"] += 1; continue
        seen.add(k); final.append((r, h, cs))
    st["kept"] = len(final)
    tgts = [h for _, h, _ in final]
    div = {n: dict(synth=distinct(tgts[:5000], n), pool=distinct(pool_t[:5000], n)) for n in (1, 2, 3)}
    novelty = float(np.mean([g not in pool_g4 for h in tgts[:3000] for g in grams(words(h), 4)])) if tgts else None
    dup_rate_sources = 1 - len({" ".join(words(r["english"])) for r, _, _ in final}) / max(1, len(final))
    # windows per source_dataset (consecutive pairs), R10-style bins
    by = collections.defaultdict(list)
    for r, h, cs in final: by[r["source_dataset"]].append(dict(english=nfc(r["english"]), target=h, doc="", source_dataset="gemma_synthetic", src=r["source_dataset"], src_licence=r["licence"], domain=r["domain"], cons=cs, sid=r["sid"]))
    nt = nv = ttok = 0
    for srcname, rr in sorted(by.items()):
        wins, ntok = B.windows_for(rr, tok, rng)
        for w in wins:
            en = " ".join(rr[i]["english"] for i in w); tg = " ".join(rr[i]["target"] for i in w)
            wid = hashlib.md5(("gs" + iso + en).encode()).hexdigest()[:16]
            split = "val" if int(wid, 16) % 10000 < VALFRAC * 10000 else "train"
            ex = fmt.build_example(en, iso, tg, tok)
            if ex.n_tokens > SEQ: continue
            from types import SimpleNamespace
            ns = SimpleNamespace(input_ids=np.asarray(ex.input_ids, dtype=np.uint32), prompt_len=int(ex.prompt_len), target_len=int(ex.target_len))
            (items_va if split == "val" else items_tr).append((ns, iso_to_id(iso)))
            all_wins.append(dict(window_id=wid, iso=iso, english=en, target=tg, n_sent=len(w), en_tokens=sum(ntok[i] for i in w), total_tokens=int(ex.n_tokens), target_tokens=int(ex.target_len),
                                 source_dataset="gemma_synthetic", src_source_dataset=srcname, teacher="google/gemma-4-31B-it-qat-w4a16-ct (retrieval 8-shot, greedy)", licence="apache-2.0 (Gemma 4 outputs); source text licence in src_licence",
                                 licence_class="gemma_synthetic", synthetic_target=True, split=split))
            for i in w:
                x = rr[i]; all_rows.append(dict(iso=iso, english=x["english"], target=x["target"], source_dataset="gemma_synthetic", src_source_dataset=srcname, src_licence=x["src_licence"], domain=x["domain"],
                                                teacher="google/gemma-4-31B-it-qat-w4a16-ct", licence_class="gemma_synthetic", synthetic_target=True, consistency_chrf=x["cons"], sid=x["sid"], window_id=wid, in_split=split))
            if split == "val": nv += 1
            else: nt += 1
            ttok += int(ex.target_len)
    manifest[iso] = dict(stages=dict(st), yield_=len(final) / max(1, len(rows)), windows_train=nt, windows_val=nv, target_tokens=ttok, distinct_ngram=div, novelty_4gram_vs_pool=novelty, duplicate_source_rate=dup_rate_sources,
                         tau=TAU, length_ratio_bounds=[float(lo), float(hi)], by_source={k: len(v) for k, v in by.items()})
    print(iso, json.dumps(manifest[iso], ensure_ascii=False)[:1500], flush=True)
os.makedirs(OUTD, exist_ok=True)
meta_splits = {"train": R4.write_split(items_tr, OUTD, SEQ, 20261022, "train")}
if items_va: meta_splits["val"] = R4.write_split(items_va, OUTD, SEQ, 20261022, "val")
meta = dict(base_meta); meta["splits"] = meta_splits; meta["seq_len"] = SEQ; meta.pop("config", None)
meta["mixture"] = dict(group="gemma_synth_v1", built_by="R15 filter_pack.py", isos=ISOS, tau=TAU, teacher="google/gemma-4-31B-it-qat-w4a16-ct")
json.dump(meta, open(f"{OUTD}/meta.json", "w"), indent=1, ensure_ascii=False)
pd.DataFrame(all_rows).to_parquet(f"{OUTD}/rows.parquet", index=False); pd.DataFrame(all_wins).to_parquet(f"{OUTD}/windows.parquet", index=False)
json.dump(manifest, open(f"{OUTD}/manifest.json", "w"), indent=1, ensure_ascii=False)
print("PACKED", {k: (v["windows_train"], v["windows_val"], v["target_tokens"]) for k, v in manifest.items()})
