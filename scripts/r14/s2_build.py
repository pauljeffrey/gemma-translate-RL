"""R14 stage 2, phase C: filter the NLLB-3.3B translations, assemble chunk-level SFT samples, trim to the per-language token budget, pack (R4 format, 8192).
usage: s2_build.py BUILD.json  BUILD = {"out": dir, "budgets": {iso: target_total_tokens}, "seq_len": 8192, "seed": 14, "dev_chrf": {iso: x}}"""
import collections, glob, hashlib, json, os, random, sys, time, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/jobs/r4/gemma_mt_sft/scripts")
import numpy as np
import build_sft_science_mix as R4
from gemma_mt import format as fmt
from gemma_mt.data.prepare_sft import tokenizer_fingerprint
from gemma_mt.languages import ALL_ISOS, LANGUAGES, iso_to_id
from gemma_mt.rl import anchors as A, lid as LID
from gemma_mt.rl.segments import split_sentences
from transformers import AutoTokenizer

R = "/workspace/jobs/r14"; S2 = R + "/s2"
cfg = json.load(open(sys.argv[1]))
OUT = cfg["out"]; os.makedirs(OUT, exist_ok=True)
SEQ = cfg.get("seq_len", 8192)
shp = json.load(open("/workspace/jobs/r14/code/shaping_flores_dev_v8.json"))
TEACHER, LICENCE_CLASS = "nllb-200-3.3B", "nc_teacher"
nfc = lambda s: unicodedata.normalize("NFC", s)
tok = AutoTokenizer.from_pretrained("/workspace/sft_build/runs/final_1433")
fmt.assert_template_matches_reference(tok)
fp = tokenizer_fingerprint(tok)
lidbin = glob.glob("/workspace/sft_build/hf/hub/models--cis-lmu--glotlid/snapshots/*/model_v3.bin")[0]
lid = LID.make_lid(lidbin)

def bounds(iso):
    e = shp.get(iso) or {}
    lr = e.get("len_ratio")
    if lr:
        p05, p50, p95 = lr["p05"], lr["p50"], lr["p95"]
    else:
        wr = e.get("word_ratio")
        if not wr: return None
        p05, p50, p95 = wr["p05"], wr["p50"], wr["p95"]
    return max(0.2, p50 - 2.2 * (p50 - p05)), p50 + 2.2 * (p95 - p50)

def lid_cfg(iso):
    e = (shp.get(iso) or {}).get("lid") or {}
    labels, minc, active = LID.accepted_labels(iso, e or None, LANGUAGES[iso].glotlid)
    return set(labels), max(minc, 40), active

def sent_filter(src, hyp, iso, bnd):
    """reason for dropping the sentence translation, or None"""
    if not hyp or not hyp.strip(): return "empty"
    sw, hw = A.words(src), A.words(hyp)
    if len(hw) == 0: return "empty"
    if len(sw) >= 5 and len(set(sw) & set(hw)) / max(1, len(set(sw) | set(hw))) > 0.6: return "copy"
    if len(hw) >= 5:
        rs = A.repeat_shares(hw, (3, 4)); ss = A.repeat_shares(sw, (3, 4)) if len(sw) >= 5 else {3: 0, 4: 0}
        if rs[3] > 0.25 and rs[3] > ss[3] + 0.15: return "repetition"
    if bnd and len(sw) >= 6:
        r = len(hyp) / max(1, len(src))
        if r < bnd[0]: return "ratio_low"
        if r > bnd[1]: return "ratio_high"
    return None

def main():
    t0 = time.time(); rows_out, items, stats = [], [], {}
    for iso, budget in cfg["budgets"].items():
        pp, tp = f"{S2}/plan_{iso}.jsonl", f"{S2}/trans_{iso}.jsonl"
        if not (os.path.exists(pp) and os.path.exists(tp)): print("skip", iso, "no translations"); continue
        wins = [json.loads(l) for l in open(pp)]
        tr = {}
        for l in open(tp):
            d = json.loads(l); tr[d["english"]] = d["hyp"]
        bnd = bounds(iso); labels, minc, lid_on = lid_cfg(iso)
        st = collections.Counter()
        # sentence level filters (+ per-sentence LID for longer sentences)
        bad = {}
        longs = [(s, h) for s, h in tr.items() if lid_on and len(h) >= minc]
        tops = dict(zip([s for s, _ in longs], lid([h for _, h in longs]))) if longs else {}
        for s, h in tr.items():
            r = sent_filter(s, h, iso, bnd)
            if r is None and s in tops and tops[s] not in labels: r = "lid"
            if r: bad[s] = r; st["sent_" + r] += 1
        # AfriCOMET-QE (reference-free) sentence filter where the language has validated anchors: keep qe >= t_fail (= P90 of QE on mismatched translations, the reward's own "fail" knot)
        qe_p = f"{S2}/qe_{iso}.jsonl"; blk = shp.get(iso) or {}
        if os.path.exists(qe_p) and blk.get("anchors") and blk.get("usable", True) and (blk.get("content") or {}).get("mode", "qe") == "qe":
            tf = float(blk["anchors"][0])
            for l in open(qe_p):
                d = json.loads(l)
                if d["english"] in tr and d["qe"] < tf and d["english"] not in bad: bad[d["english"]] = "qe_low"; st["sent_qe_low"] += 1
            st["qe_threshold"] = tf
        st["sentences"] = len(tr)
        rng = random.Random(f"{cfg['seed']}-{iso}-chunks")
        cand = []
        for wi, w in enumerate(wins):
            sents = w["sents"]
            mode = R4.MODES[rng.choices(range(4), weights=[R4.MODE_WEIGHTS[m] for m in R4.MODES])[0]]
            for a, b in R4.segment(len(sents), mode, rng):
                ch = sents[a:b]
                if any(s not in tr for s in ch): continue
                if any(s in bad for s in ch): st["chunks_dropped_sentence_filter"] += 1; continue
                en = " ".join(ch); tgt = " ".join(tr[s] for s in ch)
                if lid_on and len(tgt) >= minc and lid([tgt])[0] not in labels: st["chunks_dropped_lid"] += 1; continue
                ex = fmt.build_example(en, iso, nfc(tgt), tok)
                if ex.n_tokens > SEQ: st["chunks_too_long"] += 1; continue
                cand.append((w, mode, en, nfc(tgt), ex))
        rng.shuffle(cand)
        st["candidate_tokens"] = sum(c[4].n_tokens for c in cand); st["candidate_target_tokens"] = sum(c[4].target_len for c in cand)
        got = 0
        for w, mode, en, tgt, ex in cand:
            if got >= budget: break
            items.append((R4._ex(ex.input_ids, ex.prompt_len, ex.target_len), iso_to_id(iso)))
            got += ex.n_tokens
            rows_out.append({"iso": iso, "english": en, "translation": tgt, "mode": mode, "source_dataset": w["source_dataset"], "english_licence": w["licence"],
                             "licence_class": LICENCE_CLASS, "teacher": TEACHER, "teacher_decode": cfg.get("decode", "beam4_no_repeat_ngram4"),
                             "n_tokens": ex.n_tokens, "prompt_len": ex.prompt_len, "target_len": ex.target_len,
                             "ids_sha1": hashlib.sha1(np.asarray(ex.input_ids, dtype=np.uint32).tobytes()).hexdigest()})
        st.update({"chunks_candidates": len(cand), "kept_samples": sum(1 for r in rows_out if r["iso"] == iso), "kept_tokens": got,
                   "kept_target_tokens": sum(r["target_len"] for r in rows_out if r["iso"] == iso), "budget": budget})
        stats[iso] = dict(st); print(iso, dict(st), flush=True)
    meta_split = R4.write_split(items, OUT, SEQ, cfg["seed"], "train")
    import pandas as pd
    pd.DataFrame(rows_out).to_parquet(f"{OUT}/rows.parquet", index=False)
    isos = list(stats)
    man = {"teacher": TEACHER, "licence_class": LICENCE_CLASS, "created": time.strftime("%F %T"), "seq_len": SEQ, "total_tokens": sum(s["kept_tokens"] for s in stats.values()),
           "total_samples": len(items), "languages": []}
    isos = [i for i in isos if stats[i].get("kept_samples")]
    for iso in isos:
        s = stats[iso]
        man["languages"].append({"language": iso, "name": LANGUAGES[iso].name, "rows": s["kept_samples"], "tokens": s["kept_tokens"], "target_tokens": s["kept_target_tokens"],
                                 "teacher_quality_vs_human": (cfg.get("teacher_quality") or {}).get(iso), "filters": {k: v for k, v in s.items() if k.startswith(("sent_", "chunks_"))},
                                 "sentences_translated": s["sentences"], "candidate_tokens": s.get("candidate_tokens"),
                                 "pool_sentences_available": (json.load(open(f"{R}/s2_pool_sizes.json")).get(iso) or {}).get("sentences"),
                                 "yield_estimate_full_pool_tokens": (int(s.get("candidate_tokens", 0) / max(1, s["sentences"]) * (json.load(open(f"{R}/s2_pool_sizes.json")).get(iso) or {}).get("sentences", 0))
                                                                     if s.get("candidate_tokens") else None)})
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1, ensure_ascii=False)
    meta = {"format_version": 1, "dtype": "uint32", "seq_len": SEQ, "instruction_template": fmt.INSTRUCTION_TEMPLATE, "end_of_turn": fmt.END_OF_TURN, "ignore_index": fmt.IGNORE_INDEX,
            "tokenizer": {"name": "final_1433", "fingerprint": fp, "vocab_size": len(tok), "bos_id": tok.bos_token_id, "eos_id": tok.eos_token_id, "pad_id": tok.pad_token_id,
                          "end_of_turn_id": tok.convert_tokens_to_ids(fmt.END_OF_TURN)},
            "iso_table": list(ALL_ISOS), "directions": ["en2x"], "splits": {"train": meta_split}, "teacher": TEACHER, "licence_class": LICENCE_CLASS, "config": cfg}
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1, ensure_ascii=False)
    print("packed", len(items), "samples", man["total_tokens"], "tokens in", round(time.time() - t0), "s")

main()
