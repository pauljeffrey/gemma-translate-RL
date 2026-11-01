"""R14 stage 2, phase A+B: pick English windows of the language's RL prompt pools (hold-out filtered), split into sentences, translate with NLLB-3.3B.
usage: s2_translate.py CONFIG.json   CONFIG = {"langs": {iso: n_sentences, ...}, "config": "beam4nr4", "seed": 14}
Writes /workspace/jobs/r14/s2/plan_<iso>.jsonl (windows with their sentences) and trans_<iso>.jsonl (sentence, hyp, score). Resumable per language."""
import hashlib, json, os, random, re, sys, time, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq
from nllb_langmap import EXACT
from nllb_ct2 import NLLB
from gemma_mt.rl.segments import split_sentences
import s2_holdout

R = "/workspace/jobs/r14"; S2 = R + "/s2"; os.makedirs(S2, exist_ok=True)
cfg = json.load(open(sys.argv[1]))
nfc = lambda s: unicodedata.normalize("NFC", s)
POOLS = ["/workspace/sft_build/gemma/rl_prompts_v3/{iso}.parquet", "/workspace/jobs/r7/assets/rl_prompts_sci_v1/{iso}.parquet"]
_ok = re.compile(r"[^\W\d_]")

def good_sentence(s):
    w = s.split()
    if len(w) < 3 or len(s) > 600 or len(s) < 15:
        return False
    letters = sum(1 for c in s if c.isalpha())
    return letters / len(s) > 0.6 and not re.search(r"https?://|www\.|\{|\}|\\\\|<[a-z/]+>", s)

def select(iso, n_sent, ho, rng):
    plan_path = f"{S2}/plan_{iso}.jsonl"
    if os.path.exists(plan_path):
        return [json.loads(l) for l in open(plan_path)]
    rows = []
    for p in POOLS:
        p = p.format(iso=cfg.get("pool_alias", {}).get(iso, iso))
        if os.path.exists(p):
            t = pq.read_table(p, columns=["english", "source_dataset", "licence", "length_bin"]).to_pylist()
            rows += t
    order = list(range(len(rows))); rng.shuffle(order)
    out, total, stat = [], 0, {"windows_seen": 0, "hold_out_sentence": 0, "hold_out_ngram8": 0, "bad": 0}
    seen = set()
    for i in order:
        r = rows[i]; en = nfc(r["english"]); stat["windows_seen"] += 1
        h = ho.hit(en)
        if h:
            stat["hold_out_" + h] += 1; continue
        sents = [s for s in split_sentences(en)]
        if not sents or not all(good_sentence(s) for s in sents) or len(sents) > 90:
            stat["bad"] += 1; continue
        k = hashlib.sha1(en.encode()).hexdigest()
        if k in seen: continue
        seen.add(k)
        out.append({"iso": iso, "english": en, "sents": sents, "source_dataset": r["source_dataset"], "licence": r["licence"], "bin": r["length_bin"]})
        total += len(sents)
        if total >= n_sent: break
    with open(plan_path, "w") as f:
        for w in out: f.write(json.dumps(w, ensure_ascii=False) + "\n")
    json.dump(stat, open(f"{S2}/select_stats_{iso}.json", "w"))
    print(iso, "selected", len(out), "windows", total, "sentences; stats", stat, flush=True)
    return out

if __name__ == "__main__":
    ho = s2_holdout.build()
    m = NLLB(R + "/models/nllb3p3b-ct2-int8", R + "/models/nllb600m-hf", compute_type="int8_float16")
    for iso, n_sent in cfg["langs"].items():
        rng = random.Random(f"{cfg['seed']}-{iso}")
        wins = select(iso, n_sent, ho, rng)
        tp = f"{S2}/trans_{iso}.jsonl"
        done = set()
        if os.path.exists(tp):
            done = {json.loads(l)["english"] for l in open(tp)}
        todo = sorted({s for w in wins for s in w["sents"]} - done)
        if not todo:
            continue
        t0 = time.time(); CH = 2000
        with open(tp, "a") as f:
            for j in range(0, len(todo), CH):
                part = todo[j:j + CH]
                hyps, sc = m.translate(part, EXACT[iso], cfg["config"], return_scores=True)
                for e, h, s in zip(part, hyps, sc):
                    f.write(json.dumps({"english": e, "hyp": h, "score": s}, ensure_ascii=False) + "\n")
                f.flush()
        print(f"{iso} translated {len(todo)} sents in {time.time()-t0:.0f}s ({len(todo)/(time.time()-t0):.0f}/s)", flush=True)
    open(f"{S2}/TRANSLATE_DONE_{os.path.basename(sys.argv[1])}", "w").write("ok")
