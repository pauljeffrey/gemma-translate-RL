"""Turn token ids of OUR models into per-item translations. usage: lc_assemble_ours.py NAME   -> hyp/NAME__whole.jsonl, NAME__sent.jsonl (greedy units), NAME__sentfix.jsonl (units repaired by best-of-4)."""
import sys, os, json, gzip, collections
from lc_common import *
from transformers import AutoTokenizer
from gemma_mt.rl.text import decode_completion, translation_from
from gemma_mt.eval.longdoc import join_sentences, norm_text, split_long_sentence
sys.path.insert(0, ROOT)
from lc_consensus import consensus_pick
STOP = [106, 1]
tok = AutoTokenizer.from_pretrained(TOK_DIR)
docs, items = load_docs(), {i["iid"]: i for i in load_items()}


def dec(iso, o):
    d = decode_completion(tok, o["ids"], STOP)
    t = translation_from(d, iso)
    return {"text": t.text, "finished": bool(d.finished), "new_tokens": len(o["ids"]), "status": t.status}


def read(name, tag):
    with gzip.open(f"{OUT}/{name}/{tag}.jsonl.gz", "rt", encoding="utf-8") as f:
        for l in f:
            yield json.loads(l)


def main(name):
    os.makedirs(HYP, exist_ok=True)
    rows = []
    for r in read(name, "full"):
        x = dec(items[r["rid"]]["iso"], r["outs"][0])
        rows.append({"iid": r["rid"], "hyp": x["text"], "finished": x["finished"], "new_tokens": x["new_tokens"], "status": x["status"]})
    wr(f"{HYP}/{name}__whole.jsonl", rows); print("whole", len(rows))
    reqs = {(q["iso"], q["text"]): q["rid"] for q in rd(WORK + "/req_sent.jsonl")}
    unit = {r["rid"]: r["outs"][0] for r in read(name, "sent")}
    fix = {r["rid"]: r["outs"] for r in read(name, "fix")} if os.path.exists(f"{OUT}/{name}/fix.jsonl.gz") else {}
    stat = collections.Counter()
    cache = {}
    def tr(iso, piece, repaired):
        k = (iso, piece, repaired)
        if k not in cache:
            rid = reqs[(iso, piece)]
            d = dec(iso, unit[rid])
            if repaired and rid in fix and not d["finished"]:
                cs = [dec(iso, o) for o in fix[rid]]
                fin = [c for c in cs if c["finished"]]
                if len(fin) >= 2:
                    b, _ = consensus_pick([c["text"] for c in fin]); d = fin[b]; stat["fixed"] += 1
                elif len(fin) == 1:
                    d = fin[0]; stat["fixed"] += 1
                else:
                    stat["still_bad"] += 1
            cache[k] = d
        return cache[k]
    for repaired, suffix in ((False, "sent"), (True, "sentfix")):
        out = []
        for iid, it in items.items():
            doc = docs[it["doc_id"]]
            so, fin, nt, bad = [], True, 0, 0
            for s in doc["sents"]:
                ps = [tr(it["iso"], p, repaired) for p in split_long_sentence(s, 700)]
                so.append(norm_text(" ".join(x["text"] for x in ps)))
                fin &= all(x["finished"] for x in ps); nt += sum(x["new_tokens"] for x in ps); bad += sum(x["status"] != "ok" for x in ps)
            out.append({"iid": iid, "hyp": join_sentences(so, doc["para_starts"]), "hyp_sents": so, "finished": fin, "new_tokens": nt, "status": "ok" if bad == 0 else f"{bad}_bad_units"})
        wr(f"{HYP}/{name}__{suffix}.jsonl", out); print(suffix, len(out), dict(stat))
    json.dump({"units": len(unit), "unfinished_units": sum(1 for r in unit.values() if not (r['ids'] and r['ids'][-1] in STOP)), "repair": dict(stat)}, open(f"{HYP}/{name}__repair_stats.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
