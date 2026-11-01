"""Request sets for OUR models (Gemma-3 tokenizer + our chat template; token-id prompts). venv_gemma, CPU.
req_full.jsonl: one request per item (whole document); req_sent.jsonl: one request per unique (iso, sentence piece)."""
import sys, collections
from lc_common import *
from transformers import AutoTokenizer
from gemma_mt.format import build_prompt
from gemma_mt.eval.longdoc import split_long_sentence

tok = AutoTokenizer.from_pretrained(TOK_DIR)
def ntok(t): return len(tok(t, add_special_tokens=False)["input_ids"])
docs, items = load_docs(), load_items()

def req_row(rid, ids, src_tok, kind, extra=None):
    room = CTX - len(ids)
    want = int(3.5 * src_tok) if kind == "full" else min(int(3.5 * src_tok) + 16, 1024)
    r = {"rid": rid, "ids": ids, "max_new": min(want, room), "capped": want > room, "src_tokens": src_tok}
    if extra: r.update(extra)
    return r

full = []
for it in items:
    d = docs[it["doc_id"]]
    full.append(req_row(it["iid"], build_prompt(d["text"], it["iso"], tok), d["src_tokens"], "full"))
wr(WORK + "/req_full.jsonl", full)
uniq = {}
for it in items:
    for s in docs[it["doc_id"]]["sents"]:
        for p in split_long_sentence(s, 700):
            uniq.setdefault((it["iso"], p), None)
sent = []
for k, (iso, p) in enumerate(sorted(uniq)):
    sent.append(req_row(f"s{k}", build_prompt(p, iso, tok), ntok(p), "unit", {"iso": iso, "text": p}))
wr(WORK + "/req_sent.jsonl", sent)
print("full", len(full), "capped", sum(r["capped"] for r in full), "max prompt", max(len(r["ids"]) for r in full), "| sent units", len(sent),
      "src tokens full", sum(r["src_tokens"] for r in full), "sent", sum(r["src_tokens"] for r in sent))
