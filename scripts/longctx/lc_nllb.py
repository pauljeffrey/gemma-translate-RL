"""NLLB-200 (CTranslate2) on the LC1 benchmark.  usage: lc_nllb.py {600m|3b3} [sent] [whole]
sent  : every source sentence on its own, beam 4, max 256 tokens (the sentence-by-sentence wrapper; for 3b3 the core items are copied from the published pseudo-references)
whole : the whole document in ONE input, beam 4; the model's documented limits apply: max_position_embeddings 1024 (input truncated at 1024 tokens by CTranslate2, as in the library
        default), max_decoding_length 1024; the model card says it was trained on inputs <= 512 tokens and is not intended for document translation."""
import sys, os, time, json, collections
sys.path.insert(0, "/workspace/jobs/r14/pylibs")
from lc_common import *
import ctranslate2
from transformers import AutoTokenizer
from gemma_mt.eval.longdoc import join_sentences, split_long_sentence, norm_text

size = sys.argv[1]; modes = sys.argv[2:] or ["sent", "whole"]
HF = "/dev/shm/lc1_models/nllb600m-hf"
CT = {"600m": ("/dev/shm/lc1_models/nllb600m-ct2-fp16", "float16"), "3b3": ("/dev/shm/lc1_models/nllb3p3b-ct2-int8", "int8_float16")}[size]
name = {"600m": "nllb600m", "3b3": "nllb3b3"}[size]
tok = AutoTokenizer.from_pretrained(HF, src_lang="eng_Latn")
tr = ctranslate2.Translator(CT[0], device="cuda", compute_type=CT[1], intra_threads=4)
docs, items = load_docs(), load_items()
BEAM = 4


def enc(texts, max_len):
    return [tok.convert_ids_to_tokens(tok.encode(t, truncation=True, max_length=max_len)) for t in texts]


def run_batch(texts, code, max_in, max_dec, bs_tokens=4096, batch_type="tokens"):
    src = enc(texts, max_in)
    res = tr.translate_batch(src, target_prefix=[[code]] * len(src), beam_size=BEAM, max_batch_size=bs_tokens, batch_type=batch_type, max_decoding_length=max_dec,
                             max_input_length=max_in, return_scores=False)
    out = []
    for r in res:
        toks = r.hypotheses[0]
        toks = toks[1:] if toks and toks[0] == code else toks
        out.append((tok.decode(tok.convert_tokens_to_ids(toks), skip_special_tokens=True).strip(), len(toks)))
    return out


if "sent" in modes and not os.path.exists(f"{HYP}/{name}__sent.jsonl"):
    t0 = time.time()
    core_ref = {}
    if size == "3b3":                                   # the published pseudo-references ARE this system's sentence-by-sentence output on the core documents
        core_ref = {r["iid"]: r["ref_sents"] for r in rd(WORK + "/refs_nllb_core.jsonl")}
    need = collections.defaultdict(set)
    for it in items:
        if it["iid"] in core_ref:
            continue
        for s in docs[it["doc_id"]]["sents"]:
            for p in split_long_sentence(s, 700):
                need[it["iso"]].add(p)
    cache = {}
    n = 0
    for iso in sorted(need):
        us = sorted(need[iso], key=len)
        outs = run_batch(us, NLLB_CODE[iso], 256, 256)
        for u, (h, nt) in zip(us, outs):
            cache[(iso, u)] = (h, nt, nt < 255)
        n += len(us)
        print(f"[{name} sent] {iso} {len(us)} units, total {n}, {time.time() - t0:.0f}s", flush=True)
    rows = []
    for it in items:
        d = docs[it["doc_id"]]
        if it["iid"] in core_ref:
            so = core_ref[it["iid"]]; fin, nt = True, 0
        else:
            so, fin, nt = [], True, 0
            for s in d["sents"]:
                ps = [cache[(it["iso"], p)] for p in split_long_sentence(s, 700)]
                so.append(norm_text(" ".join(x[0] for x in ps))); fin &= all(x[2] for x in ps); nt += sum(x[1] for x in ps)
        rows.append({"iid": it["iid"], "hyp": join_sentences(so, d["para_starts"]), "hyp_sents": so, "finished": fin, "new_tokens": nt, "status": "ok", "extra": {"copied_from_pseudo_ref": it["iid"] in core_ref}})
    wr(f"{HYP}/{name}__sent.jsonl", rows)
    if size == "3b3":                                   # pseudo-reference file for the non-core items = this system's sentence outputs
        wr(WORK + "/refs_nllb_sent.jsonl", [{"iid": r["iid"], "ref_sents": r["hyp_sents"]} for r in rows if not r["extra"]["copied_from_pseudo_ref"]])
    print("sent done", len(rows), f"{time.time() - t0:.0f}s", flush=True)

if "whole" in modes and not os.path.exists(f"{HYP}/{name}__whole.jsonl"):
    # compute budget: whole-document input is run on 6 core/human languages (+ both extra AfriScience languages), without b4096 (the input is cut at 1,024 tokens anyway);
    # results are appended per language and the run is resumable.
    WL = set(os.environ.get("WHOLE_LANGS", "hau,yor,swh,amh,zul,ibo").split(","))
    t0 = time.time()
    part = f"{HYP}/{name}__whole.part.jsonl"
    done = {r["iid"]: r for r in rd(part)} if os.path.exists(part) else {}
    sel = [i for i in items if i["bin"] != "b4096" and (i["set"] == "sci" or i["iso"] in WL)]
    byiso = collections.defaultdict(list)
    for it in sorted(sel, key=lambda it: docs[it["doc_id"]]["src_tokens"]):
        if it["iid"] not in done:
            byiso[it["iso"]].append(it)
    for iso in sorted(byiso):
        its = byiso[iso]
        texts = [docs[i["doc_id"]]["text"] for i in its]
        n_src = [len(tok.encode(t)) for t in texts]                       # untruncated NLLB token count (incl. <s> lang </s>)
        outs = []
        for k in range(0, len(its), 8):
            outs += run_batch(texts[k:k + 8], NLLB_CODE[iso], 1024, 1024, bs_tokens=8, batch_type="examples")
        new = []
        for it, (h, nt), ns in zip(its, outs, n_src):
            new.append({"iid": it["iid"], "hyp": h, "finished": nt < 1022, "new_tokens": nt, "status": "ok", "extra": {"n_src_nllb_tokens": ns, "truncated_input": ns > 1024}})
        with open(part, "a", encoding="utf-8") as fh:
            for r in new:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{name} whole] {iso} {len(its)} docs {time.time() - t0:.0f}s", flush=True)
    wr(f"{HYP}/{name}__whole.jsonl", rd(part))
    print("whole done", f"{time.time() - t0:.0f}s", flush=True)
