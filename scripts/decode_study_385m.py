#!/usr/bin/env python3
"""Greedy vs beam-4 vs MBR-16 on a FLORES+ devtest subset (post-SFT evaluation of the 385M model), one vLLM engine, one process.

    GEMMA_MT_TAG_MODE=single_token python scripts/decode_study_385m.py --model DIR --flores DIR/devtest --out out.json [--langs yor hau ...] [--n 200] [--gpu-util 0.08]

Systems (all on the same prompts, `gemma_mt.format.build_prompt`):
  greedy   temperature 0
  beam4    vLLM beam search, width 4, length penalty 1.0
  mbr16    16 samples (temperature 0.7, top_p 0.95) + the greedy output as 17th candidate; the pick is the candidate with the highest mean chrF++ against the other candidates
Scores: corpus chrF++ per language (sacrebleu word_order 2, NFC) and the macro over languages. Hypotheses are cut at the first end token and decoded without special tokens.
"""
import argparse, json, os, sys, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
LANGS = "yor hau ibo swh amh zul xho kin som wol lug sna twi ewe lin tir orm nso tsn afr".split()
EVAL_ONLY = {"nso": "nso_Latn"}


def flores_code(iso, files):
    from gemma_mt import languages as L
    c = getattr(L.LANGUAGES[iso], "flores", None)
    if c and f"{c}.jsonl" in files:
        return c
    for f in files:
        if f.startswith(iso + "_"):
            return f[:-6]
    raise KeyError(iso)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--flores", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="*", default=LANGS); ap.add_argument("--n", type=int, default=200); ap.add_argument("--gpu-util", type=float, default=0.08)
    ap.add_argument("--max-new", type=int, default=256); ap.add_argument("--k", type=int, default=16); ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--greedy-only", action="store_true"); ap.add_argument("--no-mbr", action="store_true")
    a = ap.parse_args()
    import sacrebleu
    from transformers import AutoTokenizer
    from gemma_mt import format as fmt
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import BeamSearchParams
    tok = AutoTokenizer.from_pretrained(a.model)
    files = set(os.listdir(a.flores)); rd = lambda f: [json.loads(l)["text"] for l in open(os.path.join(a.flores, f + ".jsonl"), encoding="utf-8")]
    en = rd("eng_Latn"); step = max(1, len(en) // a.n); idx = list(range(0, len(en), step))[: a.n]
    items = []
    for iso in a.langs:
        try:
            ref = rd(flores_code(iso, files))
        except KeyError:
            print("skip", iso); continue
        items += [(iso, en[i], ref[i]) for i in idx]
    prompts = []
    for iso, s, _ in items:
        ids = fmt.build_prompt(s, iso, tok); prompts.append(list(ids["input_ids"]) if isinstance(ids, dict) else list(ids))
    llm = LLM(model=a.model, dtype=a.dtype, gpu_memory_utilization=a.gpu_util, max_model_len=2048, enforce_eager=True, seed=0)
    stops = [1, 106]

    def dec(g):
        g = list(g)
        for i, t in enumerate(g):
            if t in stops or t == 0:
                g = g[:i]; break
        return unicodedata.normalize("NFC", tok.decode(g, skip_special_tokens=True)).strip()
    P = [{"prompt_token_ids": p} for p in prompts]
    gre = [dec(o.outputs[0].token_ids) for o in llm.generate(P, SamplingParams(temperature=0.0, max_tokens=a.max_new, stop_token_ids=stops))]
    print("greedy done", flush=True)
    if a.greedy_only:
        res = {"dtype": a.dtype, "n_per_language": len(idx), "hyp": {"items": items, "greedy": gre}, "systems": {}}
        per = {}
        for iso in a.langs:
            ii = [i for i, it in enumerate(items) if it[0] == iso]
            if ii: per[iso] = round(sacrebleu.corpus_chrf([gre[i] for i in ii], [[unicodedata.normalize("NFC", items[i][2]) for i in ii]], word_order=2).score, 2)
        res["systems"]["greedy"] = {"per_language": per, "macro": round(sum(per.values()) / len(per), 3)}
        json.dump(res, open(a.out, "w"), ensure_ascii=False); print(json.dumps({"greedy": res["systems"]["greedy"]["macro"]})); return
    if a.no_mbr:
        samp = []
    else:
        samp = llm.generate(P, SamplingParams(temperature=a.temperature, top_p=0.95, n=a.k, max_tokens=a.max_new, stop_token_ids=stops, seed=0))
    print("samples done", flush=True)
    mbr = []
    for g, o in zip(gre, samp if samp else [None] * len(gre)):
        if o is None:
            mbr.append(g); continue
        cands = [g] + [dec(c.token_ids) for c in o.outputs]
        best, bs = g, -1.0
        for i, c in enumerate(cands):
            s = sum(sacrebleu.sentence_chrf(c, [d], word_order=2).score for j, d in enumerate(cands) if j != i) / (len(cands) - 1) if c.strip() else -1.0
            if s > bs: best, bs = c, s
        mbr.append(best)
    print("mbr done", flush=True)
    bo = llm.beam_search(P, BeamSearchParams(beam_width=4, max_tokens=a.max_new, length_penalty=1.0))
    beam = []
    for p, o in zip(prompts, bo):
        seq = o.sequences[0]; beam.append(dec(seq.tokens[len(p):]))
    print("beam done", flush=True)
    res = {"n_per_language": len(idx), "systems": {}, "sampling": {"k": a.k, "temperature": a.temperature, "top_p": 0.95}, "hyp": {}}
    nfc = lambda x: unicodedata.normalize("NFC", x)
    for name, hyp in (("greedy", gre), ("beam4", beam)) + (() if a.no_mbr else (("mbr16", mbr),)):
        per = {}
        for iso in a.langs:
            ii = [i for i, it in enumerate(items) if it[0] == iso]
            if not ii: continue
            per[iso] = round(sacrebleu.corpus_chrf([hyp[i] for i in ii], [[nfc(items[i][2]) for i in ii]], word_order=2).score, 2)
        res["systems"][name] = {"per_language": per, "macro": round(sum(per.values()) / len(per), 3)}
    res["hyp"] = {"items": items, "greedy": gre, "beam4": beam, "mbr16": mbr}
    json.dump(res, open(a.out, "w"), ensure_ascii=False)
    print(json.dumps({k: v["macro"] for k, v in res["systems"].items()}))


if __name__ == "__main__":
    main()
