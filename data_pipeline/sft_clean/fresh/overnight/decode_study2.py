#!/usr/bin/env python
"""E-02: greedy vs beam-4 vs MBR-16 at 768 new tokens, 20 languages x 200 FLORES+ devtest sentences (every 5th), one vLLM engine, with the FINISHED flag of every output (decode_study_385m.py,
extended: the old study capped at 256 tokens and did not record finishing).
    GEMMA_MT_TAG_MODE=single_token python decode_study2.py --model DIR --flores DIR/devtest --out out.json [--gpu-util 0.2]
MBR utility = chrF++ (metric-matched): report AfriCOMET (decode_comet.py) and BLEU next to it."""
import argparse, json, os, sys, unicodedata
sys.path.insert(0, "/workspace/eval_385m/gemma_mt_tree")
LANGS = "yor hau ibo swh amh zul xho kin som wol lug sna twi ewe lin tir orm nso tsn afr".split()


def flores_code(iso, files):
    from gemma_mt import languages as L
    c = getattr(L.LANGUAGES[iso], "flores", None)
    if c and f"{c}.jsonl" in files: return c
    for f in files:
        if f.startswith(iso + "_"): return f[:-6]
    raise KeyError(iso)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--flores", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="*", default=LANGS); ap.add_argument("--n", type=int, default=200); ap.add_argument("--gpu-util", type=float, default=0.2)
    ap.add_argument("--max-new", type=int, default=768); ap.add_argument("--k", type=int, default=16); ap.add_argument("--temperature", type=float, default=0.7); a = ap.parse_args()
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
        try: ref = rd(flores_code(iso, files))
        except KeyError: print("skip", iso); continue
        items += [(iso, en[i], ref[i]) for i in idx]
    prompts = []
    for iso, s, _ in items:
        ids = fmt.build_prompt(s, iso, tok); prompts.append(list(ids["input_ids"]) if isinstance(ids, dict) else list(ids))
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_util, max_model_len=2048, enforce_eager=True, seed=0)
    stops = [1, 106]

    def dec(g):
        g = list(g); fin = False
        for i, t in enumerate(g):
            if t in stops or t == 0: g = g[:i]; fin = True; break
        return unicodedata.normalize("NFC", tok.decode(g, skip_special_tokens=True)).strip(), fin
    P = [{"prompt_token_ids": p} for p in prompts]
    G = [dec(o.outputs[0].token_ids) for o in llm.generate(P, SamplingParams(temperature=0.0, max_tokens=a.max_new, stop_token_ids=stops))]
    print("greedy done", flush=True)
    samp = llm.generate(P, SamplingParams(temperature=a.temperature, top_p=0.95, n=a.k, max_tokens=a.max_new, stop_token_ids=stops, seed=0))
    print("samples done", flush=True)
    mbr = []
    for g, o in zip(G, samp):
        cands = [g] + [dec(c.token_ids) for c in o.outputs]
        best, bs = g, -1.0
        for i, (c, f) in enumerate(cands):
            s = sum(sacrebleu.sentence_chrf(c, [d], word_order=2).score for j, (d, _) in enumerate(cands) if j != i) / (len(cands) - 1) if c.strip() else -1.0
            if s > bs: best, bs = (c, f), s
        mbr.append(best)
    print("mbr done", flush=True)
    bo = llm.beam_search(P, BeamSearchParams(beam_width=4, max_tokens=a.max_new, length_penalty=1.0))
    beam = []
    for p, o in zip(prompts, bo):
        beam.append(dec(o.sequences[0].tokens[len(p):]))
    print("beam done", flush=True)
    nfc = lambda x: unicodedata.normalize("NFC", x)
    res = {"model": a.model, "n_per_language": len(idx), "max_new_tokens": a.max_new, "items": items, "systems": {}, "hyp": {}}
    for name, hyp in (("greedy", G), ("beam4", beam), ("mbr16", mbr)):
        per = {}; fin = []
        for iso in a.langs:
            ii = [i for i, it in enumerate(items) if it[0] == iso]
            if not ii: continue
            per[iso] = round(sacrebleu.corpus_chrf([hyp[i][0] for i in ii], [[nfc(items[i][2]) for i in ii]], word_order=2).score, 2)
        fin = [h[1] for h in hyp]
        res["systems"][name] = {"per_language": per, "macro": round(sum(per.values()) / len(per), 3), "finished": round(sum(fin) / len(fin), 4)}
        res["hyp"][name] = [{"hyp": h[0], "finished": h[1]} for h in hyp]
    json.dump(res, open(a.out, "w"), ensure_ascii=False)
    print(json.dumps({k: (v["macro"], v["finished"]) for k, v in res["systems"].items()}))


if __name__ == "__main__":
    main()
