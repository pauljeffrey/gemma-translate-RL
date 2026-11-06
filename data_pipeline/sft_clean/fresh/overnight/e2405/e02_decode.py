#!/usr/bin/env python
"""E-02 generation: greedy / MBR-16 / beam-4 at 768 new tokens on 20 languages x 200 FLORES+ devtest sentences (every 5th), one vLLM engine per process.
(Successor of decode_study_385m.py / decode_study2.py; same sentence selection `idx = range(0, len(en), len(en)//200)[:200]`, same prompt builder, same MBR recipe.)

    GEMMA_MT_TAG_MODE=single_token python e02_decode.py --model DIR --name sft --out OUT.json --systems greedy,mbr16,beam4 [--eos-mode eot|eos] [--gpu-util 0.6]

Differences to the old study, all deliberate and documented in the report:
  * max_new_tokens 768 (old: 256); `finished` recorded for every output (a stop token <end_of_turn>=106 / <eos>=1 / <pad>=0 before the cap);
  * beam-4 (AMENDMENT 1 of the pre-registration, made after a 120-sentence smoke test of the SFT and before any real result): vLLM 0.32.0 `LLM.beam_search` (a) completes a beam only on
    tokenizer.eos_token_id = <eos> (1), not <end_of_turn> (106) -> the engine gets a tokenizer copy with eos <end_of_turn> (asserted 106); (b) never stops an instance before max_tokens and ranks completed
    and still-running beams by cumulative log-prob / length, so at 768 tokens a runaway continuation beats every ended hypothesis (smoke: 0 of 120 SFT outputs ended). The primary `beam4` therefore uses vLLM's own
    beam step with the standard stopping rule (an instance stops when 4 hypotheses have ended; `beam4` = --beam-stop width, fairseq / HF early_stopping=True); `beam4hf` = the HF early_stopping=False rule
    (diagnostic); --beam-impl vllm = vLLM's exhaustive loop (diagnostic, smoke only); --eos-mode eos reproduces the 256-token study's eos=<eos> (diagnostic).
  * MBR utility = sentence chrF++ (sacrebleu CHRF word_order 2) mean agreement with the other 16 candidates (greedy + 16 samples at T 0.7, top-p 0.95): 'metric-matched'.
"""
import argparse, json, os, shutil, sys, time, unicodedata
from multiprocessing import Pool

sys.path.insert(0, os.environ.get("GEMMA_MT_TREE", "/workspace/eval_385m/gemma_mt_tree"))
LANGS = "yor hau ibo swh amh zul xho kin som wol lug sna twi ewe lin tir orm nso tsn afr".split()
STOPS = [1, 106]


def flores_code(iso, files):
    from gemma_mt import languages as L
    c = getattr(L.LANGUAGES[iso], "flores", None)
    if c and f"{c}.jsonl" in files:
        return c
    for f in files:
        if f.startswith(iso + "_"):
            return f[:-6]
    raise KeyError(iso)


def _mbr_pick(cands):
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=2)
    best, bs = 0, -1.0
    for i, c in enumerate(cands):
        if not c.strip():
            continue
        s = sum(chrf.sentence_score(c, [d]).score for j, d in enumerate(cands) if j != i) / (len(cands) - 1)
        if s > bs:
            best, bs = i, s
    return best


def beam_early(llm, P, width, max_tokens, eos_id, rule="width"):
    """FALLBACK only (--beam-impl early): vLLM's own beam loop on the instances that are not finished yet. An instance is finished when it holds >= width completed beams and no active beam
    scores above the worst of the best `width` completed ones (the Hugging Face early_stopping=False rule, length penalty 1.0). vLLM's `LLM.beam_search` has no such rule and runs every instance to max_tokens."""
    from vllm import SamplingParams
    from vllm.entrypoints.generate.beam_search.utils import BeamSearchInstance, create_sort_beams_key_function
    key = create_sort_beams_key_function(eos_id, 1.0)
    base = SamplingParams(logprobs=2 * width, max_tokens=1, temperature=0.0, watermarking=False, detokenize=False, skip_clone=True)
    insts = [BeamSearchInstance(p, lora_request=None, logprobs=None) for p in llm._preprocess_cmpl(P)]
    def done(i):
        if len(i.completed) < width: return False
        if rule == "width": return True          # standard stopping (fairseq / HF early_stopping=True): stop when `width` hypotheses have ended
        worst = key(sorted(i.completed, key=key, reverse=True)[:width][-1])
        return max(key(b) for b in i.beams) <= worst
    active = insts
    for t in range(max_tokens):
        active = [i for i in active if i.beams and not done(i)]
        if not active: break
        llm._beam_search_step(instances_batch=active, base_sampling_params=base, eos_token_id=eos_id, ignore_eos=False, beam_width=width, sort_beams_key=key,
                              structured_output_backend=None, structured_output_key=None, structured_output_bitmask=None)
        if t % 50 == 0: print("beam-early step", t, "active instances", len(active), flush=True)
    res = []
    for i in insts:
        if not (rule == "width" and len(i.completed) >= width):      # stopped by the width rule: only the ended hypotheses compete (running beams are not candidates)
            i.completed.extend(i.beams)
        res.append(sorted(i.completed, key=key, reverse=True)[:width])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--flores", default="/workspace/eval_385m/bench/test_sets/flores_plus/devtest")
    ap.add_argument("--langs", nargs="*", default=LANGS); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--systems", default="greedy,mbr16,beam4"); ap.add_argument("--eos-mode", default="eot", choices=["eot", "eos"])
    ap.add_argument("--gpu-util", type=float, default=0.6); ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first LIMIT items")
    ap.add_argument("--tok-dir", default="/dev/shm/e02/tok_eot"); ap.add_argument("--beam-impl", default="early", choices=["vllm", "early"]); ap.add_argument("--beam-stop", default="width", choices=["width", "hf"])
    a = ap.parse_args()
    systems = a.systems.split(",")
    from transformers import AutoTokenizer
    from gemma_mt import format as fmt
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import BeamSearchParams
    import vllm
    tok = AutoTokenizer.from_pretrained(a.model)
    files = set(os.listdir(a.flores))
    rd = lambda f: [json.loads(l)["text"] for l in open(os.path.join(a.flores, f + ".jsonl"), encoding="utf-8")]
    en = rd("eng_Latn"); step = max(1, len(en) // a.n); idx = list(range(0, len(en), step))[: a.n]
    items = []
    for iso in a.langs:
        try:
            ref = rd(flores_code(iso, files))
        except KeyError:
            print("skip", iso); continue
        items += [(iso, en[i], ref[i], i) for i in idx]
    if a.limit:
        items = items[:: max(1, len(items) // a.limit)][: a.limit]
    prompts = []
    for iso, s, _, _ in items:
        ids = fmt.build_prompt(s, iso, tok); prompts.append(list(ids["input_ids"]) if isinstance(ids, dict) else list(ids))
    tok_arg = a.model
    if a.eos_mode == "eot" and ("beam4" in systems or "beam4hf" in systems):
        shutil.rmtree(a.tok_dir, ignore_errors=True); os.makedirs(a.tok_dir)
        for f in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"):
            if os.path.exists(os.path.join(a.model, f)): shutil.copy(os.path.join(a.model, f), a.tok_dir)
        for f in ("tokenizer_config.json", "special_tokens_map.json"):
            p = os.path.join(a.tok_dir, f); d = json.load(open(p)); d["eos_token"] = "<end_of_turn>"; json.dump(d, open(p, "w"), ensure_ascii=False)
        tok_arg = a.tok_dir
    t0 = time.time()
    llm = LLM(model=a.model, tokenizer=tok_arg, dtype="bfloat16", gpu_memory_utilization=a.gpu_util, max_model_len=2048, enforce_eager=True, seed=0)
    eos_engine = (llm.renderer.get_tokenizer() if hasattr(llm, 'renderer') else llm.get_tokenizer()).eos_token_id      # the object beam_search reads
    print("vllm", vllm.__version__, "engine tokenizer eos_token_id", eos_engine, "eos-mode", a.eos_mode, "load s", round(time.time() - t0), flush=True)
    if ("beam4" in systems or "beam4hf" in systems) and a.eos_mode == "eot":
        assert eos_engine == 106, f"beam eos is {eos_engine}, expected 106"

    def dec(tokens, fin_hint=False):
        g = list(tokens); fin = bool(fin_hint)
        for i, t in enumerate(g):
            if t in STOPS or t == 0:
                g = g[:i]; fin = True; break
        return {"hyp": unicodedata.normalize("NFC", tok.decode(g, skip_special_tokens=True)).strip(), "finished": fin, "ntok": len(g)}
    P = [{"prompt_token_ids": p} for p in prompts]
    res = {"model": a.model, "name": a.name, "vllm": vllm.__version__, "dtype": "bfloat16", "max_new_tokens": a.max_new, "eos_mode": a.eos_mode, "engine_eos_id": eos_engine,
           "n_per_language": len(idx), "langs": a.langs, "items": items, "hyp": {}, "timing_s": {}}
    gre = None
    if "greedy" in systems or "mbr16" in systems:
        t = time.time()
        outs = llm.generate(P, SamplingParams(temperature=0.0, max_tokens=a.max_new, stop_token_ids=STOPS), use_tqdm=False)
        gre = [dec(o.outputs[0].token_ids, o.outputs[0].finish_reason == "stop") for o in outs]
        res["hyp"]["greedy"] = gre; res["timing_s"]["greedy"] = round(time.time() - t); print("greedy done", res["timing_s"]["greedy"], flush=True)
    if "mbr16" in systems:
        t = time.time()
        samp = llm.generate(P, SamplingParams(temperature=a.temperature, top_p=0.95, n=a.k, max_tokens=a.max_new, stop_token_ids=STOPS, seed=0), use_tqdm=False)
        res["timing_s"]["samples"] = round(time.time() - t); print("samples done", res["timing_s"]["samples"], flush=True)
        cand = []
        for g, o in zip(gre, samp):
            cand.append([g] + [dec(c.token_ids, c.finish_reason == "stop") for c in o.outputs])
        t = time.time()
        with Pool(min(48, os.cpu_count() or 8)) as pool:
            pick = pool.map(_mbr_pick, [[c["hyp"] for c in cs] for cs in cand], chunksize=8)
        res["hyp"]["mbr16"] = [dict(cs[p], pick=p) for cs, p in zip(cand, pick)]
        res["timing_s"]["mbr_pick"] = round(time.time() - t)
        res["mbr_pick_is_greedy_share"] = round(sum(1 for p in pick if p == 0) / len(pick), 4)
        json.dump({"cand": [[c["hyp"] for c in cs] for cs in cand], "finished": [[c["finished"] for c in cs] for cs in cand]}, open(a.out + ".mbr_candidates.json", "w"), ensure_ascii=False)
        print("mbr done", res["timing_s"]["mbr_pick"], flush=True)
    for bname in [b for b in ("beam4", "beam4hf") if b in systems]:
        t = time.time(); beam = []
        if a.beam_impl == "vllm":
            bo = llm.beam_search(P, BeamSearchParams(beam_width=4, max_tokens=a.max_new, length_penalty=1.0), use_tqdm=True)
            for p, o in zip(prompts, bo):
                beam.append(dec(o.sequences[0].tokens[len(p):]))
        else:
            rule = "hf" if bname == "beam4hf" else a.beam_stop
            for p, seqs in zip(prompts, beam_early(llm, P, 4, a.max_new, eos_engine, rule)):
                beam.append(dec(seqs[0].tokens[len(p):]))
        key = bname if a.eos_mode == "eot" else bname + "_oldeos"
        res["hyp"][key] = beam; res["timing_s"][key] = round(time.time() - t); print("beam done", key, res["timing_s"][key], flush=True)
    res["beam_impl"] = a.beam_impl
    json.dump(res, open(a.out, "w"), ensure_ascii=False)
    print("WROTE", a.out, {k: round(sum(h["finished"] for h in v) / len(v), 4) for k, v in res["hyp"].items()}, flush=True)


if __name__ == "__main__":
    main()
