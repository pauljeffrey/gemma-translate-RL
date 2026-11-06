#!/usr/bin/env python
"""E-01 POST-HOC EXTENSION (the pre-registered grid peaked at its edge, freq 0.10): stronger penalties on the same dev set, SFT only. (grid on dev; one model per process, the choice is made by guard_choose.py): choose the decode-time loop guard ON THE FLORES+ DEV HELD-OUT HALF (assets/dev/bench_dev_hsel.jsonl: 18 languages x 40 sentences, split 'selection' of the
D2 rule), never on devtest. Pre-registered rule (written before any number): among the guard configurations below, take the one with the highest dev macro chrF++ of the SFT model
(the strongest control for the claim 'a decoding fix recovers the SFT's deficit'); a tie within 0.05 goes to the weaker penalty. The chosen parameters are then fixed and applied
unchanged to the SFT and the RL model on the full evaluation set. Both models are run through the grid for transparency; only the SFT decides.

    python guard_select.py --model sft=DIR --model rl=DIR --dev FILE --out guard_dev.json
"""
import argparse, importlib.util, json, os, sys, unicodedata

HERE = "/workspace/eval_385m/gemma_mt_tree"; sys.path.insert(0, HERE)
os.environ.setdefault("GEMMA_MT_TAG_MODE", "single_token")
GRID = [("greedy", 0.0, 1.0), ("freq010", 0.10, 1.0), ("freq020", 0.20, 1.0), ("freq030", 0.30, 1.0), ("freq050", 0.50, 1.0), ("freq020_rep102", 0.20, 1.02)]   # POST-HOC extension (E-01 grid edge)
ORDER = [g[0] for g in GRID]       # weaker first (tie-break)


def load_guards():
    spec = importlib.util.spec_from_file_location("g", "/workspace/clean_rerun/tree/gemma_mt/gemma_mt/rl/guards.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g); return g


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", action="append", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-util", type=float, default=0.2); ap.add_argument("--max-new", type=int, default=768); a = ap.parse_args()
    import sacrebleu
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from gemma_mt import format as fmt
    G = load_guards()
    rows = [json.loads(l) for l in open(a.dev, encoding="utf-8")]
    langs = sorted(set(r["iso"] for r in rows))
    res = {"rows": len(rows), "languages": langs, "grid": [{"name": n, "frequency_penalty": fp, "repetition_penalty": rp} for n, fp, rp in GRID], "max_new_tokens": a.max_new, "models": {}}
    for spec in a.model:
        name, path = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        prompts = []
        for r in rows:
            ids = fmt.build_prompt(r["english"], r["iso"], tok); prompts.append(list(ids["input_ids"]) if isinstance(ids, dict) else list(ids))
        llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=a.gpu_util, max_model_len=2048, enforce_eager=True, seed=0)
        stops = [1, 106]
        out = {}
        for gname, fp, rp in GRID:
            sp = SamplingParams(temperature=0.0, max_tokens=a.max_new, stop_token_ids=stops, frequency_penalty=fp, repetition_penalty=rp)
            o = llm.generate([{"prompt_token_ids": p} for p in prompts], sp)
            hyps, fin = [], []
            for x in o:
                ids = list(x.outputs[0].token_ids); cut = next((i for i, t in enumerate(ids) if t in stops or t == 0), None)
                fin.append(cut is not None or x.outputs[0].finish_reason == "stop")
                hyps.append(unicodedata.normalize("NFC", tok.decode(ids[:cut] if cut is not None else ids, skip_special_tokens=True)).strip())
            per = {}
            for l in langs:
                ii = [i for i, r in enumerate(rows) if r["iso"] == l]
                per[l] = sacrebleu.corpus_chrf([hyps[i] for i in ii], [[unicodedata.normalize("NFC", rows[i]["reference"]) for i in ii]], word_order=2).score
            loops = [bool(G.is_repetitive(h) or G.is_self_repetitive(h)) for h in hyps]
            out[gname] = {"chrfpp_macro": sum(per.values()) / len(per), "finished": sum(fin) / len(fin), "loop": sum(loops) / len(loops), "per_language": per}
            print(name, gname, round(out[gname]["chrfpp_macro"], 3), "finished", round(out[gname]["finished"], 3), "loop", round(out[gname]["loop"], 3), flush=True)
        res["models"][name] = out
        del llm
        import gc, torch; gc.collect(); torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1); print("WROTE", a.out)


if __name__ == "__main__":
    main()
