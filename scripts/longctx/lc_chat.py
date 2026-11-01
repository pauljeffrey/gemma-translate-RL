"""Zero-shot / chat translation models through vLLM (TranslatePsy-AfriSLM, Qwen3.5, Gemma-3-it) on the LC1 benchmark.  venv_vllm.
usage: lc_chat.py NAME MODEL_DIR STYLE LANGS(csv|all) [--modes whole,sent] [--gpu-util 0.07] [--max-seqs 64] [--skip-bins b4096]
STYLE: translatepsy (system+user, the model card / paper prompt) | gemma (one user turn).  Output budget per request = 2.5 x (expected target tokens) + slack,
expected = source Gemma tokens x the model-tokenizer/Gemma token ratio measured on the references of that language (so a model that stops late is not cut,
a model that loops is stopped).  Output hyp/NAME__{whole,sent}.jsonl."""
import argparse, collections, json, os, re, sys, time, statistics
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1"); os.environ.setdefault("DO_NOT_TRACK", "1")
from lc_common import *
from vllm import LLM, SamplingParams
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("model_dir"); ap.add_argument("style"); ap.add_argument("langs")
    ap.add_argument("--modes", default="whole,sent"); ap.add_argument("--gpu-util", type=float, default=0.07); ap.add_argument("--max-seqs", type=int, default=64)
    ap.add_argument("--skip-bins", default=""); ap.add_argument("--max-model-len", type=int, default=24576); ap.add_argument("--mm", action="store_true", help="multimodal wrapper (Qwen3.5): limit image/video to 0")
    a = ap.parse_args()
    from gemma_mt.eval.longdoc import join_sentences, split_long_sentence, norm_text
    from transformers import AutoTokenizer


    docs, items = load_docs(), load_items()
    skip = set(x for x in a.skip_bins.split(",") if x)
    langs = None if a.langs == "all" else set(a.langs.split(","))
    items = [i for i in items if i["bin"] not in skip and (langs is None or i["iso"] in langs)]
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    refs = {**load_refs("human")}
    refs.update({r["iid"]: r for r in rd(WORK + "/refs_nllb_core.jsonl")})
    # tokenizer ratio per language (model tokens of the reference text / Gemma tokens of the English source)
    rat = collections.defaultdict(list)
    for it in items:
        r = refs.get(it["iid"])
        if r and docs[it["doc_id"]]["src_tokens"] >= 100:
            rat[it["iso"]].append(len(tok(" ".join(r["ref_sents"]), add_special_tokens=False)["input_ids"]) / docs[it["doc_id"]]["src_tokens"])
    ratio = {k: statistics.median(v) for k, v in rat.items()}
    print("target/source token ratio by language:", {k: round(v, 2) for k, v in ratio.items()}, flush=True)
    tg = NAMES

    def prompt(text, iso):
        t = tg[iso]
        user = f"Please translate the following English text into {t}: {text}.\n\nTranslation:"
        if a.style == "translatepsy":
            msgs = [{"role": "system", "content": f"You are a professional English to {t} translator. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to {t} grammar, vocabulary, and cultural sensitivities. Produce only the {t} translation, without any additional explanations or commentary. "}, {"role": "user", "content": user}]
        else:
            msgs = [{"role": "user", "content": f"You are a professional English to {t} translator. Produce only the {t} translation, without any additional explanations or commentary.\n\n" + user}]
        try:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def clean(t):
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
        if "</think>" in t:
            t = t.split("</think>")[-1]
        return t.strip()

    kw = dict(model=a.model_dir, dtype="bfloat16", gpu_memory_utilization=a.gpu_util, max_model_len=a.max_model_len, max_num_seqs=a.max_seqs, enforce_eager=True, seed=0)
    if a.mm:
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    llm = LLM(**kw)
    modes = a.modes.split(",")

    def gen(reqs):
        """reqs: [(key, prompt_text, max_new)] -> {key: (text, finished, ntok)}"""
        reqs = sorted(reqs, key=lambda r: -(len(r[1]) + r[2]))
        res, t0, n = {}, time.time(), 0
        CH = 1500
        for s in range(0, len(reqs), CH):
            part = reqs[s:s + CH]
            outs = llm.generate([r[1] for r in part], [SamplingParams(temperature=0.0, max_tokens=r[2]) for r in part], use_tqdm=False)
            for r, o in zip(part, outs):
                c = o.outputs[0]; res[r[0]] = (clean(c.text), c.finish_reason == "stop", len(c.token_ids)); n += len(c.token_ids)
            print(f"  {min(s + CH, len(reqs))}/{len(reqs)} reqs {n} tok {time.time() - t0:.0f}s", flush=True)
        return res

    if "sent" in modes and not os.path.exists(f"{HYP}/{a.name}__sent.jsonl"):
        need = {}
        for it in items:
            for s in docs[it["doc_id"]]["sents"]:
                for p in split_long_sentence(s, 700):
                    need.setdefault((it["iso"], p), None)
        reqs = []
        for k, (iso, p) in enumerate(sorted(need)):
            st = max(4, int(len(p) / 4.0))
            reqs.append((k, prompt(p, iso), min(int(2.5 * ratio.get(iso, 3.0) * st) + 64, 1024)))
        keys = {kk: i for i, kk in enumerate(sorted(need))}
        print("sentence units", len(reqs), flush=True)
        res = gen(reqs)
        unit = {kk: res[i] for kk, i in keys.items()}
        rows = []
        for it in items:
            d = docs[it["doc_id"]]; so, fin, nt = [], True, 0
            for s in d["sents"]:
                ps = [unit[(it["iso"], p)] for p in split_long_sentence(s, 700)]
                so.append(norm_text(" ".join(x[0] for x in ps))); fin &= all(x[1] for x in ps); nt += sum(x[2] for x in ps)
            rows.append({"iid": it["iid"], "hyp": join_sentences(so, d["para_starts"]), "hyp_sents": so, "finished": fin, "new_tokens": nt, "status": "ok"})
        wr(f"{HYP}/{a.name}__sent.jsonl", rows); print("sent done", len(rows), flush=True)

    if "whole" in modes and not os.path.exists(f"{HYP}/{a.name}__whole.jsonl"):
        reqs = []
        for it in items:
            d = docs[it["doc_id"]]
            mx = min(int(2.5 * ratio.get(it["iso"], 3.0) * d["src_tokens"]) + 128, 16384)
            reqs.append((it["iid"], prompt(d["text"], it["iso"]), mx))
        print("whole docs", len(reqs), flush=True)
        res = gen(reqs)
        wr(f"{HYP}/{a.name}__whole.jsonl", [{"iid": it["iid"], "hyp": res[it["iid"]][0], "finished": res[it["iid"]][1], "new_tokens": res[it["iid"]][2], "status": "ok"} for it in items])
        print("whole done", flush=True)
    print("DONE", a.name)


if __name__ == "__main__":
    main()
