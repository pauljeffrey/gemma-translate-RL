"""E2: zero-shot en->xx translation of FLORES+ devtest with a chat model through vLLM (greedy). usage: e2_gen_chat.py NAME MODEL_DIR STYLE LANGS(comma iso) [gpu_util]
STYLE translatepsy = system+user prompt of the TranslatePsy-AfriSLM README / paper Table 15 (256 new tokens); gemma = single user turn with the same instruction."""
import sys, json, re, os
name, mdir, style, langs = sys.argv[1:5]; util = float(sys.argv[5]) if len(sys.argv) > 5 else 0.09
NAMES = {"afr":"Afrikaans","amh":"Amharic","hau":"Hausa","ibo":"Igbo","kin":"Kinyarwanda","lin":"Lingala","lug":"Luganda","mlg":"Malagasy","nya":"Nyanja","orm":"Oromo","sna":"Shona","som":"Somali","sot":"Southern Sotho","swh":"Swahili","tsn":"Tswana","wol":"Wolof","xho":"Xhosa","yor":"Yoruba","zul":"Zulu",
         "tir":"Tigrinya","kab":"Kabyle","bam":"Bambara","ewe":"Ewe","twi":"Twi","fon":"Fon","tso":"Tsonga","ssw":"Swati","run":"Rundi","kik":"Kikuyu","luo":"Luo","sag":"Sango","nso":"Northern Sotho","bem":"Bemba","tum":"Tumbuka","umb":"Umbundu","kmb":"Kimbundu","mos":"Mossi","kbp":"Kabiye","kam":"Kamba","lua":"Luba-Kasai","nus":"Nuer","dyu":"Dyula","knc":"Central Kanuri","kea":"Kabuverdianu"}
from vllm import LLM, SamplingParams
def main():
    rows = {}
    for l in open("/workspace/jobs/e2/refs_flores_plus.jsonl"):
        d = json.loads(l)
        if d["iso"] in langs.split(",") and (os.environ.get("ROWS") != "no_pair" or d["pair_seen"] is False): rows.setdefault(d["iso"], []).append(d)
    llm = LLM(mdir, dtype="bfloat16", gpu_memory_utilization=util, max_model_len=2048, enforce_eager=True, seed=0, limit_mm_per_prompt={"image": 0, "video": 0}) if os.environ.get("TEXTONLY") != "1" else LLM(mdir, dtype="bfloat16", gpu_memory_utilization=util, max_model_len=2048, enforce_eager=True, seed=0)
    nper = int(os.environ.get("NPER", "0"))
    if nper:
        rows = {k: v[:nper] for k, v in rows.items()}
    tok = llm.get_tokenizer()
    def prompt(text, tgt):
        user = f"Please translate the following English text into {tgt}: {text}.\n\nTranslation:"
        if style == "translatepsy":
            msgs = [{"role": "system", "content": f"You are a professional English to {tgt} translator. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to {tgt} grammar, vocabulary, and cultural sensitivities. Produce only the {tgt} translation, without any additional explanations or commentary. "}, {"role": "user", "content": user}]
        else:
            msgs = [{"role": "user", "content": f"You are a professional English to {tgt} translator. Produce only the {tgt} translation, without any additional explanations or commentary.\n\n" + user}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    out = open(f"/workspace/jobs/e2/hyp_{name}.jsonl", "w")
    for iso, rs in sorted(rows.items()):
        outs = llm.generate([prompt(r["english"], NAMES[iso]) for r in rs], sp, use_tqdm=False)
        for r, o in zip(rs, outs):
            t = o.outputs[0].text; t = re.sub(r"<think>.*?</think>", "", t, flags=re.S); t = (t.split("</think>")[-1] if "</think>" in t else t).strip()
            out.write(json.dumps({"iso": iso, "id": r["id"], "hyp": t, "finished": o.outputs[0].finish_reason == "stop"}, ensure_ascii=False) + "\n")
        out.flush(); print(iso, len(rs), flush=True)
    print("done")

if __name__ == "__main__":
    main()
