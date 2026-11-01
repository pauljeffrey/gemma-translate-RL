# Lead's one-off probe (2026-10-23): tokens needed for one 74-character Amharic sample (no spaces) and its English translation.
# Run on the GPU box with the repo env (HF tokens in /workspace/sft_build); results: paper/data/tokenizer_amharic_probe_2026-10-23.csv
# CAVEAT: a single short sample, one sentence group; it supports a ranking only, not precise ratios (the FLORES+ measurement in the repo is the main evidence).
from transformers import AutoTokenizer
txt_am = "ኢትዮጵያ በምስራቅ አፍሪካ የምትገኝ ሀገር ናት። ዋና ከተማዋ አዲስ አበባ ነው፣ እና ብዙ ቋንቋዎች በዚያ ይነገራሉ። እኔ ደህና ነኝ፣ አመሰግናለሁ።"
txt_en = "Ethiopia is a country in East Africa. Its capital is Addis Ababa, and many languages are spoken there. I am fine, thank you."
chars = len(txt_am.replace(" ", ""))
models = [("gemma-3-270m (ours)", "/workspace/sft_build/runs/final_1433"), ("SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M"),
          ("SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M"), ("Qwen3-0.6B", "Qwen/Qwen3-0.6B"), ("GPT-2", "gpt2"),
          ("Pythia-160M", "EleutherAI/pythia-160m"), ("BLOOM-560M", "bigscience/bloom-560m"), ("mGPT", "ai-forever/mGPT")]
for name, p in models:
    t = AutoTokenizer.from_pretrained(p)
    a = len(t(txt_am, add_special_tokens=False)["input_ids"]); e = len(t(txt_en, add_special_tokens=False)["input_ids"])
    print(f"{name:22s} am_tokens={a:4d} per_char={a/chars:.2f} am/en={a/e:.2f} vocab={len(t)}")
