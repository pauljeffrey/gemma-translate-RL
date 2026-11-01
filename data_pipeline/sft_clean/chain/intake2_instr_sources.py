"""Source registry of intake2_instr.py (instruction / chat / QA streams). Priority order = order of the `add` calls (first source keeps cross-source duplicates).

Licence classes: commercial_ok | share_alike | unspecified. `synthetic=True` = machine-translated or LLM-generated (at least the instruction or the answer side).
Everything not listed here was NOT processed: see reports/intake2_instr_2026-10-31.md (skipped-and-why).
"""
import json, re

NEW_OK = {"din", "teo", "adh", "arz", "ary", "aeb", "bci"}      # African languages that are not in gemma_mt/languages.py; listed as NEW in the report


def register(M):
    add, S_instr, S_qa, S_msgs, S_chatml, rx_files, dir_iso, col_iso, file_iso, norm_iso, clean = (
        M.add, M.S_instr, M.S_qa, M.S_msgs, M.S_chatml, M.rx_files, M.dir_iso, M.col_iso, M.file_iso, M.norm_iso, M.clean)
    CO, SA, UN = "commercial_ok", "share_alike", "unspecified"
    M.NEW_OK = NEW_OK

    def S_msgs2(col, rk="role", ck="content", strip_think=False, drop_system=False, positional=False):
        base = S_msgs(col, rk, ck)

        def f(r, path):
            t = base(r, path)
            if not t:
                return t
            out = []
            for i_, (role, c) in enumerate(t):
                if positional:
                    role = "user" if i_ % 2 == 0 else "assistant"
                if drop_system and str(role).lower() == "system":
                    continue
                if strip_think and isinstance(c, str):
                    c = re.sub(r"<think>.*?</think>", "", c, flags=re.S)
                    if "<think>" in c:                       # unterminated reasoning block: drop the whole row
                        return None
                out.append((role, c))
            return out
        return f

    # ---------------------------------------------------------------------------------------------------------- human-written outputs first (these keep cross-source duplicates)
    add("aya_dataset", "CohereLabs/aya_dataset", rx_files(r"^data/train-"), S_instr("inputs", None, "targets"), col_iso("language_code"), CO, "apache-2.0 (human-annotated Aya dataset; NOT the NLLB-translated Aya collection)",
        task="instruction", note="human-annotated prompts and completions (Aya Annotation Platform); African-language rows only", order=10)
    add("muri_it", "akoksal/muri-it-language-split",
        rx_files(r"^(afr|amh|bam|din|ewe|fon|ful|hau|ibo|kbp|kik|kin|kon|lin|lug|mlg|nso|nya|orm|run|sag|sna|som|sot|ssw|swa|tir|tsn|tso|tum|ven|wol|xho|yor|zul|aka)/train-"),
        S_instr("input", None, "output"), dir_iso(0), CO,
        "apache-2.0 (card). Reverse instructions: outputs are human-written (CulturaX / Wikipedia); the instruction side is LLM-generated from an English translation and translated back (paper arXiv:2409.12958 uses NLLB-200: instruction side is MT output)",
        synthetic=False, synth_fn=lambda r, p: (r.get("dataset_name") == "MRI"), task_fn=lambda r, p: "instruction",
        note="synthetic=true where dataset_name == MRI (reverse-instruction rows: LLM + NLLB-translated instruction, human output); false for rows taken from other resources", order=11, vs_old=True)
    add("tydiqa_swh", "google-research-datasets/tydiqa", rx_files(r"^secondary_task/train-"), lambda r, p: [("user", clean(r.get("context")) + "\n\n" + clean(r.get("question"))), ("assistant", clean((r.get("answers") or {}).get("text", [""])[0] if (r.get("answers") or {}).get("text") else ""))],
        lambda p, r: "swh" if str(r.get("id", "")).startswith("swahili") else None, CO, "apache-2.0 (TyDi QA train split, gold-passage task; dev split not used)", task="qa", order=12)
    add("webfaq_v2", "PaDaS-Lab/webfaq-v2", rx_files(r"^data/(afr|amh|arz|mlg|som|swa|yor|zul|xho|hau|ibo|kin|sna|nya|sot|tsn|tir|orm|wol|lin|lug|ewe|bam|twi|aka|ful)\.parquet$"),
        S_qa("question", "answer"), lambda p, r: norm_iso(re.sub(r"\.parquet$", "", p.split("/")[-1])), UN, "no licence stated on the card (web FAQ pairs mined from schema.org markup, WebFAQ); disclose", task="qa",
        rowfilter=lambda r, p: (r.get("semantic_similarity_score") or 0) >= 0.7, note="real web FAQ pairs; question/answer similarity >= 0.7", order=13)
    add("yoseali_amh", "YoseAli/amharic-llm-training-data", rx_files(r"^data/train-"), S_instr("instruction", "input", "output"), "amh", CO, "mit (card)", task="instruction",
        rowfilter=lambda r, p: not (clean(r.get("instruction")).lower().startswith("complete the following") and not clean(r.get("input"))),
        note="rows whose instruction is the bare 'Complete the following task' with no input are raw text continuations and are dropped", order=14)
    add("african_ultrachat", "masakhane/african-ultrachat", rx_files(r"^train/"), S_msgs2("messages"), file_iso({"amharic": "amh", "hausa": "hau", "igbo": "ibo", "kinyarwanda": "kin", "shona": "sna", "somali": "som", "sesotho": "sot", "swahili": "swh", "xhosa": "xho", "yoruba": "yor", "zulu": "zul"}), CO,
        "mit (card)", synthetic=True, task="chat", note="UltraChat translated into 11 African languages (machine-translated)", order=20)
    add("uganda_multilingual_qa", "BUAIR/Uganda-Multilingual-QA", rx_files(r"^(luganda|runyankore|ateso|japadhola)/"), S_qa("question", "answer"),
        lambda p, r: {"luganda": "lug", "runyankore": "nyn", "ateso": "teo", "japadhola": "adh"}.get(p.split("/")[0]), CO, "cc-by-4.0 (card)", synthetic=True, task="qa", note="agricultural QA; machine-translated from English originals (English-language question/answer style)", order=21)
    # small hand-collected / community sets
    add("twi_everyday_conv", "ik/twi_everyday_conversation", rx_files(r"\.parquet$"), S_qa("question", "answer"), "twi", CO, "card states a permissive licence (catalogue: commercial_ok)", task="chat", order=30)
    add("oromiffa_convo", "kkenna/oromiffa_convo_data", rx_files(r"\.jsonl$"), S_msgs2("messages"), "orm", UN, "no licence stated", synthetic=True, task="chat", note="small, probably LLM-written", order=31)
    add("pidgin_clean", "Ephraimmm/pidgin_clean", rx_files(r"\.jsonl$"), S_msgs2("conversations"), "pcm", CO, "catalogue: commercial_ok", task="chat", order=32)
    add("lingala_alpaca", "Congo-digital-service/Lingala-Alpaca-Dataset-Augmented", rx_files(r"^data/train-"), S_instr("instruction", "input", "output"), "lin", UN, "no licence stated", task="instruction", order=33)
    add("kon_disolo_qa", "Svngoku/disolo-ya-afrika-qa", rx_files(r"^data/train-"), S_qa("question_text", "explanation"), "kon", CO, "catalogue: commercial_ok", task="qa", synthetic=True, note="LLM-written explanations", order=34)
    add("tigqa", "Hailay/tigqa-extractive-qa", rx_files(r"\.parquet$"), lambda r, p: [("user", clean(r.get("context")) + "\n\n" + clean(r.get("question"))), ("assistant", clean((r.get("answers") or {}).get("text", [""])[0] if (r.get("answers") or {}).get("text") else ""))],
        "tir", CO, "catalogue: commercial_ok", task="qa", order=35)
    add("kin_alpaca_ukweli", "map-boy/ukwelitimes-kinyarwanda-alpaca", rx_files(r"\.parquet$"), lambda r, p: _alpaca_text(r.get("text")), "kin", UN, "no licence stated", synthetic=True, task="instruction", order=36)
    add("kin_roleplay", "jojo-ai-mst/Roleplay-Kinyarwanda", rx_files(r"\.csv$"), S_instr("amabwiriza", "Iyinjiza", "igisubizo"), "kin", CO, "catalogue: commercial_ok (GPTeacher translation)", synthetic=True, task="chat", order=37)
    add("setswana_gpt4o", "OxxoCodes/gpt4o-setswana-instruct", rx_files(r"^data/train-"), S_msgs2("conversation"), "tsn", UN, "no licence stated (GPT-4o generated)", synthetic=True, task="instruction", order=38)
    add("luganda_fln", "CraneAILabs/luganda-fln-training-data", rx_files(r"\.jsonl$"), lambda r, p: _gemma_text(r.get("text")), "lug", CO, "catalogue: commercial_ok", synthetic=True, task="instruction", note="LLM-generated Luganda lesson content", order=39)
    add("ugandaedu_luganda", "gimmy256/UgandaEdu-5K", rx_files(r"_luganda\.jsonl$"), lambda r, p: [("user", clean(r.get("prompt") or r.get("instruction") or r.get("input"))), ("assistant", clean(r.get("response") or r.get("output")))], "lug", CO, "catalogue: commercial_ok", synthetic=True, task="qa", order=40)
    add("bambara_lm_qa", "oza75/bambara-lm-qa", rx_files(r"^data/"), S_qa("question", "answer"), "bam", UN, "no licence stated (templated LM tasks over Bambara text)", task="other", order=41)
    add("bambara_conv", "Makan09/Bambara-dataset_conversation", rx_files(r"\.jsonl$"), S_instr("instruction", "input", "output"), "bam", CO, "apache-2.0 (card)", synthetic=True, task="instruction", note="LLM-assisted", order=42)
    add("som_conversations", "yacdev/somali-100k-saaxiib-conversations-v2", rx_files(r"\.parquet$"), S_msgs2("messages"), "som", CO, "apache-2.0 (card)", synthetic=True, task="chat", order=43)
    # translated Alpaca-style sets (machine-translated: synthetic)
    add("alpaca_twi", "ik/twi_alpaca-cleaned", rx_files(r"^data/train-"), S_instr("instruction", "input", "output"), "twi", UN, "no licence stated (Alpaca translated)", synthetic=True, task="instruction", order=50)
    add("alpaca_swahili", "Mollel/alpaca-swahili", rx_files(r"^data/train-"), S_instr("instruction", "input", "output"), "swh", UN, "no licence stated (Alpaca translated)", synthetic=True, task="instruction", order=51)
    add("alpaca_zulu", "africa-intelligence/alpaca-zulu", rx_files(r"\.csv$"), S_instr("zu-instruction", "zu-input", "zu-output"), "zul", CO, "mit (card)", synthetic=True, task="instruction", order=52)
    add("alpaca_wolof", "ngia/alpaca-data-in-wolof", rx_files(r"\.csv$"), S_instr("instruction", "input", "output"), "wol", CO, "apache-2.0 (card)", synthetic=True, task="instruction", order=53)
    add("alpaca_somali_a", "burtugeey/alpaca_somali_final", rx_files(r"\.json$"), S_instr("instruction", "input", "output"), "som", CO, "apache-2.0 (card)", synthetic=True, task="instruction", order=54)
    add("alpaca_somali_b", "burtugeey/Somali_alpaca", rx_files(r"\.(json|jsonl|parquet|csv)$"), S_instr("instruction", "input", "output"), "som", CO, "mit (card)", synthetic=True, task="instruction", order=55)
    add("alpaca_somali_c", "burtugeey/Somali_dataset", rx_files(r"\.(json|jsonl|parquet|csv)$"), S_instr("instruction", "input", "output"), "som", CO, "mit (card)", synthetic=True, task="instruction", order=56)
    add("math_malagasy", "Lo-Renz-O/grade-school-math-instructions-Malagasy", rx_files(r"\.jsonl$"), S_instr("instruction", None, "response"), "mlg", CO, "catalogue: commercial_ok (GSM8K translated)", synthetic=True, task="instruction", order=57)
    add("amharic_freedom_alpaca", "addisai/FreedomIntelligence-alpaca-amharic", rx_files(r"^data/"), S_msgs2("conversations_amharic", positional=True), "amh", UN, "no licence stated (machine-translated)", synthetic=True, task="instruction", order=58)
    add("amharic_finetome", "addisai/FineTome-single-turn-dedup-amharic", rx_files(r"^data/"), S_msgs2("conversations_amharic", positional=True), "amh", CO, "apache-2.0 (card; machine-translated FineTome)", synthetic=True, task="instruction", order=59)
    add("amharic_sft_combined", "abdukuzi45/amharic-sft-combined-2026-with-cot", rx_files(r"^data/train-"), S_chatml("text"), "amh", CO, "apache-2.0 (card; machine-translated mix)", synthetic=True, task="instruction", order=60)
    add("tumbuka_instruction", "Mwanzau/Tumbuka_Instruction_Tuning", rx_files(r"^translated_tumbuka_dataset\.parquet$"), S_instr("instruction", "input", "output"), "tum", CO,
        "apache-2.0 (card); machine-translated from English (translation system not stated; Tumbuka is an NLLB-200 language: treat as possibly NLLB-derived)", synthetic=True, task="instruction",
        note="the 2.0M-row train.parquet of this repo is the ENGLISH original (SuperNatural-Instructions style) and is not used; only the 705k Tumbuka translations", order=61)
    add("tumbuka_language", "Mwanzau/Tumbuka_language", rx_files(r"^processed_gemma/train-"), lambda r, p: _gemma_text(r.get("text")), "tum", CO,
        "apache-2.0 (card); Gemma-chat-formatted Tumbuka tasks, provenance not stated (machine-translated / LLM-generated)", synthetic=True, task="instruction", order=62)
    add("amasquad_amh", "nebhailema/AmaSquad", rx_files(r"train\.json$"), _squad, "amh", CO, "apache-2.0 (card; SQuAD v2 machine-translated to Amharic)", synthetic=True, task="qa",
        rowfilter=lambda r, p: not r.get("is_impossible"), order=63)
    add("baoule_alpaca", "Tree-AI-lab/fr_bci_4d_instruct", rx_files(r"^data/train\.jsonl$"), S_instr("instruction", "input", "output"), "bci", CO, "apache-2.0 (catalogue)", synthetic=True, task="instruction",
        rowfilter=lambda r, p: r.get("direction") == "bci_bci", note="French Alpaca translated into Baoule (bci); only the bci->bci direction rows (instruction and answer both Baoule)", order=64)
    # LLM-generated multilingual sets
    for code in "amh hau ibo kin nya plt run sna sot tir xho yor zul swa arz".split():
        add("kakugo_" + code, "ptrdvn/kakugo-" + code, rx_files(r"^data/train-"), S_msgs2("messages", strip_think=True, drop_system=True), norm_iso(code), CO, "apache-2.0 (catalogue)", synthetic=True, task="chat",
            note="LLM-generated chats; <think> blocks removed", order=70)
    add("fikira", "vamboai/fikira", rx_files(r"\.jsonl$"), lambda r, p: [("user", clean(r.get("problem") or r.get("question") or r.get("prompt"))), ("assistant", clean(r.get("solution") or r.get("answer") or r.get("response")))],
        lambda p, r: norm_iso(r.get("language_code")), CO, "mit (card)", synthetic=True, task="qa", note="LLM-generated reasoning / maths QA in 10 languages", order=71)
    add("kiya_sft", "NaolBM/Kiya-SFT", rx_files(r"^data/train-"), S_msgs2("text"), col_iso("language"), CO, "mit (card; machine-translated Alpaca / FineTome-style mix)", synthetic=True, task="instruction",
        rowfilter=lambda r, p: r.get("language") in ("om", "yo", "sw", "am"), note="only the om / yo / sw / am rows (the rest is English)", order=71)
    add("bsc_m_personas_swh", "BSC-LT/m-personas", rx_files(r"^data/sw-"), S_msgs2("messages"), "swh", CO, "apache-2.0 (card)", synthetic=True, task="chat", order=72)


def _alpaca_text(t):
    m = re.search(r"### Instruction:\s*(.*?)\s*### Input:\s*(.*?)\s*### Response:\s*(.*)$", t or "", re.S)
    if not m:
        return None
    u = m.group(1).strip() + (("\n\n" + m.group(2).strip()) if m.group(2).strip() else "")
    return [("user", u), ("assistant", m.group(3).strip())]


def _gemma_text(t):
    out = []
    for role, body in re.findall(r"<start_of_turn>(user|model)\n(.*?)<end_of_turn>", t or "", re.S):
        out.append(("assistant" if role == "model" else "user", body))
    return out or None


def _squad(r, p):
    a = r.get("answers")
    txt = ""
    if isinstance(a, list) and a and isinstance(a[0], dict):
        txt = a[0].get("translated_text") or a[0].get("text") or ""
    elif isinstance(a, dict):
        t = a.get("text")
        txt = t[0] if isinstance(t, list) and t else (t or "")
    return [("user", (r.get("context") or "").strip() + "\n\n" + (r.get("question") or "").strip()), ("assistant", txt)]
