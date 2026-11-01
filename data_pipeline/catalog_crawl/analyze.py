"""Stage D: turn the cached crawl into per-language CSVs under african_hf/{pretrain,sft,rl} and a transparent _excluded/ folder.

Everything here is a HEURISTIC read of (card metadata, README text, column names/types, one sample row). The `analysis_method` and `confidence`
columns say so; low-confidence rows are the ones worth a human look.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import crawl_hf as C
from crawl_detail import candidates

ROOT = C.ROOT
PIVOT = {"en", "eng", "fr", "fra", "fre"}      # only English and French are kept next to African languages (owner, 2026-10-11)
CODE2ISO = {c: iso for c, (iso, _) in C.ALL.items() if iso not in C.NOT_AFRICAN}
ISO2NAME = {iso: n for _, (iso, n) in C.ALL.items()}
NAME2ISO = {}
for _iso, _n in ISO2NAME.items():
    NAME2ISO[_n.split(" (")[0].split(" ")[0].lower()] = _iso
ISO2NAME["twi"] = "Twi"
NAME2ISO.update({"naija": "pcm", "pidgin": "pcm", "kiswahili": "swh", "isizulu": "zul", "isixhosa": "xho", "twi": "twi", "akan": "twi",
                 "tigrigna": "tir", "afaan": "orm", "setswana": "tsn", "sesotho": "sot"})

# ---------------------------------------------------------------- column vocab
R = lambda *w: re.compile(r"^(?:%s)$" % "|".join(w), re.I)
TEXT = R(r"text", r"content", r"sentence", r"sentences", r"body", r"article", r"document", r"passage", r"paragraph", r"raw_?text", r"story",
         r"transcript(?:ion)?", r"utterance", r"review", r"tweet", r"comment", r"message", r"line", r"verse", r"normalized_?text", r"clean_?text",
         r"original_?text", r"full_?text", r"sentence_?text", r"lyrics", r"poem", r"news", r"description", r"caption", r"phrase", r"word", r"headline", r"title")
INSTR = R(r"instruction", r"instructions", r"prompt", r"query", r"user", r"human", r"input", r"inputs", r"task", r"request")
OUTP = R(r"output", r"outputs", r"response", r"responses", r"completion", r"reply", r"assistant", r"gpt", r"answer", r"answers", r"targets?")
QUEST = R(r"question", r"questions", r"query", r"q", r"prompt")
ANS = R(r"answer", r"answers", r"answer_?text", r"label_?answer", r"correct_?answer", r"gold", r"a")
CTX = R(r"context", r"contexts", r"passage", r"document", r"paragraph", r"evidence", r"article", r"background")
OPT = R(r"options?", r"choices?", r"candidates?", r"option_[a-d]", r"choice_[a-d]", r"mc\d?_targets?", r"A", r"B", r"C", r"D")
SUMM = R(r"summary", r"summaries", r"highlights?", r"abstract", r"tl;?dr", r"gist", r"short_?summary")
LABEL = R(r"label", r"labels", r"class", r"category", r"sentiment", r"emotion", r"topic", r"intent", r"polarity", r"hate", r"toxic(?:ity)?",
          r"offensive", r"target_?label", r"tag", r"stance", r"genre", r"domain", r"lang(?:uage)?_?id")
NER = R(r"ner_?tags?", r"ner", r"entities", r"entity", r"bio_?tags?", r"spans")
POS = R(r"pos_?tags?", r"upos", r"xpos", r"pos", r"tags?_?pos")
TOKENS = R(r"tokens", r"words")
AUDIO = R(r"audio", r"speech", r"wav", r"sound", r"audio_?path", r"path", r"file", r"audio_?filepath")
CHAT = R(r"messages", r"conversations?", r"dialog(?:ue)?s?", r"turns", r"chat", r"history", r"chat_?history")
PREF = R(r"chosen", r"rejected", r"preferred", r"dispreferred", r"better", r"worse", r"winner", r"loser", r"preference")
REWARD = R(r"score", r"rating", r"reward", r"quality", r"comet.*", r"chrf.*", r"bleu.*", r"confidence", r"helpfulness", r"correctness")
TOOLS = R(r"tools?", r"tool_?calls?", r"function_?calls?", r"functions?", r"tool_?use", r"api_?call")
REASON = R(r"reasoning", r"rationale", r"cot", r"chain_?of_?thought", r"thought", r"thinking", r"explanation", r"steps", r"solution")
URLC = R(r"url", r"link", r"source_?url", r"page_?url", r"wiki_?url", r"uri")
JSONC = R(r"json", r"schema", r"structured", r"extraction", r"fields?", r"attributes", r"key_?value")
SRC = R(r"source", r"src", r"src_?text", r"source_?text", r"source_?sentence", r"sentence_?src", r"sentence1", r"input_?text", r"text_?a", r"lang1")
TGT = R(r"target", r"tgt", r"tgt_?text", r"target_?text", r"target_?sentence", r"sentence_?tgt", r"sentence2", r"output_?text", r"text_?b", r"lang2",
        r"translation", r"translated", r"translated_?text", r"mt", r"reference")
LANGCOLS = {"english", "en", "eng", "french", "fr", "swahili", "sw", "swa", "yoruba", "yo", "yor", "hausa", "ha", "hau", "igbo", "ig", "ibo", "twi", "tw",
            "akan", "ak", "ewe", "ee", "zulu", "zu", "zul", "xhosa", "xh", "amharic", "am", "amh", "tigrinya", "ti", "tir", "somali", "so", "som",
            "kinyarwanda", "rw", "kin", "shona", "sn", "sna", "wolof", "wo", "wol", "pidgin", "pcm", "afrikaans", "af", "afr", "oromo", "om", "orm",
            "chichewa", "nya", "ny", "sesotho", "st", "sot", "setswana", "tn", "tsn", "lingala", "ln", "lin", "luganda", "lg", "lug", "fon", "bambara",
            "bm", "bam", "efik", "efi", "ibibio", "ibb", "urhobo", "urh", "arabic", "ar", "portuguese", "pt"}

TECH = re.compile(r"\b(medical|medicine|clinical|patients?|diseases?|biomedical|pubmed|pharma\w*|radiolog\w*|diagnos\w*|oncolog\w*|legal|legislation|"
                  r"statutes?|jurisprudence|finance|financial|banking|cyber\w*|malware|programming|github|sql|chemistry|genom\w*|protein)\b", re.I)
BENCH = re.compile(r"(bench(?:mark)?|leaderboard|(?:^|[-_/])(?:evals?|evaluation)(?:[-_/]|$)|flores|belebele|mmlu|mgsm|afrixnli|xnli|sib-?200|truthfulqa|"
                   r"hellaswag|winogrande|ifeval|irokobench|afrobench|tukabench|uhura|(?:^|[-_/])test(?:set|-set)?(?:[-_/]|$))", re.I)
BENCH_README = re.compile(r"\bbenchmark\b|evaluation (?:set|dataset|suite|benchmark)", re.I)    # NOT "leaderboard": it is a heading of the HF card template
SYNTH = re.compile(r"synthetic|llm[- ]generated|generated (?:by|using|with)|gpt-?[345o]|chatgpt|gemini|claude|llama|mistral|machine[- ]translat\w*|"
                   r"nllb|google translate|translated (?:by|using|with)|back-?translat\w*|auto(?:matically)? translat\w*", re.I)
PROV = [("bible", r"\bbible\b|scripture|jehovah|jw\.org|watchtower|awake!"), ("news", r"\bnews\b|bbc|voa\b|dw\b|headline"),
        ("wikipedia", r"wikipedia|wikimedia"), ("web crawl", r"common ?crawl|web ?crawl|scrap\w+|mc4|oscar|fineweb|hplt|glotcc"),
        ("social media", r"twitter|tweet|facebook|youtube|reddit"), ("crowdsourced", r"crowd\w*|volunteer|native speakers?|community"),
        ("government/parliament", r"parliament|hansard|government|constitution"), ("speech recordings", r"recorded|read[- ]aloud|speakers?|studio"),
        ("education", r"textbook|curricul\w*|school|exam|quiz")]


def lic_norm(meta: dict, readme: str):
    lic = meta.get("license")
    if isinstance(lic, list):
        lic = ", ".join(map(str, lic))
    src = "card"
    if not lic:
        m = re.search(r"licen[sc]e[^\n]{0,40}?:\s*([A-Za-z0-9\-\. ]{3,40})", readme or "", re.I)
        m2 = re.search(r"\b(CC[- ]BY(?:[- ](?:NC|SA|ND))*(?:[- ]\d\.\d)?|CC0|Apache[- ]2\.0|MIT|ODC-By|ODbL|CDLA[\w-]*|AGPL|GPL|BSD)\b", readme or "", re.I)
        lic = (m2.group(1) if m2 else (m.group(1).strip() if m else None))
        src = "readme" if lic else "none"
    if lic and meta.get("license_name") and lic.lower() == "other":
        lic = f"other ({meta['license_name']})"
    l = (lic or "").lower()
    if not l:
        cls = "unknown"
    elif re.search(r"(^|[-_ ])nc([-_ ]|$)|non-?commercial|research|noncommercial|cc-by-nd", l):
        cls = "non_commercial"
    elif re.search(r"\bsa\b|-sa-|share-?alike|odbl|gpl|gfdl", l):
        cls = "share_alike"
    elif re.search(r"apache|mit|bsd|cc0|cc-by|cc by|odc-by|cdla|afl|unlicense|isc|pddl|openrail|wtfpl|mpl", l):
        cls = "commercial_ok"
    else:
        cls = "unknown"
    return (lic or "unspecified"), src, cls


def feat_types(features) -> list:
    out = []
    for f in features or []:
        name, ft = f.get("name"), f.get("type") or {}
        if isinstance(ft, dict):
            if "dtype" in ft:
                t = ft["dtype"]
            elif ft.get("_type") == "Audio":
                t = "audio"
            elif ft.get("_type") == "Image":
                t = "image"
            elif ft.get("_type") == "ClassLabel":
                t = "class_label"
            elif ft.get("_type") == "Translation":
                t = "translation(" + ",".join(ft.get("languages", [])) + ")"
            elif ft.get("_type") == "Sequence" or isinstance(ft.get("feature"), (dict, list)):
                inner = ft.get("feature")
                t = "list<" + (inner.get("dtype", inner.get("_type", "struct")) if isinstance(inner, dict) else "struct") + ">"
            elif ft.get("_type") == "Value":
                t = ft.get("dtype", "value")
            else:
                t = ft.get("_type") or "struct"
        elif isinstance(ft, list):
            t = "list"
        else:
            t = str(ft)
        out.append((name, t))
    return out


def clean_readme(rd: str) -> str:
    rd = re.sub(r"^---.*?\n---\s*", "", rd or "", count=1, flags=re.S)
    rd = re.sub(r"<[^>]+>", " ", rd)
    rd = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", rd)
    return rd


def readme_summary(rd: str) -> str:
    rd = clean_readme(rd)
    paras = [p.strip() for p in re.split(r"\n\s*\n", rd) if p.strip()]
    best = []
    for p in paras:
        if p.startswith("#") or p.startswith("|") or p.startswith("```") or p.startswith("- "):
            continue
        if len(p) >= 50:
            best.append(re.sub(r"\s+", " ", p))
        if len(best) >= 2:
            break
    return " ".join(best)[:420]


def codes_of(meta: dict) -> set:
    codes = set()
    for x in meta.get("tags") or []:
        if x.startswith("language:"):
            codes.add(x.split(":", 1)[1])
    for x in meta.get("language") or []:
        codes.add(str(x))
    return codes


def frontmatter_features(readme: str):
    """Columns/rows declared in the card's YAML `dataset_info` block (works for script-based or viewer-less repos)."""
    m = re.match(r"^---\s*\n(.*?)\n---", readme or "", re.S)
    if not m:
        return [], None
    try:
        import yaml
        y = yaml.safe_load(m.group(1)) or {}
    except Exception:
        return [], None
    info = y.get("dataset_info")
    infos = info if isinstance(info, list) else ([info] if isinstance(info, dict) else [])
    feats, rows = [], 0
    for inf in infos:
        if not isinstance(inf, dict):
            continue
        for f in inf.get("features") or []:
            if not isinstance(f, dict) or "name" not in f:
                continue
            if "dtype" in f:
                ty = str(f["dtype"])
            elif "class_label" in f:
                ty = "class_label"
            elif "list" in f:
                ty = "list"
            elif "struct" in f:
                ty = "struct"
            else:
                ty = "unknown"
            if f["name"] not in [n for n, _ in feats]:
                feats.append((f["name"], ty))
        for s in inf.get("splits") or []:
            if isinstance(s, dict) and isinstance(s.get("num_examples"), int):
                rows += s["num_examples"]
    return feats, (rows or None)


EN_STOP = set("the of and to in is that for with as on it was are be this by from or an at have not but they his her which you were has had their".split())


def english_ratio(text: str) -> float:
    w = re.findall(r"[a-z']+", (text or "").lower())
    return sum(x in EN_STOP for x in w) / len(w) if len(w) >= 8 else 0.0


def analyse(rid: str, meta: dict, det: dict) -> dict:
    readme = det.get("readme") or ""
    feats = feat_types(det.get("features"))
    from_fm = False
    if not feats:
        feats, fm_rows = frontmatter_features(readme)
        if feats:
            from_fm = True
            det = {**det, "num_rows": det.get("num_rows") or fm_rows}
    names = [n for n, _ in feats]
    ltypes = {n.lower(): t for n, t in feats}
    sample = det.get("sample") or {}
    low = [n.lower() for n in names]
    blob_meta = " ".join([rid, meta.get("pretty_name") or "", " ".join(meta.get("task_categories") or []), " ".join(meta.get("task_ids") or []),
                          " ".join(t for t in meta.get("tags") or [] if not t.startswith(("language:", "region:", "size_categories:", "license:", "format:", "modality:", "library:", "arxiv:"))),
                          meta.get("description") or ""])
    tcat = {t.lower() for t in (meta.get("task_categories") or [])}
    tids = " ".join(meta.get("task_ids") or []).lower()
    name_l = rid.lower()

    def has(rx):
        return [n for n in names if rx.match(n)]

    # ---- languages
    codes = codes_of(meta)
    african = {CODE2ISO[c] for c in codes if c in CODE2ISO and c not in C.NOT_AFRICAN}
    others = {c for c in codes if c not in CODE2ISO and c not in PIVOT and not re.match(r"^(multilingual|mul|und|zxx|code)$", c)}
    if african & {"ary", "arz", "aeb", "apd"}:      # an Arabic-dialect dataset often also carries the generic `ar` tag: not a foreign language there
        others -= {"ar", "ara", "arb", "apc", "ayl", "ars", "acm"}
    name_inferred = False
    if not african:
        for tok in re.findall(r"[a-z]+", name_l.split("/", 1)[-1]):
            if tok in NAME2ISO:
                african.add(NAME2ISO[tok]); name_inferred = True
    # configs named after a language also reveal it
    for cfg in (det.get("configs_rows") or {}):
        c = str(cfg).split("-")[0].split("_")[0].lower()
        if c in CODE2ISO and CODE2ISO[c] not in african and len(det.get("configs_rows") or {}) > 1:
            african.add(CODE2ISO[c])
    if len(codes) > 25 or any(t in ("multilingual",) for t in (meta.get("multilinguality") or [])) and len(others) > 3:
        scope = "multilingual_mixed"
    elif others:
        scope = "multilingual_mixed"
    elif african and (codes & {"en", "eng"} or any(c in PIVOT for c in codes)):
        scope = "african_plus_pivot"
    elif african:
        scope = "african_only"
    else:
        scope = "non_african"

    # ---- roles
    r = {k: has(v) for k, v in dict(text=TEXT, instr=INSTR, outp=OUTP, quest=QUEST, ans=ANS, ctx=CTX, opt=OPT, summ=SUMM, label=LABEL, ner=NER, pos=POS,
                                       tokens=TOKENS, audio=AUDIO, chat=CHAT, pref=PREF, reward=REWARD, tools=TOOLS, reason=REASON, url=URLC,
                                       jsonc=JSONC, src=SRC, tgt=TGT).items()}
    audio_cols = [n for n, t in feats if t == "audio"] or [n for n in r["audio"] if re.match(r"audio|speech", n, re.I)]
    lang_pair_cols = [n for n in names if n.lower() in LANGCOLS]
    trans_col = [n for n, t in feats if t.startswith("translation(")] or [n for n in names if n.lower() == "translation" or n.lower() == "translations"]
    if not trans_col and isinstance(sample, dict):
        trans_col = [k for k, v in sample.items() if isinstance(v, dict) and len(v) >= 2 and all(isinstance(x, str) for x in v.values()) and
                     any(str(kk).lower().split("_")[0].split("-")[0] in LANGCOLS or str(kk).lower() in LANGCOLS for kk in v)]
    parallel = bool(trans_col or len(lang_pair_cols) >= 2 or (r["src"] and r["tgt"]) or "translation" in tcat or "translat" in name_l or "parallel" in name_l or
                    re.search(r"(?:^|[-_/])(?:en|eng|english)[-_](?:[a-z]{2,3}|twi|yoruba|hausa|igbo)", name_l) or "sentence-pairs" in name_l or "sentence_pairs" in name_l or "machine-translation" in tids)

    # ---- tasks and column usage
    tasks, use, rl_use = [], [], []
    pre_cols = []

    def add(t):
        if t not in tasks:
            tasks.append(t)

    text_like = [n for n in names if (TEXT.match(n) or ltypes.get(n.lower()) == "string") and n not in audio_cols and not URLC.match(n)
                 and not re.match(r"^(id|idx|index|uid|uuid|key|split|lang|language|source|dataset|license|domain|category|label|tag|type|name)$", n, re.I)]
    if parallel:
        add("translation (sft)")
        if trans_col:
            use.append(f"Translation SFT: unpack dict column `{trans_col[0]}` into (source language -> target language) pairs, both directions; render as <translate> instruction rows.")
        elif len(lang_pair_cols) >= 2:
            use.append(f"Translation SFT: language-named columns {lang_pair_cols[:4]} are the aligned sides; emit each ordered pair as an instruction/response row.")
        elif r["src"] and r["tgt"]:
            use.append(f"Translation SFT: `{r['src'][0]}` is the source side and `{r['tgt'][0]}` the target; use both directions when both are in supported languages.")
        else:
            use.append("Translation SFT: aligned sentence pairs (columns not self-describing; map by hand).")
        rl_use.append("RL translation: prompts = source side; reward by AfriCOMET-QE (reference-free) or chrF++ against the target side; QE scores can filter noisy pairs first.")
        if len(african) >= 2:
            add("inter-african language translation")
        pre_cols += trans_col or lang_pair_cols[:3] or (r["src"][:1] + r["tgt"][:1])
    if "summarization" in tcat or r["summ"]:
        body = r["text"][:1] or r["ctx"][:1]
        add("summarization")
        use.append(f"Summarization SFT: `{body[0] if body else 'article column'}` -> `{r['summ'][0] if r['summ'] else 'summary'}`; also the reverse (summary -> article/continuation) and headline generation when a title exists.")
        pre_cols += body + r["summ"][:1]
    if re.search(r"headline|title", " ".join(low)) and (r["text"] and len([n for n in low if re.search(r"headline|title", n)])):
        t_col = [n for n in names if re.search(r"headline|title", n, re.I)][:1]
        b_col = [n for n in r["text"] if n not in t_col][:1]
        if t_col and b_col:
            add("headline/title generation"); use.append(f"Headline SFT: `{b_col[0]}` -> `{t_col[0]}`; reverse gives article-from-title generation.")
    cls_col = [n for n in r["label"] if n.lower() not in ("tag",)] or ([n for n, t in feats if t == "class_label"])
    cls_name = " ".join(low + [name_l, tids, " ".join(tcat)])
    if cls_col or "text-classification" in tcat or "sentiment" in cls_name:
        kinds = []
        for kw, nm in [("sentiment", "sentiment classification"), ("emotion", "emotion classification"), ("topic", "topic classification"),
                       ("intent", "intent classification"), ("hate", "hate-speech classification"), ("toxic", "toxicity classification"),
                       ("offens", "toxicity classification"), ("news", "topic classification"), ("polarity", "sentiment classification"),
                       ("langid", "language identification"), ("lang_id", "language identification"), ("language-id", "language identification")]:
            if kw in cls_name and nm not in kinds:
                kinds.append(nm)
        if not kinds and (cls_col or "text-classification" in tcat):
            kinds = ["text classification"]
        if kinds and r["text"] + text_like:
            txt = (r["text"] or text_like)[0]
            for k in kinds:
                add(k)
            use.append(f"Classification SFT ({', '.join(kinds)}): input `{txt}` -> label `{(cls_col or ['label'])[0]}`; render with <classify><label-type> plan tokens, class names as words (not indices).")
            rl_use.append(f"RL verifiable: reward 1 if predicted label equals `{(cls_col or ['label'])[0]}` (exact-match gold).")
            pre_cols += [txt]
    if r["ner"] or ("token-classification" in tcat and "ner" in tids + name_l):
        add("ner"); use.append(f"NER SFT: tokens `{(r['tokens'] or ['tokens'])[0]}` + tags `{(r['ner'] or ['ner_tags'])[0]}` -> entity list per type; detokenised sentence also usable as pretraining text.")
        rl_use.append("RL verifiable: entity-set F1 against gold tags.")
        pre_cols += r["tokens"][:1]
    if r["pos"] or "part-of-speech" in tids or "pos" in name_l.split("/")[-1].split("-"):
        add("pos tagging"); use.append(f"POS SFT: `{(r['tokens'] or ['tokens'])[0]}` -> `{(r['pos'] or ['pos_tags'])[0]}` word/tag pairs; tokens joined are plain text for pretraining.")
        pre_cols += r["tokens"][:1]
    qa_like = bool(r["quest"] and (r["ans"] or r["outp"])) or "question-answering" in tcat or "multiple-choice" in tcat
    if qa_like:
        add("question-answering")
        q = (r["quest"] or ["question"])[0]
        a = (r["ans"] or r["outp"] or ["answer"])[0]
        if r["ctx"]:
            add("context-based question answering"); add("question generation"); add("document analysis")
            use.append(f"Context QA SFT: `{r['ctx'][0]}` + `{q}` -> `{a}`; question generation: `{r['ctx'][0]}` (+ answer) -> `{q}`.")
            use.append(f"RAG: expose `{r['ctx'][0]}` as the result of a retrieve() tool call over a knowledge base and `{q}` as the tool query; train answer-from-tool-result turns" +
                       (f"; `{r['url'][0]}` can be the argument of a fetch_page() call." if r["url"] else "."))
            add("rag")
            pre_cols += r["ctx"][:1]
        else:
            use.append(f"Closed-book QA SFT: `{q}` -> `{a}`.")
        if r["opt"] or "multiple-choice" in tcat or "mmlu" in name_l or "belebele" in name_l:
            add("multiple-choice qa"); use.append("MCQ: render options as lettered choices; answer is the letter (verifiable).")
            rl_use.append(f"RL verifiable: exact match of the chosen option against `{a}` (letter/index) -> reward 0/1.")
        else:
            rl_use.append(f"RL verifiable (short answers): normalised exact match / token-F1 of the generated answer against `{a}`.")
        pre_cols += [q] if q in text_like else []
    elif r["ctx"] and r["url"]:
        add("rag"); use.append(f"RAG: `{r['url'][0]}` as fetch target, `{r['ctx'][0]}` as retrieved context.")
    if re.search(r"gsm|math|arithmetic|word-?problem|mgsm|numina|aime", name_l):
        add("basic math"); add("reasoning")
        rl_use.append("RL verifiable: final numeric answer exact-match (extract the last number).")
    if r["reason"] or re.search(r"reasoning|cot|chain-of-thought|r1", name_l):
        add("reasoning"); use.append(f"Reasoning SFT: put `{(r['reason'] or ['reasoning'])[0]}` inside <think> before the final answer.")
    if r["tools"] or "function-calling" in name_l or "tool" in tids:
        add("function calling"); use.append(f"Function calling SFT: `{(r['tools'] or ['tools'])[0]}` becomes the <tools> block; call/arguments become <tool_call> JSON turns.")
    if r["chat"] or "conversational" in tcat or re.search(r"chat|dialog|conversation|roleplay|persona", name_l):
        add("chat")
        use.append(f"Chat SFT: `{(r['chat'] or ['messages'])[0]}` roles map directly to <|user|>/<|assistant|> turns; join all turns as dialogue text for pretraining.")
        pre_cols += r["chat"][:1]
    if (r["instr"] and r["outp"]) or "text2text-generation" in tcat or "instruction" in name_l or "alpaca" in name_l:
        if not (parallel and not r["instr"][:1] == ["instruction"]):
            add("instruction-following")
            i = (r["instr"] or ["instruction"])[0]
            o = (r["outp"] or ["output"])[0]
            use.append(f"Instruction SFT: `{i}` (+ optional `input`) -> `{o}`; both columns are also free-text pretraining content.")
            pre_cols += [i, o]
    if r["pref"] or (r["reward"] and re.search(r"preference|dpo|rlhf|reward|ranking|feedback", name_l + " " + " ".join(low))):
        add("preference / rlhf")
        rl_use.append(f"RL / preference: `{(r['pref'] or ['chosen'])[0]}` vs `{(r['pref'][1:] or ['rejected'])[0]}` pairs for DPO or reward-model training; scalar columns {r['reward'][:3]} can weight or filter.")
    if r["jsonc"] or "json" in name_l or "structured" in name_l or "extraction" in name_l:
        add("structured output"); use.append(f"Structured output SFT: target `{(r['jsonc'] or ['json column'])[0]}` is a JSON object to be emitted from the text input.")
    if audio_cols:
        trn = [n for n in names if re.match(r"^(transcript(ion)?|sentence|text|normalized_?text|raw_?text|utterance|label|phonetic_?text)$", n, re.I)]
        add("speech-to-text (asr)"); add("tts")
        use.append(f"ASR SFT/eval: audio `{audio_cols[0]}` -> `{(trn or ['transcript'])[0]}`; TTS: reverse direction (text -> audio tokens); transcripts alone are pretraining text.")
        pre_cols += trn[:1]
        if re.search(r"translat|st-|speech-translation|covost|fleurs", name_l):
            add("speech translation")
    if "fill-mask" in tcat or "text-generation" in tcat or "language-modeling" in tids or "language-modelling" in tids or "monolingual" in " ".join(meta.get("multilinguality") or []):
        add("pretrain (monolingual text)")
    if "image" in " ".join(t for _, t in feats) and not audio_cols:
        add("vision-language (images)")
    # free-text columns -> pretraining
    long_text = [n for n in text_like if isinstance(sample.get(n), str) and len(sample.get(n)) >= 30] if sample else text_like
    pre_cols = [c for i, c in enumerate(dict.fromkeys(pre_cols + long_text)) if c in names]
    if pre_cols or not feats:
        add("pretrain (text)") if "pretrain (monolingual text)" not in tasks else None
    if not feats:
        # no viewer data and no card schema: infer from the repo name and the README prose, and say so
        hay = (name_l + " " + clean_readme(readme)[:3000]).lower()
        guess = []
        if re.search(r"sentence[-_ ]pairs|parallel|translat|bitext|paires|\bmt\b", hay):
            guess.append("translation (sft)")
            use.append("Translation SFT (inferred from the name/README only; columns unknown): treat the two aligned text fields as source/target.")
            rl_use.append("RL translation (inferred): prompts = source side, reward = AfriCOMET-QE / chrF++.")
        if re.search(r"speech|asr|audio|voice|tts|recordings?|wav", hay):
            guess += ["speech-to-text (asr)", "tts"]
            use.append("ASR/TTS (inferred from the name/README only): audio + transcript; transcripts are pretraining text.")
        if re.search(r"instruct|alpaca|chat|dialog|conversation|roleplay", hay):
            guess.append("instruction-following")
            use.append("Instruction/chat SFT (inferred from the name/README only).")
        if re.search(r"question|qa\b|mmlu|quiz", hay):
            guess.append("question-answering")
        if re.search(r"sentiment|emotion|classif|hate|topic", hay):
            guess.append("text classification")
        tasks.extend(guess + ["pretrain (text, assumed)", "unknown columns (viewer unavailable: gated, script-based or empty)"])
        pre_cols = pre_cols or ["(unknown: inspect the files)"]

    # ---- does the sample look English although the card claims an African language? (mis-tagged repos)
    main_txt = max([v for v in (sample or {}).values() if isinstance(v, str)] or [""], key=len)
    en_sus = (english_ratio(main_txt) > 0.28 and not parallel and not re.search(r"english|eng|en[-_]", name_l)
              and not set(african) <= {"pcm", "gpe", "wes", "kri"})           # Pidgin and Krio really are close to English
    no_text = bool(feats) and not from_fm and not audio_cols and not any("image" in ty for _, ty in feats) and bool(sample) and \
        not any(isinstance(v, (str, list, dict)) and len(str(v)) >= 25 for v in sample.values()) and not parallel and \
        not any(TEXT.match(n) or SRC.match(n) or TGT.match(n) or n.lower() in LANGCOLS for n in names)
    # ---- useful columns
    useful = []
    for k in ("text", "src", "tgt", "instr", "outp", "quest", "ans", "ctx", "summ", "label", "ner", "tokens", "pos", "chat", "pref", "reason", "tools", "url"):
        for n in r[k][:2]:
            if n not in useful:
                useful.append(n)
    for n in trans_col + lang_pair_cols[:3] + audio_cols[:1]:
        if n not in useful:
            useful.append(n)
    if not useful:
        useful = text_like[:3]

    # ---- flags
    tech = bool(TECH.search(blob_meta)) or bool(re.search(r"(^|[-_/])(code|coding|programming|leetcode|humaneval)([-_/]|$)", name_l))
    grounded = any(x in tasks for x in ("context-based question answering", "rag", "summarization", "document analysis", "structured output", "question generation"))
    bench = bool(BENCH.search(rid + " " + (meta.get("pretty_name") or ""))) or bool(BENCH_README.search(clean_readme(readme)[:700]))
    synth = sorted({m.group(0).lower() for m in SYNTH.finditer(clean_readme(readme)[:6000] + " " + (meta.get("description") or ""))})[:6]
    prov = [nm for nm, rx in PROV if re.search(rx, clean_readme(readme)[:8000] + " " + name_l, re.I)]
    lic, lic_src, lic_cls = lic_norm(meta, readme)
    # ---- row count per language where the config says so
    rows_total = det.get("num_rows")
    cfgs = det.get("configs_rows") or {}
    per_lang_rows = {}
    for cfg, n in cfgs.items():
        c = re.split(r"[-_.]", str(cfg).lower())
        for tok in c:
            if tok in CODE2ISO:
                per_lang_rows.setdefault(CODE2ISO[tok], 0)
                per_lang_rows[CODE2ISO[tok]] += n or 0
    extraction = {}
    if scope == "multilingual_mixed":
        lang_col = [n for n in names if re.match(r"^(lang|language|lang_?code|language_?code|iso_?code|src_?lang|tgt_?lang|locale)$", n, re.I)]
        cfg_by_iso = defaultdict(list)
        for cfg in cfgs:
            for tok in re.split(r"[-_.]", str(cfg).lower()):
                if tok in CODE2ISO:
                    cfg_by_iso[CODE2ISO[tok]].append(str(cfg))
        if cfg_by_iso:
            scope = "mixed_subset"
            african = set(cfg_by_iso)
            extraction = {iso: "load only config(s): " + ", ".join(sorted(set(v))[:6]) + (" ..." if len(set(v)) > 6 else "") for iso, v in cfg_by_iso.items()}
        elif lang_col and african:
            scope = "mixed_subset"
            extraction = {iso: f"filter rows where `{lang_col[0]}` is this language (code per card)" for iso in african}
    bucket = {"pretrain": bool(pre_cols) and (scope != "non_african"),
              "sft": any(t not in ("pretrain (text)", "pretrain (monolingual text)", "pretrain (text, assumed)") and not t.startswith("unknown") for t in tasks),
              "rl": bool(rl_use)}
    # english-only dataset (no African language) is dropped by scope == non_african
    confidence = "high" if (feats and sample and (meta.get("task_categories") or tasks)) else ("medium" if feats else "low")
    if from_fm:
        confidence = "medium"
    if en_sus:
        confidence = "low"
    return dict(
        dataset_id=rid, url="https://huggingface.co/datasets/" + rid, author=meta.get("author") or rid.split("/")[0],
        scope=scope, african_iso=sorted(african), extraction=extraction, other_languages=sorted(others), name_inferred=name_inferred,
        num_rows_total=rows_total, per_lang_rows=per_lang_rows, configs=len(cfgs) or None,
        columns="; ".join(f"{n}:{t}" for n, t in feats) if feats else "",
        sample_row=json.dumps({k: (v if not isinstance(v, (str, bytes)) else str(v)[:120]) for k, v in list(sample.items())[:8]}, ensure_ascii=False, default=str)[:600] if sample else "",
        useful_columns="; ".join(useful), tasks="; ".join(tasks), pretrain_columns="; ".join(pre_cols),
        license=lic, license_source=lic_src, commercial_use=lic_cls,
        gated=meta.get("gated"), downloads=meta.get("downloads"), likes=meta.get("likes"),
        created=(meta.get("createdAt") or "")[:10], last_modified=(meta.get("lastModified") or "")[:10],
        readme_summary=readme_summary(readme), provenance_hints="; ".join(prov), synthetic_or_mt_hints="; ".join(synth),
        domain_flag="technical/domain-specific" if (tech and not grounded) else "", benchmark=bench,
        domain_note=("domain-specific topic, but answers are grounded in supplied context (usable for RAG/doc-QA/summarisation)" if (tech and grounded) else ""),
        no_text=no_text,
        placeholder=bool(re.search(r"ce r[ée]pertoire est vide|this (?:repository|repo) is empty|cr[ée][ée] pour am[ée]liorer le r[ée]f[ée]rencement", readme, re.I)), usage="  |  ".join(use) if use else ("Plain-text pretraining: use " + ", ".join(pre_cols[:3]) + "." if pre_cols else ""),
        rl_usage="  |  ".join(rl_use), bucket=bucket, confidence=confidence,
        analysis_method=("heuristic (card tags + column names/types from the card's dataset_info; no viewer sample)" if from_fm else
                         "heuristic (card tags + column names/types + 1 sample + README keywords)") if feats else "heuristic (card + README only; no viewer data)",
        review_note=("sample text looks English although the card lists African languages: possible mis-tag, check before use" if en_sus else ""))


SERVE = {"pcm", "hau", "ibo", "yor", "efi", "ibb", "urh", "twi", "ewe", "fon", "kin", "sna", "zul", "swh", "orm", "wol", "xho", "tsn", "nya", "afr", "som", "sot"}   # languages the model serves (English implied)
FIELDS = ["dataset_id", "url", "language_iso3", "language", "scope", "all_african_languages", "other_languages", "num_rows", "num_rows_basis",
          "columns", "useful_columns", "tasks", "pretrain_columns", "usage_explanation", "license", "commercial_use", "license_source",
          "readme_summary", "provenance_hints", "synthetic_or_mt_hints", "gated", "downloads", "likes", "created", "last_modified", "author",
          "confidence", "review_note", "extraction_hint", "domain_note", "analysis_method", "sample_row"]


def row_for(a: dict, iso: str, kind: str) -> dict:
    pl = a["per_lang_rows"].get(iso)
    basis = "config for this language" if pl is not None else ("whole dataset (single language)" if len(a["african_iso"]) <= 1 else "whole dataset (all languages combined)")
    use = a["usage"]
    if kind == "rl":
        use = a["rl_usage"]
    return dict(dataset_id=a["dataset_id"], url=a["url"], language_iso3=iso, language=C.ALL.get(iso, (iso, ISO2NAME.get(iso, iso)))[1] if iso in C.ALL else ISO2NAME.get(iso, iso),
                scope=a["scope"], all_african_languages=",".join(a["african_iso"]), other_languages=",".join(a["other_languages"]),
                num_rows=pl if pl is not None else a["num_rows_total"], num_rows_basis=basis, columns=a["columns"], useful_columns=a["useful_columns"],
                tasks=a["tasks"], pretrain_columns=a["pretrain_columns"], usage_explanation=use, license=a["license"], commercial_use=a["commercial_use"],
                license_source=a["license_source"], readme_summary=a["readme_summary"], provenance_hints=a["provenance_hints"],
                synthetic_or_mt_hints=a["synthetic_or_mt_hints"], gated=a["gated"], downloads=a["downloads"], likes=a["likes"], created=a["created"],
                last_modified=a["last_modified"], author=a["author"], confidence=a["confidence"], review_note=a["review_note"], extraction_hint=a["extraction"].get(iso, ""), domain_note=a["domain_note"], analysis_method=a["analysis_method"], sample_row=a["sample_row"])


def main():
    cand = candidates()
    det_dir = C.RAW / "C_detail"
    recs, missing = [], 0
    for rid, meta in cand.items():
        f = det_dir / (rid.replace("/", "__") + ".json")
        if not f.exists():
            missing += 1
            continue
        det = json.loads(f.read_text())
        a = analyse(rid, meta, det)
        recs.append(a)
    print(f"analysed {len(recs)}; waiting on {missing}")
    for d in ("pretrain", "sft", "rl", "_excluded"):
        for old in (ROOT / d).glob("*.csv"):
            old.unlink()
    buckets = {"pretrain": defaultdict(list), "sft": defaultdict(list), "rl": defaultdict(list)}
    excluded = []
    for a in recs:
        reason = None
        if a["benchmark"]:
            reason = "benchmark / evaluation / test set (owner: no benchmarks)"
        elif a["author"].lower() == "electricsheepafrica" or re.search(r"jailbreak", a["dataset_id"], re.I):
            reason = "mislabelled: country-statistics tables / jailbreak-prompt forks carrying African language tags"
        elif a["placeholder"]:
            reason = "empty placeholder repo (SEO mirror that only points to the original dataset)"
        elif a["scope"] == "non_african":
            reason = "no African language (english-only / not African)"
        elif a["scope"] == "multilingual_mixed":
            reason = "mixed-scope collection; African part cannot be isolated (no per-language config or language column)"
        elif a["domain_flag"]:
            reason = "technical / domain-specific"
        elif a["no_text"]:
            reason = "no free-text content (tabular / numeric)"
        if not reason and not any(a["bucket"].values()):
            reason = "no pretrain / sft / rl use detected"
        if reason:
            excluded.append({"reason": reason, **{k: a[k] for k in ("dataset_id", "url", "scope", "african_iso", "other_languages", "num_rows_total", "tasks", "license", "commercial_use", "downloads", "readme_summary")}})
            continue
        for iso in a["african_iso"]:
            for kind in ("pretrain", "sft", "rl"):
                if a["bucket"][kind]:
                    buckets[kind][iso].append(row_for(a, iso, kind))
    manifest = []
    for kind, per in buckets.items():
        for iso, rows in sorted(per.items()):
            seen, uniq = set(), []
            for r in rows:
                if r["dataset_id"] not in seen:
                    seen.add(r["dataset_id"]); uniq.append(r)
            uniq.sort(key=lambda r: -(r["downloads"] or 0))
            nm = re.sub(r"[^a-z0-9]+", "-", ISO2NAME.get(iso, iso).split(" (")[0].lower()).strip("-")
            path = ROOT / kind / f"{iso}_{nm}.csv"
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(uniq)
            manifest.append({"bucket": kind, "language_iso3": iso, "language": ISO2NAME.get(iso, iso), "serve_priority": "yes" if iso in SERVE else "", "datasets": len(uniq), "file": str(path.relative_to(ROOT))})
    with (ROOT / "MANIFEST.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bucket", "language_iso3", "language", "serve_priority", "datasets", "file"])
        w.writeheader(); w.writerows(manifest)
    if excluded:
        with (ROOT / "_excluded" / "excluded_datasets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(excluded[0]))
            w.writeheader(); w.writerows(excluded)
    print({k: sum(len(v) for v in per.values()) for k, per in buckets.items()}, "excluded:", len(excluded), "files:", len(manifest))


if __name__ == "__main__":
    main()
