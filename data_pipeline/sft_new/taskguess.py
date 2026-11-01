"""Task of an instruction-following row, guessed from keywords -> (task name, task_plan verbs).

Only used where a source has NO task column. The rules are deliberately conservative and ordered from the most specific to the most general; anything not
recognised is `<|chat|>`. Instructions are matched in English and in the languages of our template pools (pcm, yor, hau, ibo, a few words of twi/ewe).
Domain flags (<|is_medical|> ...) are added only when a domain word is present in the REQUEST (never from the answer).
"""

from __future__ import annotations

import re

_R = lambda p: re.compile(p, re.I)

RULES = [
    ("translation", ["<translate>"], _R(r"\btranslat(e|ed|ion|ing)\b|\bfassara\b|\bsụgharịa\b|\btúmọ̀\b|\bkyerɛ\s+\w+\s+ase\b|\bɖe\s+\w+\s+gɔme\b|\binto (english|yoruba|hausa|igbo|swahili|french|pidgin|zulu|xhosa|twi|ewe)\b")),
    ("summarization", ["<summarize>"], _R(r"\bsummar(y|ize|ise|ised|ized)\b|\btl;?dr\b|\bin (one|two|three|a few) sentences?\b|\btakaita\b|\bchịkọta\b|\bàkótán\b|\bàkópọ̀\b|\bshort version\b|\bcondense\b")),
    ("headline", ["<summarize>"], _R(r"\bheadline\b|\btitle for (this|the)\b|\bgive (it|this) a title\b|\bsuggest a title\b")),
    ("sentiment", ["<classify>"], _R(r"\bsentiment\b|\bpositive,? negative\b|\bis (this|the) (review|text|sentence|tweet) positive\b")),
    ("ner", ["<|data_extract|>"], _R(r"\bnamed entit(y|ies)\b|\bextract (all )?(the )?(names|entities|dates|places|people|organi[sz]ations)\b|\blist (all )?the (names|entities)\b")),
    ("classification", ["<classify>"], _R(r"\bclassif(y|ication)\b|\bcategori[sz]e\b|\bwhich category\b|\bidentify the (topic|genre|intent|language)\b|\bwhat language is\b")),
    ("correction", ["<clean>", "<correct>"], _R(r"\bcorrect (the )?(grammar|spelling|errors|mistakes)\b|\bproofread\b|\bfix (the )?(grammar|spelling|typos|errors)\b|\bgrammar(atical)? errors?\b|\bspell(ing)? check\b")),
    ("editing", ["<|edit|>"], _R(r"\brewrite\b|\brephrase\b|\bparaphrase\b|\bedit (the|this)\b|\bmake (it|this) (more )?(formal|concise|polite|shorter|simpler)\b|\bimprove (the|this) (sentence|text|paragraph)\b")),
    ("code", ["<|is_technical|>", "<|code|>"], _R(r"```|\bwrite (a |an |the )?(python|java|javascript|c\+\+|sql|html|css|bash|regex|program|script|function|code)\b|\bdef \w+\(|\bpython\b|\bjavascript\b|\bsql query\b|\balgorithm\b|\bpseudocode\b|\bregular expression\b|\bcode snippet\b")),
    ("json", ["<|is_technical|>", "<|JSON|>"], _R(r"\bjson\b|\bin (xml|yaml|csv) format\b")),
    ("math", ["<|is_technical|>", "<|math|>"], _R(r"\bcalculate\b|\bsolve\b|\bequation\b|\bhow many\b.*\d|\bwhat is \d[\d\s+\-*/^().x×÷%]*\??|\bsum of\b|\bpercentage\b|\bprobability\b|\barea of\b|\bderivative\b|\bintegral\b")),
    ("nli", ["<nli>"], _R(r"\bcan we conclude\b|\bdoes the (premise|first sentence) (entail|imply)\b|\b(premise|hypothesis)\b.*\b(entail|contradict|neutral)|\bnatural language inference\b|\bis the (hypothesis|statement) (true|entailed)\b|\bentailment\b")),
    ("multiple_choice", ["<qa>"], _R(r"\bmultiple[- ]choice\b|\bchoose the (correct|best|right)\b|\bwhich of the following\b|\boptions?:\s*\n?\s*[-(]?\s*[Aa1][\).:]|\bselect the (correct|best)\b|\bchoose the most suitable\b|\boptions?:\s*\n\s*[-*]\s|\bpick the (correct|best|right)\b")),
    ("generation", ["<|generate|>"], _R(r"\bwrite (a|an|the|me|some)\b.{0,30}\b(story|poem|essay|song|letter|email|article|speech|script|dialogue|paragraph|slogan|joke|blog|description|review|caption|tweet|haiku|limerick|lyrics)\b|\bcompose\b|\bcreate (a|an)\b.{0,25}\b(story|poem|slogan|name|title|tagline|dialogue)\b|\bcome up with\b|\bbrainstorm\b|\bgenerate (a|an|some)\b")),
    ("planning", ["<|plan|>"], _R(r"\bplan (a|an|the|my|for)\b|\bitinerary\b|\bschedule\b|\bstep[- ]by[- ]step (plan|guide)\b|\bhow (can|do|should) i\b.*\b(start|prepare|organi[sz]e|achieve)\b|\broadmap\b|\boutline\b")),
    ("recommendation", ["<|recommend|>"], _R(r"\brecommend\b|\bsuggest(ions)?\b|\badvice\b|\btips? (for|on|to)\b|\bideas? for\b|\bwhat should i\b|\bbest way to\b")),
    ("analysis", ["<|analyze|>"], _R(r"\banaly[sz]e\b|\bevaluate\b|\bcompare\b|\bcontrast\b|\bassess\b|\bcritique\b|\bpros and cons\b|\bdifference between\b|\bimplications?\b")),
    ("explanation", ["<|explain|>"], _R(r"\bexplain\b|\bwhy\b|\bhow does\b|\bhow do\b|\bwhat (is|are|was|were) (the )?(meaning|purpose|reason|cause|difference)\b|\bdescribe\b|\bdefine\b|\bwhat does .* mean\b|\bṣàlàyé\b|\bbayyana\b|\bkọwaa\b|\bexplain am\b")),
    ("qa", ["<qa>"], _R(r"^\s*(who|what|when|where|which|whose|how (many|much|long|old|far))\b.*\?\s*$")),
]

DOMAIN = [
    ("<|is_medical|>", _R(r"\b(symptom|diagnos|disease|medicine|medical|patient|doctor|hospital|surgery|therapy|dose|drug|vaccine|infection|cancer|diabetes|malaria|pregnan|nutrition|health)\w*")),
    ("<|is_legal|>", _R(r"\b(law|legal|court|lawsuit|contract|attorney|lawyer|constitution|statute|liability|copyright|patent|tenant|verdict)\w*")),
    ("<|is_financial|>", _R(r"\b(invest|stock|bond|mortgage|loan|interest rate|tax|budget|savings?|retirement|credit|inflation|insurance|portfolio|bank)\w*")),
    ("<|is_scientific|>", _R(r"\b(physics|chemistry|biology|quantum|molecule|atom|cell|genetic|ecosystem|astronomy|geology|experiment|hypothesis|species)\w*")),
]


def guess(instruction: str, input_: str = "") -> tuple[str, list]:
    """-> (task, plan). `instruction` is the request text (instruction and input joined is fine when they cannot be told apart)."""
    text = (instruction or "")[:1500]
    task, plan = "chat", ["<|chat|>"]
    for name, verbs, rx in RULES:
        if rx.search(text):
            task, plan = name, list(verbs)
            break
    flags = [f for f, rx in DOMAIN if rx.search(text)]
    technical = any(v == "<|is_technical|>" for v in plan)
    if flags and not technical:
        plan = [flags[0]] + plan
    elif not flags and task in ("chat", "explanation", "qa", "recommendation", "analysis", "generation", "planning") and technical:
        pass
    return task, plan
