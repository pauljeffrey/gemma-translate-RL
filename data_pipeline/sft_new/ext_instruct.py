"""Second-round instruction / chat / function-calling sources.

  instruct rows (instruction, input, output)  -> one scaffolded turn; the task comes from taskguess.py (keywords), languages from GlotLID
  chat rows (list of messages)                -> every assistant turn scaffolded; <|input_lang|> from the preceding user turn, target from the answer
  function calling (glaive)                   -> tools in the system turn (+ distractors), call / result / answer as ONE assistant turn
Empty fields, the strings None / null / nan / !!NULL / NULL!! and other placeholders are normalised by textutil.blank everywhere.
"""

from __future__ import annotations
import os

import json
import re

from sft_new import hubio
from sft_new import langdetect as L
from sft_new import templates as T
from sft_new.compose import A, Rec, Skip
from sft_new.converters import (SEARCH_TOOLS, register, search_tool_def, tools_with_distractors)
from sft_new.ext_common import batched, code_of, files_of, label_code, resolve, unwrap_markup
from sft_new.taskguess import guess
from sft_new.textutil import blank, clean, degenerate, rep_ratio

MAX_PROMPT_CHARS = 14000
MAX_ANSWER_CHARS = 20000


# ------------------------------------------------------------------ instruct triples

def join_prompt(instruction: str, input_: str) -> str:
    if instruction and input_:
        return f"{instruction}\n\n{input_}"
    return instruction or input_


def instruct_rows(rows, ik: str, ink, ok: str, dataset: str, rng, *, declared: str | None = None, limit: int | None = None, meta: dict | None = None,
                  min_answer: int = 2, think_of=None, row_ok=None, task_of=None):
    """Triples of (instruction, optional input, output). `declared` = language the repo says all fields are in (audited; relabelled when a field is
    confidently another language of ours, dropped when it is confidently a language we do not carry)."""
    kept = 0
    for chunk in batched(rows, 128):
        items = []
        for r in chunk:
            if row_ok is not None and not row_ok(r):
                yield Skip("row_filter")
                continue
            ins, inp, out = blank(r.get(ik)), blank(r.get(ink)) if ink else "", blank(r.get(ok))
            if not (ins or inp):
                yield Skip("empty_prompt")
            elif not out or len(out) < min_answer:
                yield Skip("empty_response")
            elif len(ins) + len(inp) > MAX_PROMPT_CHARS or len(out) > MAX_ANSWER_CHARS:
                yield Skip("too_long")
            elif degenerate(out):
                yield Skip("repetitive_response")
            else:
                items.append((clean_keep_lines(ins), clean_keep_lines(inp), clean_keep_lines(out), r))
        if not items:
            continue
        r_ins = resolve([(i or p)[:900] for i, p, _, _ in items], declared)
        r_inp = {k: v for k, v in zip([k for k, it in enumerate(items) if it[1] and it[0]],
                                      resolve([p[:900] for i, p, _, _ in items if p and i], declared))}
        r_out = resolve([o[:900] for _, _, o, _ in items], declared)
        for k, ((ins, inp, out, r), (cs, us), (co, uo)) in enumerate(zip(items, r_ins, r_out)):
            if not cs or not co:
                yield Skip("language_unsupported_or_undetermined")
                continue
            langs, unsure = [cs], us
            if k in r_inp:
                ci, ui = r_inp[k]
                if ci and ci != cs:
                    langs.append(ci)
                unsure = unsure or ui
            prompt = join_prompt(ins, inp)
            task, plan = (task_of(r) if task_of else None) or guess(ins or inp, inp)
            think = think_of(r) if think_of else None
            yield Rec(task, dataset, langs[-1] if len(langs) > 1 else cs, co, out, input="", prompt_override=prompt, plan=plan, input_langs=langs[:3],
                      think=think, unsure_src=unsure, unsure_tgt=uo, meta={**(meta or {}), "guessed_task": task})
            kept += 1
        if limit and kept >= limit:
            return


def clean_keep_lines(s: str) -> str:
    """Whitespace tidy that keeps line structure (code, lists, poems): strip trailing spaces, collapse 3+ blank lines."""
    s = re.sub(r"[ \t]+\n", "\n", s or "")
    return re.sub(r"\n{3,}", "\n\n", s).strip()


@register("alpaca_native", (os.environ['HF_ORG'] + "/alpaca-{hausa,igbo,yoruba}-cleaned, saillab/alpaca-ewe-cleaned (instruction, input, output; language declared + audited, task from keywords)"))
def alpaca_native(rng, limit=None):
    for repo, code, match in (((os.environ['HF_ORG'] + "/alpaca-hausa-cleaned"), "hau", None), ((os.environ['HF_ORG'] + "/alpaca-igbo-cleaned"), "ibo", None),
                              ((os.environ['HF_ORG'] + "/alpaca-yoruba-cleaned"), "yor", None), ("saillab/alpaca-ewe-cleaned", "ewe", "train")):
        yield from instruct_rows(hubio.iter_repo(repo, match=match), "instruction", "input", "output", repo, rng, declared=code, limit=limit,
                                 meta={"license_note": "Alpaca-derived: upstream CC-BY-NC / OpenAI terms"})


TACO = re.compile(r"Instruction in English:\s*(?P<ie>.*?)\s*\.?\s*Response in English:\s*(?P<re>.*?)\s*\.?\s*Response in (?P<lang>[A-Za-z]+):\s*(?P<ans>.*)\s*$", re.S)


@register("alpaca_taco", (os.environ['HF_ORG'] + "/alpaca_{hausa,igbo,yoruba}_taco, saillab/alpaca_ewe_taco: instruction in X -> English pivot (<think>) -> answer in X"))
def alpaca_taco(rng, limit=None):
    for repo, code, match in (((os.environ['HF_ORG'] + "/alpaca_hausa_taco"), "hau", None), ((os.environ['HF_ORG'] + "/alpaca_igbo_taco"), "ibo", None),
                              ((os.environ['HF_ORG'] + "/alpaca_yoruba_taco"), "yor", None), ("saillab/alpaca_ewe_taco", "ewe", "train")):
        kept = 0
        for chunk in batched(hubio.iter_repo(repo, match=match), 128):
            items = []
            for r in chunk:
                ins, inp = blank(r.get("instruction")), blank(r.get("input"))
                m = TACO.search(r.get("output") or "")
                if not ins or not m:
                    yield Skip("no_taco_structure")
                    continue
                ie, re_, ans = clean_keep_lines(m.group("ie")), clean_keep_lines(m.group("re")), clean_keep_lines(m.group("ans"))
                if len(ie) < 3 or len(re_) < 3 or len(ans) < 3 or degenerate(ans):
                    yield Skip("empty_or_degenerate")
                    continue
                items.append((ins, inp, ie, re_, ans))
            if not items:
                continue
            ri = resolve([i[:600] for i, *_ in items], code)
            ra = resolve([a[:600] for *_, a in items], code)
            for (ins, inp, ie, re_, ans), (ci, ui), (ca, ua) in zip(items, ri, ra):
                if ci != code or ca != code:
                    yield Skip(f"lid_mismatch:{code}>{ci if ci != code else ca}")
                    continue
                think = f"Instruction in English: {ie}\nResponse in English: {re_}"
                prompt = join_prompt(ins, inp)
                task, plan = guess(ie, inp)
                yield Rec(task, repo, code, code, ans, input="", prompt_override=prompt, plan=plan, think=think, input_langs=[code], unsure_src=ui, unsure_tgt=ua,
                          meta={"pivot": "english", "guessed_task": task})
                kept += 1
            if limit and kept >= limit:
                break


@register("muri_it", (os.environ['HF_ORG'] + "/muri-it-language-split (input, output, per-row language) -> instruction following"))
def muri_it(rng, limit=None):
    ds = (os.environ['HF_ORG'] + "/muri-it-language-split")
    for chunk in batched(hubio.iter_repo(ds, match="train"), 128):
        by: dict = {}
        for r in chunk:
            code = code_of(str(r.get("language") or r.get("language_name") or ""))
            if not code:
                yield Skip("language_not_carried")
                continue
            by.setdefault(code, []).append(r)
        for code, rs in by.items():
            yield from instruct_rows(iter(rs), "input", None, "output", ds, rng, declared=code, meta={"muri_dataset": rs[0].get("dataset_name")})


@register("english_instruct", (os.environ['HF_ORG'] + ("/finance-alpaca, akoksal/LongForm (train), " + os.environ['HF_ORG'] + "/personal_finance_v0.2 (context -> chosen): English instruction following")))
def english_instruct(rng, limit=None):
    ds = (os.environ['HF_ORG'] + "/finance-alpaca")
    yield from instruct_rows(hubio.iter_repo(ds), "instruction", "input", "output", ds, rng, declared="eng", limit=limit,
                             meta={"domain": "finance"})
    ds = "akoksal/LongForm"
    yield from instruct_rows(hubio.iter_repo(ds, match="train"), "input", None, "output", ds, rng, declared="eng", limit=limit,
                             meta={"subset_source": "LongForm"})
    ds = (os.environ['HF_ORG'] + "/personal_finance_v0.2")
    def strip_human(r):
        c = re.sub(r"^\s*Human:\s*", "", r.get("context") or "")
        return {"q": re.split(r"\n\s*Assistant:", c)[0].strip(), "a": r.get("chosen")}
    rows = (strip_human(r) for r in hubio.iter_repo(ds))
    yield from instruct_rows(rows, "q", None, "a", ds, rng, declared="eng", limit=limit, meta={"domain": "personal_finance", "preference": "chosen"})


# ------------------------------------------------------------------ multi-turn chat

ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system", "bot": "assistant", "model": "assistant"}


def chat_convo(msgs: list, dataset: str, rng, *, system_keep: bool = True, default_plan=None, meta: dict | None = None, declared: str | None = None,
               max_turns: int = 12) -> "Rec | Skip":
    """A message list (role/content or from/value) -> a Rec whose `convo` has every assistant turn scaffolded."""
    norm = []
    for m in msgs or []:
        role = ROLE.get(str(m.get("role") or m.get("from") or "").lower())
        text = blank(m.get("content") if m.get("content") is not None else m.get("value"))
        if role is None:
            return Skip("unknown_role")
        norm.append((role, clean_keep_lines(text)))
    system = None
    if norm and norm[0][0] == "system":
        system = norm.pop(0)[1]
    if not norm or norm[0][0] != "user" or len(norm) > max_turns * 2:
        return Skip("shape")
    for i, (role, text) in enumerate(norm):
        if role != ("user" if i % 2 == 0 else "assistant") or not text:
            return Skip("shape_or_empty_turn")
        if len(text) > MAX_ANSWER_CHARS:
            return Skip("too_long")
    if norm[-1][0] != "assistant":
        norm = norm[:-1]
    if len(norm) < 2:
        return Skip("no_exchange")
    texts = [t for _, t in norm]
    res = resolve([t[:900] for t in texts], declared)
    out = [{"role": "system", "content": system}] if system and system_keep else []
    langs_seen = []
    for i in range(0, len(norm), 2):
        (cu, uu), (ca, ua) = res[i], res[i + 1]
        if not cu or not ca:
            return Skip("language_unsupported_or_undetermined")
        if degenerate(texts[i + 1]):
            return Skip("repetitive_answer")
        task, plan = guess(texts[i])
        if default_plan and plan == ["<|chat|>"]:
            plan = list(default_plan)
        from sft_new.ext_common import split_think
        think, resp = split_think(texts[i + 1])
        if think is None and re.search(r"</?(think|thinking|reasoning|scratchpad|reflection)>", resp, re.I):
            return Skip("unbalanced_think")
        try:
            content = A.assistant_content(input_lang=[cu], target_lang=ca, task_plan=plan, response=unwrap_markup(resp) if "<answer>" in resp else resp, think=think,
                                          unsure_input=uu, unsure_target=ua)
        except A.AssemblyError as e:
            return Skip("assembly:" + str(e)[:40])
        out.append({"role": "user", "content": texts[i]})
        out.append({"role": "assistant", "content": content})
        langs_seen.append((cu, ca))
    first_task, _ = guess(texts[0])
    return Rec(first_task, dataset, langs_seen[0][0], langs_seen[-1][1], texts[len(norm) - 1], input=texts[0], convo=out,
               meta={**(meta or {}), "turns": len(norm) // 2, "guessed_task": first_task})


@register("fongbe_chat", "CapitainVigs/mece-fongbe-corpus train (system fr, user fr/fon, assistant fon/fr) -> chat; languages detected per turn")
def fongbe_chat(rng, limit=None):
    ds = "CapitainVigs/mece-fongbe-corpus"
    n = 0
    for r in hubio.iter_repo(ds, match="train"):
        rec = chat_convo(r.get("messages"), ds, rng, meta={"licence_note": "card says 'other'"})
        if isinstance(rec, Skip):
            yield rec
            continue
        yield rec
        n += 1
        if limit and n >= limit:
            return


# ------------------------------------------------------------------ OpenHermes (general-knowledge subset)

TECH = re.compile(r"```|\bdef \w+\(|\bimport \w+|\bclass \w+[:(]|\bSELECT\b.*\bFROM\b|\b(python|javascript|typescript|java|c\+\+|c#|golang|rust|kotlin|swift|php|ruby|bash|powershell|html|css|sql|json|xml|yaml|regex|docker|kubernetes|git|api|sdk|compile[rd]?|debug|stack ?trace|runtime|algorithm|pseudocode|big[- ]o|unit test|function that|write (a|the) (code|program|script))\b"
                  r"|\\frac|\\sum|\\int|\$\$|\bintegral\b|\bderivative\b|\bmatrix\b|\beigen|\btheorem\b|\bpolynomial\b|\bquadratic\b|\blogarithm\b|\bcalculus\b|\bgeometry\b|\balgebra\b|\bproof\b"
                  r"|\bGLAN\b|\bairoboros\b|\bPLAINFORMAT\b|\bBEGININPUT\b|\bBEGINCONTEXT\b|\bENDINPUT\b", re.I)


def general_knowledge(msgs: list) -> bool:
    """OpenHermes also carries code, maths and structured-extraction templates. The objective of this model is conversation and general knowledge (the
    technical request is meant for a tool), so any exchange that mentions code / formal maths / the airoboros templates is left out."""
    blob = " ".join((m.get("content") or m.get("value") or "")[:1500] for m in msgs)
    return not TECH.search(blob)


@register("openhermes_general", "mathewhe/OpenHermes-2.5-Formatted (chat file): technical / code / maths / template conversations removed, ~35% sample, English")
def openhermes(rng, limit=None):
    ds = "mathewhe/OpenHermes-2.5-Formatted"
    path = "openhermes_2.5_chat.jsonl"
    cap = limit or 400_000
    n = 0
    for r in hubio.iter_file(ds, path):
        msgs = r.get("messages")
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except ValueError:
                yield Skip("unparseable")
                continue
        if not msgs or rng.random() > 0.35:
            yield Skip("not_sampled")
            continue
        if not general_knowledge(msgs):
            yield Skip("technical_or_template")
            continue
        rec = chat_convo(msgs, ds, rng, declared="eng", meta={"domain_filter": "general_knowledge"})
        if isinstance(rec, Skip):
            yield rec
            continue
        yield rec
        n += 1
        if n >= cap:
            return


# ------------------------------------------------------------------ MaLA reasoning-instruct

INST = re.compile(r"^\s*\[INST\]\s*(?P<q>.*?)\s*\[/INST\]\s*(?P<a>.*?)\s*$", re.S)


@register("mala_reasoning", "MaLA-LM/reasoning-instruct-final ([INST] .. [/INST] answer with an optional `####` result; language detected, tasks by keywords)")
def mala_reasoning(rng, limit=None):
    ds = "MaLA-LM/reasoning-instruct-final"
    n = 0
    cap = limit or 300_000
    for chunk in batched(hubio.iter_repo(ds), 256):
        items = []
        for r in chunk:
            m = INST.match(blank(r.get("content")))
            if not m:
                yield Skip("no_inst_markers")
                continue
            q, a = clean_keep_lines(m.group("q")), clean_keep_lines(m.group("a"))
            if len(q) < 8 or len(a) < 2 or degenerate(a) or len(a) > MAX_ANSWER_CHARS or rng.random() > 0.25:
                yield Skip("short_degenerate_or_not_sampled")
                continue
            items.append((q, a))
        if not items:
            continue
        rq, ra = resolve([q[:900] for q, _ in items], None), resolve([a[:900] for _, a in items], None)
        for (q, a), (cq, uq), (ca, ua) in zip(items, rq, ra):
            if not cq or not ca:
                yield Skip("language_unsupported_or_undetermined")
                continue
            task, plan = guess(q)
            if "####" in a or re.search(r"\bThe answer is\b", a):
                task, plan = "math", ["<|is_technical|>", "<|math|>"]
                a = re.sub(r"\s*####\s*", "\n#### ", a).strip()
            yield Rec(task, ds, cq, ca, a, input="", prompt_override=q, plan=plan, input_langs=[cq], unsure_src=uq, unsure_tgt=ua, meta={"guessed_task": task})
            n += 1
            if n >= cap:
                return


@register("cot_submix", "taresco/cot_submix_filtered (inputs, targets; English chain-of-thought templates, task_name kept)")
def cot_submix(rng, limit=None):
    ds = "taresco/cot_submix_filtered"
    yield from instruct_rows(hubio.iter_repo(ds), "inputs", None, "targets", ds, rng, declared="eng", limit=limit,
                             meta={"cot": True})


# ------------------------------------------------------------------ function calling (glaive)

_RETRIEVAL = re.compile(r"search|lookup|look_up|find|get_(info|details|news|weather|definition|facts|quote|movie|recipe|stock|price|exchange|nearby)|fetch|query|retrieve", re.I)
_CALC = re.compile(r"calculat|convert|compute|math|bmi|interest|tip|discount|distance|area|mortgage|loan|tax", re.I)


def call_plan(name: str) -> list:
    if _RETRIEVAL.search(name):
        return ["<|RAG|>"]
    if _CALC.search(name):
        return ["<|is_technical|>", "<|math|>"]
    return ["<|plan|>"]


def parse_functions(system: str) -> list | None:
    """The glaive system text lists functions as concatenated JSON objects after a fixed sentence."""
    i = system.find("-")
    body = system[system.find("functions. Use them if required") :] if "functions. Use them if required" in system else system
    body = body[body.find("{") :] if "{" in body else ""
    dec = json.JSONDecoder()
    tools, pos = [], 0
    while pos < len(body):
        while pos < len(body) and body[pos] in " \n\r\t":
            pos += 1
        if pos >= len(body):
            break
        try:
            obj, end = dec.raw_decode(body, pos)
        except ValueError:
            return None
        if not isinstance(obj, dict) or "name" not in obj:
            return None
        tools.append({"type": "function", "function": {"name": obj["name"], "description": obj.get("description", ""),
                                                        "parameters": obj.get("parameters") or {"type": "object", "properties": {}}}})
        pos = end
    return tools or None


_CALL = re.compile(r"<functioncall>\s*(\{.*\})\s*(?:<\|endoftext\|>)?\s*$", re.S)


def parse_call(text: str):
    """'<functioncall> {"name": "f", "arguments": '{"a": 1}'}' -> (name, arguments dict) or None."""
    m = _CALL.search(text)
    if not m:
        return None
    raw = m.group(1)
    nm = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
    am = re.search(r'"arguments"\s*:\s*(.*)\}\s*$', raw, re.S)
    if not nm or not am:
        return None
    a = am.group(1).strip()
    if a[:1] in "'\"" and a[-1:] == a[:1]:
        a = a[1:-1]
    try:
        args = json.loads(a) if a else {}
    except ValueError:
        try:
            args = json.loads(a.replace("'", '"'))
        except ValueError:
            return None
    return nm.group(1), (args if isinstance(args, dict) else {})


def _strip_eot(t: str) -> str:
    return re.sub(r"\s*<\|endoftext\|>\s*$", "", t or "").strip()


@register("glaive_function_calling", (os.environ['HF_ORG_LEGACY'] + "/glaive-function-calling-v2 -> tool-call format v2 (tools + distractors in the system turn; call / result / answer as one assistant turn)"))
def glaive(rng, limit=None):
    ds = (os.environ['HF_ORG_LEGACY'] + "/glaive-function-calling-v2")
    n = 0
    for r in hubio.iter_repo(ds):
        system = blank(r.get("system"))
        tools = parse_functions(system) if system else None
        msgs = r.get("messages") or []
        if not tools or not msgs:
            yield Skip("no_tools_or_messages")
            continue
        used_names = {t["function"]["name"] for t in tools}
        convo, ok, called = [], True, 0
        pending_text = ""
        for m in msgs:
            role, content = m.get("role"), _strip_eot(m.get("content") or "")
            if role == "user":
                if not content:
                    ok = False
                    break
                convo.append({"role": "user", "content": content})
                pending_text = ""
            elif role == "assistant":
                call = parse_call(content) if "<functioncall>" in content else None
                if "<functioncall>" in content and not call:
                    ok = False
                    break
                if call:
                    name, args = call
                    if name not in used_names:
                        ok = False
                        break
                    convo.append(A.assistant_message({"input_lang": ["eng"], "task_plan": call_plan(name), "tool_call": {"name": name, "arguments": args}}))
                    called += 1
                else:
                    if not content:
                        ok = False
                        break
                    prev_tool = bool(convo) and convo[-1]["role"] == "tool"
                    plan = (call_plan(next(iter(used_names))) if prev_tool else ["<|chat|>"])
                    if prev_tool:
                        plan = [p for p in plan if p.startswith("<|") or p] + ["<|explain|>"]
                    try:
                        text = A.assistant_content(input_lang=["eng"], target_lang="eng", task_plan=plan, response=content)
                    except A.AssemblyError as e:
                        ok = False
                        pending_text = str(e)[:40]
                        break
                    convo.append({"role": "assistant", "content": text})
            elif role == "function":
                if not content or not convo or convo[-1].get("role") != "assistant" or "tool_calls" not in convo[-1]:
                    ok = False
                    break
                convo.append({"role": "tool", "content": content})
            else:
                ok = False
                break
        if not ok or called == 0 and rng.random() < 0.5:
            yield Skip("unconvertible_or_call_free_sampled_out" if ok else "unconvertible:" + pending_text)
            continue
        if not convo or convo[-1]["role"] != "assistant" or "tool_calls" in convo[-1]:
            yield Skip("ends_without_answer")
            continue
        distract = [t for t in tools_with_distractors(tools[0], rng, 3, 5) if t["function"]["name"] not in used_names]
        allt = tools + distract[: max(0, 5 - len(tools)) + 1]
        rng.shuffle(allt)
        sysmsg = {"role": "system", "content": "You are a helpful assistant. When you need information or an action you cannot produce yourself, call one of the functions listed below, then answer from what it returns."}
        convo = [sysmsg] + convo
        last = convo[-1]["content"].split("<response>", 1)[-1]
        yield Rec("function_calling" if called else "chat", ds, "eng", "eng", last, input=convo[1]["content"], convo=convo, tools=allt,
                  meta={"calls": called, "tools_in_source": len(tools)})
        n += 1
        if limit and n >= limit:
            return
