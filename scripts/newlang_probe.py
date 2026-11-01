#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Zero-shot generation probe of the new-language study (D-47). CPU only, resumable, raw outputs (metrics are computed by newlang_probe_metrics.py).

For every (language key, reference set, condition) it generates, for the first N sentences of the probe set,
    * 1 greedy completion per sentence                                  (kind "greedy")
    * G temperature-1.0 samples per sentence (top_p 1, top_k 0 = the RL sampling of D-39)   (kind "sample")
and writes one jsonl per (model label, key, set, condition): {pi, kind, k, tag, tag_status, text, finished, n_tok, ids, logp}.

conditions
    free       the instruction names the language ("Translate the text to Fula:  ..."), the model writes its own tag (what the RL policy would face for a new language)
    prefill    like free, but the plain ISO tag of the language is appended to the prompt, so the model only has to write the translation
    ctrl:<t>   the instruction names the trained RELATIVE <t> (its table name); the output is scored against the new language's reference (does the model just write the relative?)
    trained    a trained language with its own tag (the yardstick)
    base       google/gemma-3-270m-it, same prompt as `free` (a leading tag is stripped when present)

    python scripts/newlang_probe.py --model DIR --label rl2c --sets-dir DIR --out DIR --keys fuv gaa --conds free prefill --n 60 --samples 4 --threads 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import newlang_registry as R  # noqa: E402

TAG_RE = re.compile(r"^([A-Za-z]{3})(?:\s+|$)")


def split_tag(text, known):
    """-> (tag or None, rest). A leading 3-letter token that is a known language tag (ours, plus those of the registry) is the tag."""
    m = TAG_RE.match(text or "")
    if m and m.group(1).lower() in known:
        return m.group(1).lower(), text[m.end():].strip()
    return None, (text or "").strip()


def user_prompt_ids(tok, fmt, english, name):
    from gemma_mt.format import INSTRUCTION_TEMPLATE

    content = INSTRUCTION_TEMPLATE.format(african_language=name, english_text=english.strip())
    text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    fmt._check_single_bos(tok, ids)
    return ids


class LpRecorder:
    """LogitsProcessor that records the log-prob of every token the sampler chose, WITHOUT keeping the full-vocabulary scores of all steps (HF `output_scores` keeps
    [batch, 262k] floats per step: 16 GB for 80 sequences x 192 steps). At call t+1 it reads the token chosen at step t from `input_ids[:, -1]` and gathers it from the
    log-softmax saved at call t. The very last chosen token (normally the stop id) is therefore not recorded. T = 1 and top_p = 1 / top_k = 0: the raw logits ARE the sampler's."""

    def __init__(self):
        self.prev = None
        self.lps = []

    def __call__(self, input_ids, scores):
        import torch

        if self.prev is not None:
            self.lps.append(self.prev.gather(1, input_ids[:, -1:]).squeeze(1).cpu())
        self.prev = torch.log_softmax(scores.float(), dim=-1)
        return scores


def gen(model, prompts, G, greedy, max_new, stop_ids, pad_id):
    """-> (list of kept id lists (stop id included), finished flags, list of per-token logps (one shorter than ids when finished), post-stop counts)."""
    import torch
    from transformers import LogitsProcessorList

    from gemma_mt.rl.text import cut_at_stop

    width = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, p in enumerate(prompts):
        ids[i, width - len(p):] = torch.tensor(list(p), dtype=torch.long)
        mask[i, width - len(p):] = 1
    k = 1 if greedy else G
    kw = dict(do_sample=False, temperature=None, top_p=None, top_k=None) if greedy else dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
    rec = LpRecorder()
    model.eval()
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, num_return_sequences=k, eos_token_id=list(stop_ids), pad_token_id=pad_id,
                             use_cache=True, logits_processor=LogitsProcessorList([rec]), **kw)
    gen_ids = out[:, width:].cpu().tolist()
    lp = torch.stack(rec.lps, dim=1).tolist() if rec.lps else [[] for _ in gen_ids]       # [N, steps-1]
    comps, fin, lps = [], [], []
    for r, row in enumerate(gen_ids):
        kept, finished = cut_at_stop(row, stop_ids)
        comps.append(kept)
        fin.append(finished)
        lps.append(lp[r][: len(kept)])
    return comps, fin, lps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sets-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--conds", nargs="+", default=["free"])
    ap.add_argument("--sets", default="primary", choices=["primary", "all"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--greedy-only", action="store_true", help="skip the sampled completions (controls)")
    ap.add_argument("--max-new-tokens", type=int, default=0, help="0 = 128 for smol / local sets (short sentences), 160 for FLORES / NTREX")
    ap.add_argument("--prompts-per-call", type=int, default=30)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoTokenizer

    from gemma_mt import format as fmt
    from gemma_mt.languages import ALL_ISOS, LANGUAGES
    from gemma_mt.modeling import load_model
    from gemma_mt.rl.text import decode_completion

    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_model(a.model, dtype=torch.float32, device="cpu")
    stop = fmt.resolve_stop_tokens(tok, model.generation_config)
    fmt.ensure_generation_config(model.generation_config, stop, pad_id=tok.pad_token_id)
    known = set(ALL_ISOS) | {"eng"} | set(R.NEW)
    os.makedirs(a.out, exist_ok=True)

    def sets_of(key):
        reg = R.NEW.get(key) or R.YARDSTICK[key]
        specs = reg["sets"] if a.sets == "all" else reg["sets"][:1]
        return [s.replace(":", "_").replace("/", "_") for s in specs]

    for key in a.keys:
        for cond in a.conds:
            for sname in sets_of(key):
                path = os.path.join(a.sets_dir, f"{key}__{sname}.jsonl")
                if not os.path.exists(path):
                    print("missing set", path, flush=True)
                    continue
                outp = os.path.join(a.out, f"{a.label}__{key}__{sname}__{cond.replace(':', '-')}.jsonl")
                if os.path.exists(outp):
                    print("skip (exists)", outp, flush=True)
                    continue
                rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()][: a.n]
                trained = key in R.YARDSTICK
                if (trained and cond not in ("trained", "base")) or (not trained and cond == "trained"):
                    continue
                if cond == "trained":
                    name, tag = LANGUAGES[key].name, key
                    build = lambda en: fmt.build_prompt(en, key, tok)        # noqa: E731
                elif cond in ("base", "free"):
                    name, tag = (LANGUAGES[key].name if trained else R.NEW[key]["name"]), None
                    build = lambda en: user_prompt_ids(tok, fmt, en, name)      # noqa: E731
                elif cond == "prefill":
                    name, tag = R.NEW[key]["name"], R.NEW[key]["tag"]
                    pre = tok(tag, add_special_tokens=False)["input_ids"]
                    build = lambda en: user_prompt_ids(tok, fmt, en, name) + pre      # noqa: E731
                elif cond.startswith("ctrl:"):
                    rel = cond.split(":", 1)[1]
                    name, tag = LANGUAGES[rel].name, rel
                    build = lambda en: fmt.build_prompt(en, rel, tok)          # noqa: E731
                else:
                    raise ValueError(cond)
                max_new = a.max_new_tokens or (128 if sname.startswith(('smol', 'local')) else 160)
                t0 = time.time()
                out_rows = []
                for s in range(0, len(rows), a.prompts_per_call):
                    chunk = rows[s: s + a.prompts_per_call]
                    prompts = [build(r["english"]) for r in chunk]
                    torch.manual_seed(a.seed * 100003 + s)
                    rolls = [("greedy", 1, gen(model, prompts, 1, True, max_new, stop.ids, tok.pad_token_id))]
                    if not a.greedy_only and a.samples > 0:
                        rolls.append(("sample", a.samples, gen(model, prompts, a.samples, False, max_new, stop.ids, tok.pad_token_id)))
                    for kind, g, (comps, fins, lps) in rolls:
                        for i, ids in enumerate(comps):
                            pi, k = s + i // g, i % g
                            d = decode_completion(tok, ids, stop.ids, pad_ids=[tok.pad_token_id])
                            ptag, rest = split_tag(d.text, known)
                            out_rows.append({"pi": pi, "kind": kind, "k": k, "tag": ptag, "prefilled": cond == "prefill", "text": rest,
                                             "raw": d.text, "finished": bool(fins[i]), "n_tok": len(ids),
                                             "ids": ids, "logp": [round(x, 4) for x in lps[i]]})
                    print(f"{a.label} {key} {sname} {cond}: {min(s + a.prompts_per_call, len(rows))}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
                with open(outp + ".tmp", "w", encoding="utf-8") as f:
                    f.write(json.dumps({"meta": {"label": a.label, "model": a.model, "key": key, "set": sname, "cond": cond, "name": name, "tag": tag, "n": len(rows), "samples": a.samples,
                                                 "seconds": time.time() - t0, "max_new_tokens": max_new, "english": [r["english"] for r in rows],
                                                 "reference": [r["reference"] for r in rows]}}, ensure_ascii=False) + "\n")
                    for r in out_rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                os.replace(outp + ".tmp", outp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
