#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A MEASURED micro SFT top-up on CPU (D-47): how far do a few thousand human pairs of a language the RL model cannot write move it?

Fine-tunes the RL checkpoint on N human English->Fula pairs (NLLB-Seed `fuv_Latn`, CC-BY-SA-4.0, English = `eng_Latn` of the same ids; the table row `ful`, name "Fulah", tag `ful` is used so that no
table row is added), mixed with a small replay of the model's OWN greedy outputs for trained languages (so the update is not pure new-language), and evaluates at checkpoints
    * Fula: FLORES+ dev fuv_Latn probe set (60) and SmolSent ff (60): free generation (the model must write the `ful` tag itself), GlotLID right-language rate, chrF++, mean log-prob
    * forgetting: chrF++ of 4 trained languages (yor hau wol kbp, first 30 probe sentences) before / after
Nothing is saved to disk except the result json; nothing is pushed. CPU only.

    python scripts/newlang_cpu_topup.py --model DIR --sets-dir DIR --out result.json --n-train 3100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def load_pairs(n_max, seed):
    from huggingface_hub import hf_hub_download

    en = {r["id"]: r["text"] for r in map(json.loads, open(hf_hub_download("openlanguagedata/oldi_seed", "seed/eng_Latn.jsonl", repo_type="dataset"), encoding="utf-8"))}
    fu = {r["id"]: r["text"] for r in map(json.loads, open(hf_hub_download("openlanguagedata/oldi_seed", "seed/fuv_Latn.jsonl", repo_type="dataset"), encoding="utf-8"))}
    rows = [(en[i], fu[i]) for i in sorted(en) if i in fu and 5 <= len(en[i].split()) <= 45 and fu[i].strip()]
    random.Random(seed).shuffle(rows)
    return rows[:n_max]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=3100)
    ap.add_argument("--replay-langs", nargs="+", default=["yor", "hau", "swh", "wol", "bem", "ach", "kbp", "nus"])
    ap.add_argument("--replay-per-lang", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--eval-fracs", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--lid-model", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-n", type=int, default=60)
    ap.add_argument("--save-probe", default=None, help="after training: greedy + 4 T=1 samples for the FLORES+ dev probe set, written in the format of newlang_probe.py (label topup, key fuv, cond free) for newlang_probe_metrics.py")
    ap.add_argument("--forget-n", type=int, default=30)
    a = ap.parse_args(argv)

    import torch
    from sacrebleu.metrics import CHRF
    from transformers import AutoTokenizer

    from gemma_mt import format as fmt
    from gemma_mt.languages import ALL_ISOS
    from gemma_mt.modeling import load_model
    from gemma_mt.rl.lid import GlotLIDClassifier
    from gemma_mt.rl.sampling import generate_completions
    from gemma_mt.rl.text import decode_completion, nfc

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_model(a.model, dtype=torch.float32, device="cpu")
    stop = fmt.resolve_stop_tokens(tok, model.generation_config)
    fmt.ensure_generation_config(model.generation_config, stop, pad_id=tok.pad_token_id)
    chrf = CHRF(word_order=2)
    lid = GlotLIDClassifier(path=a.lid_model)
    lid._load()
    FULA = {"fuv_Latn", "fub_Latn", "fuh_Latn", "fuf_Latn", "fuq_Latn", "ffm_Latn", "fue_Latn", "fud_Latn"}

    def load_set(name, n):
        return [json.loads(l) for l in open(os.path.join(a.sets_dir, name), encoding="utf-8") if l.strip()][:n]

    probes = {"fuv_flores": ("fuv__flores_fuv_Latn.jsonl", "ful", a.probe_n), "ff_smol": ("fuv__smol_ff.jsonl", "ful", a.probe_n)}
    forget = {k: (f"{k}__flores_{k}_Latn.jsonl" if k in ("yor", "hau", "wol", "kbp") else None, k, 30) for k in ("yor", "hau", "wol", "kbp")}

    def evaluate(tag):
        res = {"tag": tag}
        model.eval()
        for nm, (fn, iso, n) in probes.items():
            rows = load_set(fn, n)
            outs, lps = [], []
            for s in range(0, len(rows), 20):
                b = rows[s: s + 20]
                roll = generate_completions(model, [fmt.build_prompt(r["english"], "ful", tok) for r in b], group_size=1, max_new_tokens=128, temperature=1.0, top_p=1.0, top_k=0,
                                            stop_ids=stop.ids, pad_id=tok.pad_token_id, device="cpu", greedy=True)
                for r, c, fin in zip(b, roll.completions, roll.finished):
                    d = decode_completion(tok, c, stop.ids, pad_ids=[tok.pad_token_id])
                    txt = d.text
                    t_tag = txt.split(" ", 1)[0] if txt else ""
                    body = txt[len(t_tag):].strip() if t_tag == "ful" else txt
                    outs.append((nfc(body), t_tag == "ful", fin, r["reference"]))
            labs = lid.top([o[0] for o in outs], 1)
            res[nm] = dict(n=len(outs), tag_ful_rate=sum(o[1] for o in outs) / len(outs), finished=sum(o[2] for o in outs) / len(outs),
                           lid_fula_rate=sum(1 for l in labs if l[0][0] in FULA) / len(outs), chrf=sum(chrf.sentence_score(o[0], [o[3]]).score for o in outs) / len(outs),
                           lid_top=dict(sorted({x: sum(1 for l in labs if l[0][0] == x) for x in {l[0][0] for l in labs}}.items(), key=lambda kv: -kv[1])[:3]), example=outs[3][0][:140])
        fg = {}
        for k, (fn, iso, n) in forget.items():
            rows = load_set(fn, n)
            hyp = []
            for s in range(0, len(rows), 15):
                b = rows[s: s + 15]
                roll = generate_completions(model, [fmt.build_prompt(r["english"], k, tok) for r in b], group_size=1, max_new_tokens=128, temperature=1.0, top_p=1.0, top_k=0,
                                            stop_ids=stop.ids, pad_id=tok.pad_token_id, device="cpu", greedy=True)
                for c in roll.completions:
                    d = decode_completion(tok, c, stop.ids, pad_ids=[tok.pad_token_id])
                    t = d.text
                    hyp.append(nfc(t[len(k):].strip() if t.startswith(k + " ") else t))
            fg[k] = sum(chrf.sentence_score(h, [r["reference"]]).score for h, r in zip(hyp, rows)) / len(rows)
        res["forgetting_chrf"] = fg
        return res

    # ---- training examples
    pairs = load_pairs(a.n_train, a.seed)
    train = [fmt.build_example(en, "ful", fu, tok) for en, fu in pairs]
    # replay: the model's own greedy output on seed English sentences, for trained languages (keeps the update from being pure new-language)
    rng = random.Random(a.seed + 1)
    seed_en = [r[0] for r in load_pairs(2000, a.seed + 5)]
    rep = []
    t0 = time.time()
    model.eval()
    for k in a.replay_langs:
        src = rng.sample(seed_en, a.replay_per_lang)
        for s in range(0, len(src), 20):
            b = src[s: s + 20]
            roll = generate_completions(model, [fmt.build_prompt(e, k, tok) for e in b], group_size=1, max_new_tokens=128, temperature=1.0, top_p=1.0, top_k=0,
                                        stop_ids=stop.ids, pad_id=tok.pad_token_id, device="cpu", greedy=True)
            for e, c, fin in zip(b, roll.completions, roll.finished):
                if not fin:
                    continue
                d = decode_completion(tok, c, stop.ids, pad_ids=[tok.pad_token_id])
                if d.text.startswith(k + " ") and len(d.text) > len(k) + 5:
                    rep.append(fmt.build_example(e, k, d.text[len(k) + 1:], tok))
    print(f"replay examples {len(rep)} ({time.time() - t0:.0f}s), new-language examples {len(train)}", flush=True)
    data = [x for x in train + rep if len(x.input_ids) <= a.max_len]
    random.Random(a.seed + 2).shuffle(data)
    steps = math.ceil(len(data) / a.batch)
    eval_steps = {int(round(f * steps)) for f in a.eval_fracs}
    results = {"args": vars(a), "n_new": len(train), "n_replay": len(rep), "steps": steps, "evals": [], "loss": []}
    results["evals"].append({"step": 0, **evaluate(0)})
    print("eval 0", json.dumps(results["evals"][-1]["fuv_flores"]), flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    softcap = getattr(model.config, "final_logit_softcapping", None)
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        batch = data[(step - 1) * a.batch: step * a.batch]
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / 10)
        tot, ntok = 0.0, sum(sum(1 for t in x.labels if t != -100) for x in batch)
        for s in range(0, len(batch), a.micro):
            mb = batch[s: s + a.micro]
            L = max(len(x.input_ids) for x in mb)
            ids = torch.full((len(mb), L), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(mb), L), -100, dtype=torch.long)
            att = torch.zeros((len(mb), L), dtype=torch.long)
            for i, x in enumerate(mb):
                ids[i, : len(x.input_ids)] = torch.tensor(x.input_ids)
                lab[i, : len(x.labels)] = torch.tensor(x.labels)
                att[i, : len(x.input_ids)] = 1
            h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
            tgt = lab[:, 1:]
            m = tgt != -100
            logits = model.lm_head(h[m])
            if softcap:
                logits = torch.tanh(logits / softcap) * softcap
            loss = torch.nn.functional.cross_entropy(logits.float(), tgt[m], reduction="sum") / ntok
            loss.backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        results["loss"].append(round(tot, 4))
        if step % 10 == 0:
            print(f"step {step}/{steps} loss {tot:.3f} {time.time() - t0:.0f}s", flush=True)
        if step in eval_steps:
            ev = {"step": step, "pairs_seen": min(step * a.batch, len(data)), **evaluate(step)}
            results["evals"].append(ev)
            print("eval", step, json.dumps(ev["fuv_flores"]), json.dumps(ev["forgetting_chrf"]), flush=True)
            json.dump(results, open(a.out, "w"), ensure_ascii=False, indent=1)
    results["train_seconds"] = time.time() - t0
    if a.save_probe:
        import newlang_probe as NP

        rows = load_set("fuv__flores_fuv_Latn.jsonl", a.probe_n)
        out_rows = []
        model.eval()
        for s0 in range(0, len(rows), 30):
            chunk = rows[s0: s0 + 30]
            prompts = [fmt.build_prompt(r["english"], "ful", tok) for r in chunk]
            torch.manual_seed(1000 + s0)
            for kind, g, greedy in (("greedy", 1, True), ("sample", 4, False)):
                comps, fins, lps = NP.gen(model, prompts, g, greedy, 160, stop.ids, tok.pad_token_id)
                for i, ids in enumerate(comps):
                    d = decode_completion(tok, ids, stop.ids, pad_ids=[tok.pad_token_id])
                    ptag, rest = NP.split_tag(d.text, set(ALL_ISOS) | {"eng"})
                    out_rows.append({"pi": s0 + i // g, "kind": kind, "k": i % g, "tag": ptag, "prefilled": False, "text": rest, "raw": d.text, "finished": bool(fins[i]), "n_tok": len(ids),
                                     "ids": ids, "logp": [round(x, 4) for x in lps[i]]})
        os.makedirs(a.save_probe, exist_ok=True)
        with open(os.path.join(a.save_probe, "topup__fuv__flores_fuv_Latn__free.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"meta": {"label": "topup", "model": "rl2c step 925 + CPU micro top-up on 3,100 Fula pairs", "key": "fuv", "set": "flores_fuv_Latn", "cond": "free", "name": "Fulah", "tag": "ful",
                                         "n": len(rows), "samples": 4, "seconds": 0, "max_new_tokens": 160, "english": [r["english"] for r in rows], "reference": [r["reference"] for r in rows]}}, ensure_ascii=False) + "\n")
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(results, open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
