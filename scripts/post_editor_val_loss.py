#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Like-for-like validation loss of the base translator and the editor on the SAME held-out editor validation rows (owner question: "why is the editor's validation loss lower than the base model's?").

    python scripts/post_editor_val_loss.py --examples PACKED/examples_val.jsonl --base RL_DIR --editor EDITOR_DIR --out OUT.json [--limit 0]

Token-weighted mean negative log-likelihood (nats / token) of the target `{tag} {human reference}<end_of_turn>` under four conditions, on the validation rows of the editor set:
  base_translation       base model, the ordinary translation prompt (English only)                         = the task the base model was trained on
  editor_translation     editor model, the same translation prompt (did the editor forget translating?)
  editor_empty_draft     editor model, editor prompt with an EMPTY draft                                    = the editor without the information that makes its task easy
  editor_with_draft      editor model, editor prompt with the (matched-quality) draft                       = what the training log reports as val_loss
  base_editor_prompt     base model, editor prompt with the draft (never trained on it): how much of the gain is the prompt + draft alone
and the split of editor_with_draft by draft kind (matched / identity / near). The editor's val_loss in the training log is NOT comparable with the base translator's: it conditions on a draft
(often a near copy of the answer; 5-7% of the rows have draft == reference), which makes every reference token far more predictable.
"""
import argparse, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", required=True); ap.add_argument("--base", required=True); ap.add_argument("--editor", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="google/gemma-3-270m-it"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gemma_mt import format as fmt
    from gemma_mt.post_edit import build_pe_example
    fmt.set_tag_mode("single_token")
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = [json.loads(l) for l in open(a.examples, encoding="utf-8") if l.strip()]
    if a.limit: rows = rows[: a.limit]

    def nll(model, exs):
        tot, per = [0.0, 0], []
        order = sorted(range(len(exs)), key=lambda i: len(exs[i].input_ids)); res = [None] * len(exs)
        for s in range(0, len(order), a.batch):
            idx = order[s:s + a.batch]; L = max(len(exs[i].input_ids) for i in idx)
            ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long); lab = torch.full((len(idx), L), -100, dtype=torch.long); att = torch.zeros((len(idx), L), dtype=torch.long)
            for k, i in enumerate(idx):
                e = exs[i]; n = len(e.input_ids); ids[k, :n] = torch.tensor(e.input_ids); lab[k, :n] = torch.tensor(e.labels); att[k, :n] = 1
            with torch.no_grad():
                lg = model(input_ids=ids.to(a.device), attention_mask=att.to(a.device)).logits.float()
            lp = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), lab[:, 1:].to(a.device).reshape(-1), ignore_index=-100, reduction="none").reshape(len(idx), L - 1)
            m = (lab[:, 1:] != -100).to(a.device)
            for k, i in enumerate(idx):
                res[i] = (float((lp[k] * m[k]).sum()), int(m[k].sum()))
        return res

    ex_tr = [fmt.build_example(r["english"], r["iso"], r["reference"], tok) for r in rows]
    ex_pe = [build_pe_example(r["english"], r["draft"], r["iso"], r["reference"], tok) for r in rows]
    ex_empty = [build_pe_example(r["english"], "", r["iso"], r["reference"], tok) for r in rows]
    out = {"n_rows": len(rows)}
    conds = {}
    for name, path in (("base", a.base), ("editor", a.editor)):
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(a.device).eval()
        if name == "base":
            conds["base_translation"] = nll(model, ex_tr); conds["base_editor_prompt"] = nll(model, ex_pe)
        else:
            conds["editor_translation"] = nll(model, ex_tr); conds["editor_empty_draft"] = nll(model, ex_empty); conds["editor_with_draft"] = nll(model, ex_pe)
        del model; torch.cuda.empty_cache()

    def agg(res, sel=None):
        s = sum(r[0] for i, r in enumerate(res) if sel is None or sel(rows[i])); n = sum(r[1] for i, r in enumerate(res) if sel is None or sel(rows[i]))
        return round(s / max(1, n), 4)

    def macro(res):
        by = defaultdict(lambda: [0.0, 0])
        for r, row in zip(res, rows): by[row["iso"]][0] += r[0]; by[row["iso"]][1] += r[1]
        v = [s / n for s, n in by.values() if n >= 100]
        return round(sum(v) / max(1, len(v)), 4)
    for k, res in conds.items():
        out[k] = {"token_weighted_nll": agg(res), "macro_over_languages": macro(res)}
    out["editor_with_draft_by_kind"] = {kind: agg(conds["editor_with_draft"], lambda r, kind=kind: r["kind"] == kind) for kind in ("matched", "identity", "near") if any(r["kind"] == kind for r in rows)}
    out["base_translation_by_kind"] = {kind: agg(conds["base_translation"], lambda r, kind=kind: r["kind"] == kind) for kind in ("matched", "identity", "near") if any(r["kind"] == kind for r in rows)}
    bins = [(0, 20), (20, 40), (40, 60), (60, 101)]
    out["editor_with_draft_by_draft_chrf"] = {f"{lo}-{hi}": agg(conds["editor_with_draft"], lambda r, lo=lo, hi=hi: r["kind"] == "matched" and lo <= r["chrf"] < hi) for lo, hi in bins}
    out["base_translation_by_draft_chrf_bin"] = {f"{lo}-{hi}": agg(conds["base_translation"], lambda r, lo=lo, hi=hi: r["kind"] == "matched" and lo <= r["chrf"] < hi) for lo, hi in bins}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
