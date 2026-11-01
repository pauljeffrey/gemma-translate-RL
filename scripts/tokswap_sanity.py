#!/usr/bin/env python3
"""tokswap_sanity.py -- CPU sanity checks of a swapped (or control) 270M checkpoint against the original checkpoint.

For every language in --loss-langs: teacher-forced NLL of the reference translation under the exact training prompt
    <bos><start_of_turn>user\\nTranslate the text to {Language}:  {english}<end_of_turn>\\n<start_of_turn>model\\n   +   {iso} {translation}<end_of_turn>
(prompt and target tokenised as two segments, as in training), with the ORIGINAL checkpoint + ORIGINAL tokenizer and with the SWAPPED
checkpoint + SWAPPED tokenizer (control: swapped weights + original tokenizer).  Reported per language: tokens, characters,
per-token NLL, per-character NLL (characters of "{iso} {translation}" + 1 for <end_of_turn>), both for the FULL softmax and for the
softmax RESTRICTED to the ids that were not re-used (for sentences that both tokenizers segment identically the restricted loss is identical because hidden states are identical; the
full-softmax loss differs slightly because the re-used rows now carry new embeddings; the restricted loss is undefined where the target contains a new token).
Greedy decoding (--gen-langs, first --n-gen FLORES+ dev English sources): token-id identity of the generations and chrF++ vs reference.

All numerics: weights are cast to bf16 (the original SFT-final is stored in fp32) and computed in fp32 on CPU, so original and swapped
differ only by the swap.  Nothing is trained, no GPU.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

FLORES = None
NAMES = None


def load_model(path, threads):
    torch.set_num_threads(threads)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    return m.float().eval()


def flores(lang_code, split="dev"):
    return [json.loads(l)["text"] for l in open(os.path.join(FLORES, split, f"{lang_code}.jsonl"), encoding="utf-8")]


def prompt_ids(tk, name, eng):
    p = f"<bos><start_of_turn>user\nTranslate the text to {name}:  {eng.strip()}<end_of_turn>\n<start_of_turn>model\n"
    return tk.encode(p, add_special_tokens=False).ids


def target_ids(tk, iso, text):
    return tk.encode(f"{iso} {text.strip()}", add_special_tokens=False).ids + [106]


@torch.no_grad()
def nll(model, tk, items, removed, bs=4):
    """items: list of (prompt_ids_text_args...) -> returns list of (full_nll_sum, restricted_nll_sum, n_tokens)"""
    out = []
    mask = torch.zeros(262144, dtype=torch.bool)
    mask[list(removed)] = True
    for k in range(0, len(items), bs):
        batch = items[k:k + bs]
        seqs = [(prompt_ids(tk, nm, en), target_ids(tk, iso, tx)) for nm, en, iso, tx in batch]
        L = max(len(p) + len(t) for p, t in seqs)
        ids = torch.zeros(len(seqs), L, dtype=torch.long); att = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, (p, t) in enumerate(seqs):
            s = p + t
            ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
        logits = model(input_ids=ids, attention_mask=att).logits
        for i, (p, t) in enumerate(seqs):
            pos = torch.arange(len(p) - 1, len(p) + len(t) - 1)
            tgt = torch.tensor(t)
            lg = logits[i, pos].float()                      # (T, V): only the target positions, to keep memory small
            lp_full = torch.log_softmax(lg, -1)
            lp_res = torch.log_softmax(lg.masked_fill(mask, float("-inf")), -1)
            r = float(-lp_res[torch.arange(len(t)), tgt].sum())
            out.append((float(-lp_full[torch.arange(len(t)), tgt].sum()), r if np.isfinite(r) else float("nan"), len(t), tuple(t)))
        del logits
    return out


@torch.no_grad()
def greedy(model, tk, items, max_new, bs=10):
    res = []
    for k in range(0, len(items), bs):
        batch = items[k:k + bs]
        ps = [prompt_ids(tk, nm, en) for nm, en in batch]
        L = max(map(len, ps))
        ids = torch.zeros(len(ps), L, dtype=torch.long); att = torch.zeros(len(ps), L, dtype=torch.long)
        for i, p in enumerate(ps):   # left padding
            ids[i, L - len(p):] = torch.tensor(p); att[i, L - len(p):] = 1
        g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False, eos_token_id=[1, 106], pad_token_id=0,
                           num_beams=1, temperature=None, top_p=None, top_k=None)
        for i in range(len(ps)):
            o = g[i, L:].tolist()
            if 106 in o: o = o[:o.index(106)]
            if 1 in o: o = o[:o.index(1)]
            while o and o[-1] == 0: o.pop()
            res.append(o)
    return res


def main():
    global FLORES, NAMES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig-model", required=True); ap.add_argument("--orig-tok", required=True)
    ap.add_argument("--model", required=True, help="swapped or control checkpoint dir (contains tokenizer.json)")
    ap.add_argument("--swap-map", required=True); ap.add_argument("--flores", required=True); ap.add_argument("--names", required=True)
    ap.add_argument("--loss-langs", nargs="+", default=["amh_Ethi", "tir_Ethi", "hau_Latn", "yor_Latn", "swh_Latn", "kin_Latn", "ibo_Latn", "fon_Latn"])
    ap.add_argument("--gen-langs", nargs="+", default=["hau_Latn", "yor_Latn", "swh_Latn", "amh_Ethi"])
    ap.add_argument("--n-loss", type=int, default=300); ap.add_argument("--n-gen", type=int, default=50); ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--threads", type=int, default=24); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    FLORES = a.flores; NAMES = json.load(open(a.names, encoding="utf-8"))
    smap = json.load(open(a.swap_map, encoding="utf-8"))
    removed = {t["id"] for t in smap["removed_tokens"]}
    t_o = Tokenizer.from_file(a.orig_tok); t_s = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    m_o = load_model(a.orig_model, a.threads); m_s = load_model(a.model, a.threads)
    eng = flores("eng_Latn")
    rep = {"model": a.model, "n_removed_ids": len(removed), "loss": {}, "gen": {}}
    t0 = time.time()
    for lg in a.loss_langs:
        iso = lg.split("_")[0]
        tx = flores(lg)[:a.n_loss]
        items = [(NAMES[iso], eng[i], iso, tx[i]) for i in range(len(tx))]
        chars = [len(f"{iso} {t.strip()}") + 1 for t in tx]
        ro = nll(m_o, t_o, items, removed); rs = nll(m_s, t_s, items, removed)
        S = lambda r, j: np.array([x[j] for x in r])
        c = np.array(chars)
        ident = np.array([x[3] == y[3] for x, y in zip(ro, rs)])   # sentence tokenised identically by both tokenizers
        d = {"n": len(items), "chars": int(c.sum()), "n_identical_tokenisation": int(ident.sum()),
             "orig_tokens": int(S(ro, 2).sum()), "swapped_tokens": int(S(rs, 2).sum()),
             "orig_nll_per_token": S(ro, 0).sum() / S(ro, 2).sum(), "swapped_nll_per_token": S(rs, 0).sum() / S(rs, 2).sum(),
             "orig_nll_per_char": S(ro, 0).sum() / c.sum(), "swapped_nll_per_char": S(rs, 0).sum() / c.sum(),
             "sentences_with_identical_token_count": int((S(ro, 2) == S(rs, 2)).sum())}
        if ident.any():   # restricted softmax (re-used ids masked) is defined only where the targets contain no re-used id
            d["identical_sentences_full_nll_orig"] = float(S(ro, 0)[ident].sum()); d["identical_sentences_full_nll_swapped"] = float(S(rs, 0)[ident].sum())
            d["identical_sentences_restricted_nll_orig"] = float(S(ro, 1)[ident].sum()); d["identical_sentences_restricted_nll_swapped"] = float(S(rs, 1)[ident].sum())
            d["identical_sentences_max_abs_diff_full_nll"] = float(np.abs(S(ro, 0) - S(rs, 0))[ident].max())
            d["identical_sentences_max_abs_diff_restricted_nll"] = float(np.abs(S(ro, 1) - S(rs, 1))[ident].max())
        if (~ident).any():
            d["changed_sentences_max_abs_diff_full_nll"] = float(np.abs(S(ro, 0) - S(rs, 0))[~ident].max())
            d["changed_sentences_median_abs_diff_full_nll"] = float(np.median(np.abs(S(ro, 0) - S(rs, 0))[~ident]))
        d = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in d.items()}
        rep["loss"][lg] = d
        print(f"[{time.time()-t0:.0f}s] loss {lg}: tok/sent {d['orig_tokens']/d['n']:.1f}->{d['swapped_tokens']/d['n']:.1f}  nll/token {d['orig_nll_per_token']:.4f}->{d['swapped_nll_per_token']:.4f} "
              f"nll/char {d['orig_nll_per_char']:.4f}->{d['swapped_nll_per_char']:.4f}  identical-tokenisation sentences {d['n_identical_tokenisation']}/{d['n']}"
              + (f" restricted-diff max {d['identical_sentences_max_abs_diff_restricted_nll']:.3g} full-diff max {d['identical_sentences_max_abs_diff_full_nll']:.3g}" if ident.any() else ""), flush=True)
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=2)
    for lg in a.gen_langs:
        iso = lg.split("_")[0]
        src = [(NAMES[iso], eng[i]) for i in range(a.n_gen)]
        ref = flores(lg)[:a.n_gen]
        go = greedy(m_o, t_o, src, a.max_new); gs = greedy(m_s, t_s, src, a.max_new)
        def dec(tk, ids):
            t = tk.decode(ids, skip_special_tokens=True).strip()
            return t[len(iso) + 1:].strip() if t.startswith(iso + " ") else t
        to, ts = [dec(t_o, x) for x in go], [dec(t_s, x) for x in gs]
        same_ids = sum(1 for x, y in zip(go, gs) if x == y)
        same_txt = sum(1 for x, y in zip(to, ts) if x == y)
        d = {"n": a.n_gen, "token_id_identical": same_ids, "text_identical": same_txt,
             "chrF++_original_vs_ref": chrf.corpus_score(to, [ref]).score, "chrF++_swapped_vs_ref": chrf.corpus_score(ts, [ref]).score,
             "chrF++_swapped_vs_original_output": chrf.corpus_score(ts, [to]).score,
             "orig_gen_tokens_mean": float(np.mean([len(x) for x in go])), "swapped_gen_tokens_mean": float(np.mean([len(x) for x in gs])),
             "examples": [{"src": src[i][1][:120], "orig": to[i][:200], "swapped": ts[i][:200]} for i in range(len(to)) if to[i] != ts[i]][:4]}
        rep["gen"][lg] = d
        print(f"[{time.time()-t0:.0f}s] greedy {lg}: token-identical {same_ids}/{a.n_gen}, text-identical {same_txt}/{a.n_gen}, chrF++ orig {d['chrF++_original_vs_ref']:.2f} swapped {d['chrF++_swapped_vs_ref']:.2f}", flush=True)
        outs = {"orig": to, "swapped": ts, "ref": ref, "orig_ids": go, "swapped_ids": gs}
        json.dump(outs, open(a.out.replace(".json", f".gen_{lg}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
