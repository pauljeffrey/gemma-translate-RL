#!/usr/bin/env python3
"""(f) Mixer + packer of the clean science-stage SFT mixture (agent C2, 2026-10-28).

    python s4_mix.py --out MIX --model NEW_RL_MODEL --science W/afs/science_windows.jsonl --val-sci W/afs/val_sci.jsonl --replay W/replay/replay_train.jsonl \
        --replay-val W/replay/replay_val.jsonl --archive W/arch/archive_selected.jsonl --longdoc W/ld/longdoc_selected.jsonl --anchor W/an/anchor_selected.jsonl \
        --index I.npz --manifest M.json [--ratio-mode report|shrink] [--exclude-source libretexts] [--exclude-classes nc,nc_teacher]

Reproduces the A2W token shares (science 31.7 / replay 43.0 / archive 5.0 / long-document 13.5 / anchors 6.7 %, 20.88 M tokens = 319 steps of 65,536) from the CLEAN components:
  * the total is anchored on the science tokens (all of them are used, two passes): total = science / 0.317, every other component is trimmed to share x total when it is larger
    (quality score first for archive / long documents, seeded random for the rest; per-language proportional); a component that is SHORT is reported, and with --ratio-mode shrink the
    larger ones are reduced so that the shares are exact;
  * every row is checked once more against the protected index (strict, no exemption, English AND target; a hit drops the row), refused when its source/generator matches NLLB /
    TranslatePsy / teacher, and dropped by --exclude-source / --exclude-classes; exact duplicates (same English + target) are dropped;
  * packing = the old `write_split` (gemma_mt.data.prepare_sft, packs of 8,192 tokens, whole samples, mixed languages); val/ = replay validation (77-language forgetting monitor),
    val_sci/ = AfriScience-MT dev (monitor only, never trained on);
  * per-row PROVENANCE: OUT/train/provenance.parquet (one line per packed sample, in the packed order: component, language, source_dataset, licence label, licence class, the
    unresolved-licence note (LibreTexts, AfriScience), generator, prov_* references, n_tokens, target_tokens), train/source.npy (component code) and train/licence_class.npy;
  * OUT/sft_science_clean.yaml: the A2W training config (R4's hyper-parameters) pointing at this mixture and the new RL model; OUT/mixture_report.md / .json: shares vs A2W.
`s4_decon_final.py` then re-decodes the packed training tokens and runs the strict protected-index check on them (the number to report: 0).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s4_common as K  # noqa: E402

A2W_YAML = """model_name: /workspace/sft_build/runs/final_1433
tokenizer_name: null
data_dir: /workspace/jobs/r11/mix_v2A2W
out_dir: /workspace/jobs/r11/runs/sci_sft_v2A2W
attn_implementation: varlen
mask_mode: auto
loss_impl: fused
param_dtype: float32
autocast_dtype: bfloat16
micro_batch_size: 2
grad_accum_steps: 4
epochs: 1.0
max_steps: 0
optimizer: adamw
learning_rate: 3.0e-6
min_lr_ratio: 0.1
warmup_steps: 0
warmup_ratio: 0.06
weight_decay: 0.0
beta1: 0.9
beta2: 0.95
eps: 1.0e-8
grad_clip: 1.0
l2sp_coef: 0.0
track_drift: true
extra_val_splits: [val_sci]
eval_at_start: true
eval_every: 100
eval_max_packs: 0
early_stopping_patience: 0
early_stopping_min_delta: 0.0
min_epochs: 0.0
early_stop_action: stop
val_metric: macro
min_val_tokens_macro: 2000
save_every: 50
log_every: 5
save_optimizer: false
keep_single_checkpoint: true
keep_prev_checkpoint: false
resume: true
resume_optimizer: true
optimizer_rewarmup_steps: 10
gradient_checkpointing: true
loss_chunk_tokens: 512
seed: 1234
wandb_project: null
run_name: sci_sft_v2A2W
hub_repo_id: null"""

SCORE = {"archive": lambda r: r.get("ens_mean") if r.get("ens_mean") is not None else r.get("reward", 0.0),
         "longdoc": lambda r: r.get("mean_r") if r.get("mean_r") is not None else r.get("mean_conf", 0.0)}


def h12(a) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(a, dtype=np.uint32).tobytes(), digest_size=12).digest()


def rank_within_language(rows, key):
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[r["iso"]].append(i)
    pct = [0.0] * len(rows)
    for iso, idx in by.items():
        idx.sort(key=lambda i: key(rows[i]))
        for rank, i in enumerate(idx):
            pct[i] = rank / max(len(idx) - 1, 1)
    return pct


def trim(rows, budget, comp, seed):
    """Keep rows of one component up to `budget` tokens. archive/longdoc: highest within-language score percentile first; others: seeded random order. -> (kept, dropped)"""
    if sum(r["n_tokens"] for r in rows) <= budget:
        return rows, []
    rng = random.Random(f"{seed}|{comp}|trim")
    if comp in SCORE:
        pct = rank_within_language(rows, SCORE[comp])
        order = sorted(range(len(rows)), key=lambda i: (-pct[i], rng.random()))
    else:
        order = list(range(len(rows)))
        rng.shuffle(order)
    kept, used = [], 0
    drop = []
    for i in order:
        if used + rows[i]["n_tokens"] <= budget:
            kept.append(rows[i])
            used += rows[i]["n_tokens"]
        else:
            drop.append(rows[i])
    return kept, drop


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="the NEW RL model (tokenizer fingerprint; start model of the stage)")
    for c in K.COMPONENTS:
        ap.add_argument(f"--{c}", default="", help=f"jsonl of the {c} component")
    ap.add_argument("--val-sci", default="")
    ap.add_argument("--replay-val", default="")
    ap.add_argument("--index", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ratio-mode", choices=("report", "shrink"), default="report")
    ap.add_argument("--shares", default=",".join(f"{k}={v}" for k, v in K.A2W_SHARES.items()))
    ap.add_argument("--total-tokens", type=float, default=0.0, help="override; default science tokens / science share")
    ap.add_argument("--exclude-source", default="", help="comma separated substrings of source_dataset to drop from the NEW components and the replay (e.g. libretexts)")
    ap.add_argument("--exclude-classes", default="nc,nc_teacher", help="licence classes dropped (add 'unknown' for the strict commercial-clean reading)")
    ap.add_argument("--seq-len", type=int, default=K.SEQ_LEN)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--tokens-per-step", type=int, default=65536)
    ap.add_argument("--base-config", default="", help="A2W yaml to copy hyper-parameters from (default: gemma_mt/configs/sft_science_v2_A2W.yaml)")
    ap.add_argument("--run-name", default="sci_sft_clean")
    ap.add_argument("--smoke", action="store_true", help="label the outputs SMOKE (no config is written for training use)")
    a = ap.parse_args(argv)
    K.gemma_pkg()
    K.old_scripts()
    from gemma_mt import format as fmt
    from gemma_mt.data.prepare_sft import tokenizer_fingerprint
    from gemma_mt.languages import ALL_ISOS, LANGUAGES, iso_to_id
    import build_sft_science_mix as B

    os.makedirs(a.out, exist_ok=True)
    tok = K.tokenizer(a.model)
    fp = tokenizer_fingerprint(tok)
    shares = {k: float(v) for k, v in (x.split("=") for x in a.shares.split(","))}
    excl_src = [x.lower() for x in a.exclude_source.split(",") if x]
    excl_cls = {x for x in a.exclude_classes.split(",") if x}
    rs = os.path.join(os.path.dirname(os.path.abspath(a.replay)), "replay_stats.json") if a.replay else ""
    if rs and os.path.exists(rs) and json.load(open(rs)).get("tokenizer_fingerprint") not in (None, fp):
        raise SystemExit("the replay packs were tokenised with another tokenizer than --model (use the packs that match the new RL model's tokenizer, e.g. packed_S2)")
    P = K.Protected(a.index, a.manifest, a.nproc)
    K.log(f"protected index {P.version} (strict, no exemption)")
    st = {c: collections.Counter() for c in K.COMPONENTS}
    comp_rows, seen = {}, set()
    for c in K.COMPONENTS:
        path = getattr(a, c)
        if not path or not os.path.exists(path):
            comp_rows[c] = []
            continue
        rows = K.read_jsonl(path)
        st[c]["in"] = len(rows)
        texts = [r["english"] for r in rows] + [r["target"] for r in rows]
        hits = P.names(texts)
        keep = []
        for k, r in enumerate(rows):
            if not r.get("english") or not r.get("target"):
                st[c]["empty"] += 1
                continue
            if r.get("uses_nllb_or_translatepsy") or K.is_forbidden_source(r.get("source_dataset", ""), r.get("generator", ""), r.get("prov_ref", "")):
                st[c]["forbidden_source"] += 1
                continue
            if any(x in (r.get("source_dataset") or "").lower() for x in excl_src):
                st[c]["excluded_source"] += 1
                continue
            if r.get("licence_class") in excl_cls:
                st[c]["licence_excluded:" + r["licence_class"]] += 1
                continue
            h = hits[k] + hits[len(rows) + k]
            if h:
                st[c]["protected_hit"] += 1
                for nm in set(h):
                    st[c]["protected:" + nm] += 1
                continue
            ex = K.example(tok, r["english"], r["iso"], r["target"])
            if ex.n_tokens > a.seq_len:
                st[c]["longer_than_pack"] += 1
                continue
            hh = h12(np.asarray(ex.input_ids))
            if hh in seen:
                st[c]["duplicate"] += 1
                continue
            seen.add(hh)
            r["n_tokens"], r["target_tokens"] = ex.n_tokens, ex.target_len
            r["_ids"] = np.asarray(ex.input_ids, dtype=np.uint32)
            r["_pl"] = ex.prompt_len
            keep.append(r)
        st[c]["kept"] = len(keep)
        comp_rows[c] = keep
        K.log(f"{c}: {dict(st[c])}")
    avail = {c: sum(r["n_tokens"] for r in comp_rows[c]) for c in K.COMPONENTS}
    # ---- sizes
    if a.ratio_mode == "shrink":
        total = a.total_tokens or min(avail[c] / shares[c] for c in K.COMPONENTS if shares[c] > 0 and avail[c] > 0)
        # a component that is empty cannot be shrunk to: report
    else:
        total = a.total_tokens or (avail["science"] / shares["science"] if avail["science"] else sum(avail.values()))
    budget = {c: shares[c] * total for c in K.COMPONENTS}
    final, dropped_tr = {}, {}
    for c in K.COMPONENTS:
        slack = 1.0 if a.ratio_mode == "shrink" else 1.02
        final[c], dropped_tr[c] = trim(comp_rows[c], budget[c] * slack, c, a.seed)
    toks = {c: sum(r["n_tokens"] for r in final[c]) for c in K.COMPONENTS}
    allt = sum(toks.values())
    short = {c: round(1 - toks[c] / budget[c], 4) for c in K.COMPONENTS if budget[c] and toks[c] < 0.98 * budget[c]}
    K.log(f"tokens by component {toks}; total {allt:,} (A2W 20,880,766); shares " + ", ".join(f"{c} {toks[c] / max(allt, 1):.3f}" for c in K.COMPONENTS) + f"; short: {short}")
    # ---- pack
    items, meta_rows = [], []
    for ci, c in enumerate(K.COMPONENTS):
        for r in final[c]:
            items.append((B._ex(r["_ids"], r["_pl"], r["target_tokens"]), iso_to_id(r["iso"])))
            meta_rows.append((ci, r))
    perm = list(range(len(items)))
    random.Random(a.seed).shuffle(perm)                       # the packer is order sensitive: mix the components
    items = [items[i] for i in perm]
    meta_rows = [meta_rows[i] for i in perm]
    train_meta = B.write_split(items, a.out, a.seq_len, a.seed, "train")
    d = os.path.join(a.out, "train")
    off, ln = np.load(os.path.join(d, "sample_offset.npy")), np.load(os.path.join(d, "sample_len.npy"))
    tk = np.memmap(os.path.join(d, "tokens.u32.bin"), dtype=np.uint32, mode="r")
    pack_start = np.load(os.path.join(d, "pack_start.npy"))
    by_hash = {}
    for ci, r in meta_rows:
        by_hash[h12(r["_ids"])] = (ci, r)
    n = len(ln)
    src = np.zeros(n, np.uint8)
    lic = np.zeros(n, np.uint8)
    cols = collections.defaultdict(list)
    pack_of = np.searchsorted(pack_start[1:], np.arange(n), side="right")
    for i in range(n):
        ci, r = by_hash[h12(np.asarray(tk[int(off[i]):int(off[i]) + int(ln[i])]))]
        src[i] = ci
        lic[i] = K.LICENCE_CLASSES.index(r["licence_class"]) if r["licence_class"] in K.LICENCE_CLASSES else 3
        cols["sample"].append(i)
        cols["pack"].append(int(pack_of[i]))
        for k in ("component", "iso", "source_dataset", "licence", "licence_class", "licence_note", "licence_unresolved", "prov_repo", "prov_ref", "prov_index", "generator",
                  "uses_nllb_or_translatepsy", "n_tokens", "target_tokens"):
            cols[k].append(r.get(k, ""))
    np.save(os.path.join(d, "source.npy"), src)
    np.save(os.path.join(d, "licence_class.npy"), lic)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({k: v for k, v in cols.items()}), os.path.join(d, "provenance.parquet"), compression="zstd")
    # ---- validation splits
    meta_splits = {"train": train_meta}
    for name, path in (("val", a.replay_val), ("val_sci", a.val_sci)):
        if not path or not os.path.exists(path):
            continue
        its = []
        for r in K.read_jsonl(path):
            ex = K.example(tok, r["english"], r["iso"], r["target"])
            if ex.n_tokens <= a.seq_len:
                its.append((B._ex(ex.input_ids, ex.prompt_len, ex.target_len), iso_to_id(r["iso"])))
        tmp = os.path.join(a.out, f"_{name}_tmp")
        meta_splits[name] = B.write_split(its, tmp, a.seq_len, a.seed + (1 if name == "val" else 2), "val")
        shutil.rmtree(os.path.join(a.out, name), ignore_errors=True)
        shutil.move(os.path.join(tmp, "val"), os.path.join(a.out, name))
        shutil.rmtree(tmp, ignore_errors=True)
    # ---- tables, meta, report
    tab = {c: collections.defaultdict(lambda: {"examples": 0, "tokens": 0, "target_tokens": 0}) for c in K.COMPONENTS}
    for c in K.COMPONENTS:
        for r in final[c]:
            t = tab[c][r["iso"]]
            t["examples"] += 1
            t["tokens"] += r["n_tokens"]
            t["target_tokens"] += r["target_tokens"]
    by_lic = collections.Counter()
    unresolved = collections.Counter()
    by_src = collections.Counter()
    for c in K.COMPONENTS:
        for r in final[c]:
            by_lic[(c, r["licence_class"])] += r["n_tokens"]
            by_src[(c, r["source_dataset"])] += r["n_tokens"]
            if r.get("licence_note"):
                unresolved[(c, r["source_dataset"], r["licence"])] += r["n_tokens"]
    steps = math.ceil(train_meta["n_tokens"] / a.tokens_per_step)
    report = {"label": "SMOKE" if a.smoke else "production-candidate", "index_version": P.version, "model": a.model, "tokenizer_fingerprint": fp, "ratio_mode": a.ratio_mode,
              "tokens_by_component": toks, "shares": {c: toks[c] / max(allt, 1) for c in K.COMPONENTS}, "a2w_shares": K.A2W_SHARES, "a2w_tokens": K.A2W_TOKENS, "total_tokens": allt,
              "a2w_total_tokens": 20_880_766, "rows_by_component": {c: len(final[c]) for c in K.COMPONENTS}, "short_components": short, "component_stats": {c: dict(v) for c, v in st.items()},
              "trimmed_rows": {c: len(dropped_tr[c]) for c in K.COMPONENTS}, "optimizer_steps": steps, "tokens_per_step": a.tokens_per_step, "packs": train_meta["n_packs"],
              "tokens_by_licence_class": {f"{c}|{k}": v for (c, k), v in sorted(by_lic.items())},
              "unresolved_licence_rows_tokens": {f"{c}|{s}|{l}": v for (c, s, l), v in sorted(unresolved.items())},
              "tokens_by_source": {f"{c}|{s}": v for (c, s), v in sorted(by_src.items(), key=lambda kv: -kv[1])[:60]}, "options": vars(a)}
    json.dump(report, open(os.path.join(a.out, "mixture_report.json"), "w"), indent=1)
    meta = {"format_version": 1, "dtype": "uint32", "seq_len": a.seq_len, "instruction_template": fmt.INSTRUCTION_TEMPLATE, "end_of_turn": fmt.END_OF_TURN, "ignore_index": fmt.IGNORE_INDEX,
            "tokenizer": {"name": getattr(tok, "name_or_path", a.model), "requested": a.model, "revision": None, "fingerprint": fp, "vocab_size": len(tok), "bos_id": tok.bos_token_id,
                          "eos_id": tok.eos_token_id, "pad_id": tok.pad_token_id, "end_of_turn_id": tok.convert_tokens_to_ids(fmt.END_OF_TURN)},
            "iso_table": list(ALL_ISOS), "directions": ["en2x"], "pivot_tag": fmt.PIVOT_ISO, "splits": meta_splits, "mixture_clean": report}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, ensure_ascii=False)
    L = ["| tag | language | " + " | ".join(f"{c} ex | {c} tok" for c in K.COMPONENTS) + " | total tok |", "|---|---|" + "---:|" * (2 * len(K.COMPONENTS) + 1)]
    for iso in ALL_ISOS:
        row = [tab[c].get(iso, {"examples": 0, "tokens": 0}) for c in K.COMPONENTS]
        t = sum(x["tokens"] for x in row)
        if t:
            L.append(f"| {iso} | {LANGUAGES[iso].name} | " + " | ".join(f"{x['examples']:,} | {x['tokens']:,}" for x in row) + f" | {t:,} |")
    L.append("| **all** | | " + " | ".join(f"{sum(v['examples'] for v in tab[c].values()):,} | {toks[c]:,}" for c in K.COMPONENTS) + f" | {allt:,} |")
    open(os.path.join(a.out, "mixture_tables.md"), "w").write("\n".join(L) + "\n")
    rep = ["# Mixture report" + (" (SMOKE)" if a.smoke else ""), "", f"index {P.version}; model {a.model}; ratio mode {a.ratio_mode}", "",
           "| component | rows | tokens | share | A2W share | A2W tokens | short? |", "|---|---:|---:|---:|---:|---:|---|"]
    for c in K.COMPONENTS:
        rep.append(f"| {c} | {len(final[c]):,} | {toks[c]:,} | {toks[c] / max(allt, 1):.3f} | {K.A2W_SHARES[c]:.3f} | {K.A2W_TOKENS[c]:,} | {'SHORT ' + str(short[c]) if c in short else ''} |")
    rep.append(f"| **total** | {sum(len(final[c]) for c in K.COMPONENTS):,} | {allt:,} | | | 20,880,766 | steps {steps} (A2W 319) |")
    open(os.path.join(a.out, "mixture_report.md"), "w").write("\n".join(rep) + "\n")
    # ---- training config (A2W hyper-parameters)
    base = a.base_config or os.path.join(K.REPO, "gemma_mt", "configs", "sft_science_v2_A2W.yaml")
    if True:
        txt = open(base).read() if os.path.exists(base) else A2W_YAML           # the embedded copy is the A2W file without its comments (R4's hyper-parameters)
        txt = "\n".join(l for l in txt.splitlines() if not l.startswith("#"))
        import re

        txt = re.sub(r"^model_name:.*$", f"model_name: {a.model}", txt, flags=re.M)
        txt = re.sub(r"^data_dir:.*$", f"data_dir: {os.path.abspath(a.out)}", txt, flags=re.M)
        txt = re.sub(r"^out_dir:.*$", f"out_dir: {os.path.abspath(a.out)}_run", txt, flags=re.M)
        txt = re.sub(r"^run_name:.*$", f"run_name: {a.run_name}", txt, flags=re.M)
        hdr = (f"# {'SMOKE ' if a.smoke else ''}clean science-stage SFT config written by s4_mix.py: A2W hyper-parameters (lr 3e-6, 65,536 tokens/step, cosine, 6 % warm-up, fp32 master weights)\n"
               f"# data {a.out}: {train_meta['n_samples']:,} samples, {train_meta['n_packs']:,} packs, {train_meta['n_tokens']:,} tokens -> {steps} steps (A2W: 319). Start model = the NEW RL model.\n"
               "# Interpolate afterwards: theta = (1-alpha) theta_RL + alpha theta_SFT, alpha in {0.3, 0.5, 0.7, 1.0} (A2W chose 0.7); guardrails as in reports/sft_science_stage_2026-10-21.md.\n")
        open(os.path.join(a.out, "sft_science_clean.yaml"), "w").write(hdr + txt + "\n")
    P.close()
    K.log(f"TRAIN: {train_meta['n_samples']:,} samples, {train_meta['n_packs']:,} packs of {a.seq_len}, {train_meta['n_tokens']:,} tokens, {steps} steps; "
          f"shares " + ", ".join(f"{c} {toks[c] / max(allt, 1):.3f}" for c in K.COMPONENTS))


if __name__ == "__main__":
    main()
