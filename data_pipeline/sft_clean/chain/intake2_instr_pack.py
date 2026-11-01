#!/usr/bin/env python3
"""Instruction / chat / QA stream -> training packs in the SAME format as the translation packs (gemma_mt.data.prepare_sft: tokens.u32.bin, sample_offset, sample_len, prompt_len, target_len, iso_id,
direction, pack_start, meta.json with the gemma_mt.languages iso_table and the tokenizer fingerprint), plus the UniMax schedule and a proposal to interleave it with the translation stream.

    python intake2_instr_pack.py build    --src /workspace/work/instr --out /workspace/work/work/packed_instr --tokenizer google/gemma-3-270m-it --workers 24
    python intake2_instr_pack.py schedule --packed /workspace/work/work/packed_instr --out /workspace/work/work/sched_instr [--trans-packs-csv iso,packs] [--share 0.25] [--lam-trans 20000]
    python intake2_instr_pack.py interleave --trans SCHED_T --instr SCHED_I --n-trans-packs N --out SCHED_TI        # proposal only; needs the translation packs/schedule

How a row becomes a sample (decisions):
  * the conversation is rendered by the tokenizer's own chat template (fmt.render_chat, system turn folded into the first user turn by the template), exactly as the translation prompts;
  * NO language tag is put in front of the answer (the translation format's `{iso} ` prefix is a translation-task convention); the answer is the plain assistant turn;
  * loss mask: prompt_len = everything before the FINAL assistant turn (earlier assistant turns of multi-turn chats carry NO loss: the pack format has one prompt span and one target span),
    target_len = final assistant turn up to and including <end_of_turn>; the trailing "\\n" is the masked tail (sample_len = prompt + target + tail), as in fmt.build_example;
  * samples longer than 4,096 tokens are dropped (never truncated); one language per pack (pack_by_language), best-fit-decreasing in shuffled buffers (pack_buffered), seq_len 4096;
  * direction = 0 for every sample (the direction field is a translation concept; 0 keeps packed_dataset/validation happy);
  * iso_id = index in gemma_mt.languages.ALL_ISOS; a language that is not in that table has no tag and its rows are NOT packed (counted in meta['excluded_no_tag']);
  * val split: 0.2% of the rows by hash of the row id, at least 60 rows for languages with >= 1,500 rows (same rule family as the translation packs).
"""
import argparse, collections, csv, glob, hashlib, json, math, os, shutil, sys, time
import numpy as np
import pyarrow.parquet as pq
import xxhash

for p in ("/workspace/work/code", os.environ.get("GEMMA_MT_PKG", ""), "/path/to/repo/gemma_mt"):
    if p and os.path.isdir(os.path.join(p, "gemma_mt")) and p not in sys.path:
        sys.path.insert(0, p)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEQ = 4096
VAL_FRACTION = 0.002
MIN_VAL = 60
MIN_ROWS_FOR_FORCED_VAL = 1500
_TOK = None


def _init(name):
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(name, use_fast=True)


def build_chat_example(msgs, tok):
    """(input_ids, prompt_len, target_len, tail_len) or an error string. Mirrors gemma_mt.format.build_example for a multi-turn conversation: loss on the final assistant turn only."""
    from gemma_mt import format as fmt
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
        return "shape"
    prompt = fmt.render_chat(tok, msgs[:-1], add_generation_prompt=True)
    full = fmt.render_chat(tok, msgs, add_generation_prompt=False)
    if not full.startswith(prompt):
        return "prompt_not_prefix"
    suffix = full[len(prompt):]
    cut = suffix.rfind(fmt.END_OF_TURN)
    if cut < 0:
        return "no_end_of_turn"
    target_str, tail_str = suffix[: cut + len(fmt.END_OF_TURN)], suffix[cut + len(fmt.END_OF_TURN):]
    p_ids, t_ids = fmt._encode(tok, prompt), fmt._encode(tok, target_str)
    tail = fmt._encode(tok, tail_str) if tail_str else []
    eot = tok.convert_tokens_to_ids(fmt.END_OF_TURN)
    if not t_ids or t_ids[-1] != eot or (len(p_ids) > 1 and p_ids[0] == p_ids[1] == tok.bos_token_id):
        return "bad_ids"
    return p_ids + t_ids + tail, len(p_ids), len(t_ids), len(tail)


def _work(chunk):
    out = []
    for rid, keep, messages in chunk:
        try:
            ex = build_chat_example(json.loads(messages), _TOK)
        except Exception as e:                                                   # template error (roles etc.)
            ex = "template_error"
        if isinstance(ex, str):
            out.append((ex, None, 0, 0, 0))
        elif keep:
            out.append(("ok", np.asarray(ex[0], np.uint32), ex[1], ex[2], ex[3]))
        else:
            out.append(("ok_untagged", None, len(ex[0]), 0, 0))              # only the length is needed (token count of languages without a tag)
    return out


def hv(s):
    return xxhash.xxh64_intdigest(s.encode("utf-8", "surrogatepass")) / 2.0 ** 64


def source_dirs(src):
    return sorted(d for d in glob.glob(os.path.join(src, "*")) if os.path.isdir(d) and not os.path.basename(d).startswith("_") and glob.glob(os.path.join(d, "part-*.parquet")))


def build(a):
    from transformers import AutoTokenizer
    from gemma_mt import format as fmt
    from gemma_mt.data import prepare_sft as P
    from gemma_mt.languages import ALL_ISOS, iso_to_id
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.tokenizer, use_fast=True)
    fmt.assert_template_matches_reference(tok)
    stop = fmt.resolve_stop_tokens(tok)
    srcs = source_dirs(a.src)
    # pass 1: ids + iso only -> validation membership (hash of the row id; at least MIN_VAL rows for big languages)
    ids_by_iso = collections.defaultdict(list)
    for d in srcs:
        for fp in sorted(glob.glob(os.path.join(d, "part-*.parquet"))):
            t = pq.read_table(fp, columns=["id", "iso"])
            for rid, iso in zip(t.column("id").to_pylist(), t.column("iso").to_pylist()):
                if iso in ALL_ISOS:
                    ids_by_iso[iso].append(rid)
    val = set()
    for iso, ids in ids_by_iso.items():
        hs = sorted((hv(i), i) for i in ids)
        k = sum(1 for h, _ in hs if h < VAL_FRACTION)
        if len(ids) >= MIN_ROWS_FOR_FORCED_VAL:
            k = max(k, MIN_VAL)
        val.update(i for _, i in hs[:k])
    print("rows with a tag:", sum(len(v) for v in ids_by_iso.values()), "val rows:", len(val), flush=True)
    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(a.out)
    cfg = P.PrepareConfig(input="(intake2 instr shards)", out_dir=a.out, tokenizer=a.tokenizer, seq_len=SEQ, max_sample_tokens=SEQ, val_fraction=VAL_FRACTION, min_val_per_language=MIN_VAL, seed=0,
                          buffer_size=100_000, directions=("en2x",), workers=a.workers, pack_by_language=True)
    stores = {s: P._SampleStore(os.path.join(a.out, "_tmp_store", s)) for s in P.SPLITS}

    class Ex:                                                                      # the attributes _SampleStore.add reads
        pass

    def chunks():
        cur, meta = [], []
        for d in srcs:
            name = os.path.basename(d)
            for fp in sorted(glob.glob(os.path.join(d, "part-*.parquet"))):
                t = pq.read_table(fp, columns=["id", "iso", "messages", "synthetic"])
                for rid, iso, m, sy in zip(*(t.column(c).to_pylist() for c in ("id", "iso", "messages", "synthetic"))):
                    cur.append((rid, iso in ALL_ISOS, m)); meta.append((name, iso, rid, bool(sy)))
                    if len(cur) >= 500:
                        yield cur, meta; cur, meta = [], []
        if cur:
            yield cur, meta

    import multiprocessing as mp
    dropped, per_src, no_tag = collections.Counter(), collections.defaultdict(collections.Counter), collections.defaultdict(lambda: collections.Counter())
    n_in = 0
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.tokenizer,)) as pool:
        pending = collections.deque()
        it = chunks()

        def consume(res, meta):
            nonlocal n_in
            for (st, ids, pl, tl, tail), (name, iso, rid, sy) in zip(res, meta):
                n_in += 1
                if st == "ok_untagged":
                    no_tag[iso]["rows"] += 1; no_tag[iso]["tokens"] += pl; no_tag[iso]["synthetic_rows"] += int(sy)
                    continue
                if st != "ok":
                    dropped["format_" + st] += 1; continue
                if len(ids) > SEQ:
                    dropped["too_long"] += 1; per_src[name]["too_long"] += 1; continue
                ex = Ex(); ex.input_ids, ex.prompt_len, ex.target_len = ids, pl, tl
                stores["val" if rid in val else "train"].add(ex, iso_to_id(iso), 0)
                per_src[name]["samples"] += 1; per_src[name]["tokens"] += len(ids)
                per_src[name]["synthetic_samples"] += int(sy)
        while True:
            try:
                ch, meta = next(it)
                pending.append((pool.apply_async(_work, (ch,)), meta))
            except StopIteration:
                break
            while len(pending) >= 3 * a.workers:
                r, m = pending.popleft(); consume(r.get(), m)
        while pending:
            r, m = pending.popleft(); consume(r.get(), m)
    split_meta = {s: P._write_split(stores[s], os.path.join(a.out, s), cfg, s) for s in P.SPLITS}
    shutil.rmtree(os.path.join(a.out, "_tmp_store"), ignore_errors=True)
    meta = {
        "format_version": P.FORMAT_VERSION, "dtype": P.DTYPE, "seq_len": SEQ, "instruction_template": "(none: free chat, see stream)", "end_of_turn": fmt.END_OF_TURN, "ignore_index": fmt.IGNORE_INDEX,
        "tokenizer": {"name": getattr(tok, "name_or_path", a.tokenizer), "requested": a.tokenizer, "revision": None, "fingerprint": P.tokenizer_fingerprint(tok), "vocab_size": len(tok),
                      "bos_id": tok.bos_token_id, "eos_id": tok.eos_token_id, "pad_id": tok.pad_token_id, "end_of_turn_id": stop.end_of_turn_id},
        "iso_table": list(ALL_ISOS), "directions": ["en2x"], "pivot_tag": fmt.PIVOT_ISO, "rows_after_cleaning": n_in, "dropped": dict(dropped), "splits": split_meta, "config": P.asdict(cfg), "warnings": list(stop.warnings),
        "stream": "instruction/chat/QA (intake2_instr)", "loss": "final assistant turn only (prompt_len = everything before it); no language tag in front of the answer; direction=0",
        "excluded_no_tag": {k: dict(v) for k, v in sorted(no_tag.items())}, "per_source": {k: dict(v) for k, v in sorted(per_src.items())}, "sources": [os.path.basename(d) for d in srcs], "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2, ensure_ascii=False)
    print("train packs", split_meta["train"]["n_packs"], "tokens", split_meta["train"]["n_tokens"], "val packs", split_meta["val"]["n_packs"], "dropped", dict(dropped), "no-tag", {k: v["rows"] for k, v in no_tag.items()})
    langs_csv(a.out)


def langs_csv(out):
    """per-language table of the packed dataset: samples, tokens, packs, mean tokens/pack"""
    from gemma_mt.languages import ALL_ISOS
    rows = []
    for sp in ("train", "val"):
        d = os.path.join(out, sp)
        iso, sl = np.load(d + "/iso_id.npy"), np.load(d + "/sample_len.npy").astype(np.int64)
        ps = np.load(d + "/pack_start.npy")
        pack_iso = iso[ps[:-1].astype(np.int64)]
        for i in np.unique(iso):
            m = iso == i
            rows.append(dict(split=sp, iso=ALL_ISOS[int(i)], samples=int(m.sum()), tokens=int(sl[m].sum()), packs=int((pack_iso == i).sum())))
    with open(os.path.join(out, "per_language.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def audit(a):
    """structural audit of a packed dir (the translation audit_packed expects the translation template, so this is the free-chat version): bos first, <end_of_turn> closes the target, lengths add up,
    packs <= 4096 tokens and single-language, labels cover only the target; prints decoded examples"""
    from transformers import AutoTokenizer
    from gemma_mt.languages import ALL_ISOS
    meta = json.load(open(os.path.join(a.packed, "meta.json")))
    tok = AutoTokenizer.from_pretrained(meta["tokenizer"]["requested"], use_fast=True)
    eot, bos = tok.convert_tokens_to_ids("<end_of_turn>"), tok.bos_token_id
    bad = collections.Counter()
    for sp in ("train", "val"):
        d = os.path.join(a.packed, sp)
        L = {n: np.load(f"{d}/{n}.npy") for n in ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "pack_start")}
        toks = np.memmap(d + "/tokens.u32.bin", dtype=np.uint32, mode="r")
        rng = np.random.default_rng(0)
        n = len(L["sample_len"]); ps = L["pack_start"].astype(np.int64)
        for i in rng.choice(n, size=min(a.n, n), replace=False):
            o, ln, pl, tl = int(L["sample_offset"][i]), int(L["sample_len"][i]), int(L["prompt_len"][i]), int(L["target_len"][i])
            if toks[o] != bos: bad["no_bos"] += 1
            if toks[o + pl + tl - 1] != eot: bad["target_not_eot"] += 1
            if not (0 <= ln - pl - tl <= 1): bad["tail_len"] += 1
            if toks[o + 1] == bos: bad["double_bos"] += 1
        for p in rng.choice(len(ps) - 1, size=min(a.n // 4, len(ps) - 1), replace=False):
            seg = slice(ps[p], ps[p + 1])
            if int(L["sample_len"][seg].sum()) > SEQ: bad["pack_over_seq"] += 1
            if len(set(L["iso_id"][seg].tolist())) != 1: bad["mixed_language_pack"] += 1
        print(sp, "samples", n, "packs", len(ps) - 1, "problems", dict(bad), flush=True)
        for i in rng.choice(n, size=min(a.decode, n), replace=False):
            o, ln, pl, tl = int(L["sample_offset"][i]), int(L["sample_len"][i]), int(L["prompt_len"][i]), int(L["target_len"][i])
            print(f"[{ALL_ISOS[int(L['iso_id'][i])]}] len={ln} prompt={pl} target={tl}\n  PROMPT: {tok.decode(toks[o:o+pl].tolist())[-300:]!r}\n  TARGET(loss): {tok.decode(toks[o+pl:o+pl+tl].tolist())[:300]!r}")
    print("AUDIT OK" if not bad else "AUDIT PROBLEMS", dict(bad))


def trans_packs_estimate(share_csv=None):
    """ESTIMATE of the translation packs per language (the real packs do not exist yet): kept pairs after the calibrated QE gate (work/gate/keep_rate_per_language.csv, column keep_cal, or the
    pairs before the cap x keep_rate_cal) x (2 x mean target tokens per pair + 30 template tokens) / (4096 x 0.9994); target tokens per pair from hub/specialist/stats/per_language_summary.csv."""
    if share_csv:
        return {r["iso"]: int(r["packs"]) for r in csv.DictReader(open(share_csv))}
    root = "/workspace/work"
    sp = list(csv.DictReader(open(root + "/hub/specialist/stats/per_language_summary.csv")))
    tpp = [float(r["gemma_target_tokens"]) / max(1, float(r["pairs"])) for r in sp if float(r["pairs"]) > 5000]
    mean_tpp = float(np.mean(tpp))
    per_lang_tpp = {r["iso"]: float(r["gemma_target_tokens"]) / max(1, float(r["pairs"])) for r in sp if float(r["pairs"]) > 0}
    out = {}
    for r in csv.DictReader(open(root + "/work/gate/keep_rate_per_language.csv")):
        n = float(r["keep_cal"] or 0) or float(r["n_pairs_before_cap"] or 0) * float(r["keep_rate_cal"] or 0)
        if n <= 0:
            continue
        t = per_lang_tpp.get(r["iso"], mean_tpp)
        out[r["iso"]] = int(math.ceil(n * (2 * t + 30) / (SEQ * 0.9994)))
    return out


def water_level(n, budget):
    """L such that sum_i clip(L, n_i, 4 n_i) = budget (a_i = clip(L, n_i, 4 n_i)); below min(n_i) it is not identifiable (every language at its 1-epoch floor): returns min(n_i) then"""
    n = np.asarray(n, np.float64)
    if budget <= n.sum():
        return int(n.min())
    lo, hi = 0.0, float((4 * n).max())
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.clip(mid, n, 4 * n).sum() > budget:
            hi = mid
        else:
            lo = mid
    return int(round(lo))


def schedule(a):
    from gemma_mt.data import packed_dataset as PD
    from gemma_mt.languages import ALL_ISOS
    d = os.path.join(a.packed, "train")
    iso, ps = np.load(d + "/iso_id.npy"), np.load(d + "/pack_start.npy").astype(np.int64)
    sl = np.load(d + "/sample_len.npy").astype(np.int64)
    pack_lang = iso[ps[:-1]].astype(np.int64)
    n_tok_pack = np.add.reduceat(sl, ps[:-1])
    counts = np.bincount(pack_lang, minlength=len(ALL_ISOS))
    live = np.flatnonzero(counts)
    tp = trans_packs_estimate(a.trans_packs_csv)
    ta = np.array([int(min(max(a.lam_trans, v), 4 * v)) for v in tp.values()])                       # translation allocation clip(L, n, 4n)
    T_packs = int(ta.sum()); T_tokens = T_packs * SEQ * 0.9994
    target = int(round(T_packs * a.share / a.per_step)) * a.per_step
    n = counts[live].astype(np.int64)
    floor = int(n.sum()); cap = int(4 * n.sum())
    info = dict(translation_languages=len(tp), translation_packs_est=T_packs, translation_tokens_est=int(T_tokens), target_packs=target, target_tokens=int(target * SEQ * 0.9994),
                instr_packs=int(floor), instr_tokens_1_epoch=int(n_tok_pack.sum()), feasible=bool(floor <= target <= cap))
    budget = target if info["feasible"] else (floor if target < floor else cap)
    budget = max(floor, int(math.ceil(budget / a.per_step)) * a.per_step)
    alloc = PD.unimax_allocation(n, budget, min_epochs=1.0, max_epochs=4.0)
    lam = water_level(n, budget)
    alt = {f"{sh:.2f}": {"budget_packs": int(round(T_packs * sh / a.per_step)) * a.per_step, "L": (water_level(n, int(round(T_packs * sh / a.per_step)) * a.per_step) if int(round(T_packs * sh / a.per_step)) * a.per_step >= floor else None)}
           for sh in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50)}
    info["alternatives_share_of_translation"] = alt
    info.update(budget_packs=int(budget), steps=int(budget // a.per_step), per_step=a.per_step, lambda_packs_per_language=lam, budget_tokens=int(budget * SEQ * 0.9994), share_of_translation=round(budget / max(1, T_packs), 4))
    os.makedirs(a.out, exist_ok=True)
    sched, stats = PD.unimax_schedule(pack_lang, budget // a.per_step, a.per_step, min_epochs=1.0, max_epochs=4.0, weights=None, seed=a.seed)
    np.save(a.out + "/schedule.npy", sched); np.save(a.out + "/pack_lang.npy", pack_lang)
    rows = []
    for lid, c, al in zip(live, n, alloc):
        toks = int(n_tok_pack[pack_lang == lid].sum())
        rows.append(dict(iso=ALL_ISOS[int(lid)], packs=int(c), tokens_1_epoch=toks, draws=int(al), epochs=round(al / c, 3), draw_tokens=int(toks * al / c)))
    with open(a.out + "/draws_per_language.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(sorted(rows, key=lambda r: -r["draws"]))
    sys.path.insert(0, HERE)
    import sampler as S
    S.write_sidecar(a.out, sched, pack_lang, ALL_ISOS, stage="instr-unimax", seed=a.seed, source=a.packed, projected=False, packed_structure_sha256=None)
    json.dump(info, open(a.out + "/meta.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


def interleave(a):
    """PROPOSAL (not launched): merge the translation schedule and the instruction schedule into ONE schedule over the extended pack set (translation packs 0..N_T-1, instruction packs N_T..).
    Each stream's draws carry the keys (k + u) / draws of the UniMax trick, so the mixture ratio (translation : instruction packs) is constant over the whole run and every evaluation
    interval sees both; steps keep 32 packs, a step may mix both streams."""
    T = np.load(os.path.join(a.trans, "schedule.npy")).reshape(-1).astype(np.int64)
    I = np.load(os.path.join(a.instr, "schedule.npy")).reshape(-1).astype(np.int64) + a.n_trans_packs
    rng = np.random.default_rng([a.seed, 7919])
    keys = np.concatenate([(np.arange(len(T)) + rng.random()) / len(T), (np.arange(len(I)) + rng.random()) / len(I)])
    allp = np.concatenate([T, I])
    out = allp[np.argsort(keys, kind="stable")]
    per_step = np.load(os.path.join(a.trans, "schedule.npy")).shape[1]
    steps = len(out) // per_step
    os.makedirs(a.out, exist_ok=True)
    np.save(a.out + "/schedule.npy", out[: steps * per_step].reshape(steps, per_step))
    print(f"interleaved {len(T)} translation + {len(I)} instruction draws -> {steps} steps x {per_step} (dropped {len(out) - steps * per_step} tail draws)")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd")
    p = sp.add_parser("build"); p.add_argument("--src", default="/workspace/work/instr"); p.add_argument("--out", default="/workspace/work/work/packed_instr")
    p.add_argument("--tokenizer", default="google/gemma-3-270m-it"); p.add_argument("--workers", type=int, default=24); p.set_defaults(fn=build)
    p = sp.add_parser("audit"); p.add_argument("--packed", required=True); p.add_argument("--n", type=int, default=20000); p.add_argument("--decode", type=int, default=3); p.set_defaults(fn=audit)
    p = sp.add_parser("schedule"); p.add_argument("--packed", default="/workspace/work/work/packed_instr"); p.add_argument("--out", default="/workspace/work/work/sched_instr")
    p.add_argument("--trans-packs-csv"); p.add_argument("--share", type=float, default=0.25); p.add_argument("--lam-trans", type=int, default=20000); p.add_argument("--per-step", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234); p.set_defaults(fn=schedule)
    p = sp.add_parser("interleave"); p.add_argument("--trans", required=True); p.add_argument("--instr", required=True); p.add_argument("--n-trans-packs", type=int, required=True)
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=1234); p.set_defaults(fn=interleave)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
