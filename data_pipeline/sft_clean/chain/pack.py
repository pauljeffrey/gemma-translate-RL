#!/usr/bin/env python3
"""STEP (d): 4,096-token, one-language packs with the SAME chat format and loss masking as the old run (gemma_mt.data.prepare_sft, unchanged).

    python pack.py prepare --windows windows_sft --out packed --tokenizer google/gemma-3-270m-it [--workers 5]
        = the exact old command (old_box_snapshot_2026-10-21/sft_build/fwd_chain.sh):
          python -m gemma_mt.data.prepare_sft --input <windows> --out <out> --seq-len 4096 --max-sample-tokens 4096 --directions en2x --pack-by-language
                 --val-fraction 0.002 --min-val-per-language 60 --decontaminate pcm --workers 5
          followed by gemma_mt.data.audit_packed (train and val, 4000 samples, --expect-single-language, 3 decoded examples)
    python pack.py replay --like packed_S1 --out packed_S2 --tokenizer <extended tokenizer dir>
        = S2 of the pre-registration: every pack of an existing packed dataset is re-tokenised with another tokenizer, SAME packs, SAME order, SAME samples (token counts change). The
          prompt and the target of each sample are decoded with the source tokenizer and re-encoded by gemma_mt.format.build_example with the new one (same template, same loss mask).
          A pack can only shrink when the new tokenizer is a superset that merges tokens; replay checks every pack still fits seq_len.
    python pack.py selftest   # replay with the SAME tokenizer must reproduce tokens.u32.bin byte for byte
"""
import argparse, json, os, re, subprocess, sys
import numpy as np


def _pkg():
    for p in (os.environ.get("GEMMA_MT_PKG"), "/workspace/work/gemma_mt_pkg", "/path/to/repo/gemma_mt"):
        if p and os.path.isdir(os.path.join(p, "gemma_mt")):
            if p not in sys.path: sys.path.insert(0, p)
            return p
    raise SystemExit("gemma_mt package not found (set GEMMA_MT_PKG)")


def prepare(a):
    pkg = _pkg()
    env = dict(os.environ, PYTHONPATH=pkg + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cmd = [sys.executable, "-m", "gemma_mt.data.prepare_sft", "--input", a.windows, "--out", a.out, "--tokenizer", a.tokenizer, "--seq-len", "4096", "--max-sample-tokens", "4096",
           "--directions", "en2x", "--pack-by-language", "--val-fraction", str(a.val_fraction), "--min-val-per-language", "60", "--decontaminate", "pcm", "--workers", str(a.workers)]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env)
    if not a.no_audit:
        for sp in ("train", "val"):
            subprocess.check_call([sys.executable, "-m", "gemma_mt.data.audit_packed", "--data", a.out, "--split", sp, "--check-samples", "4000", "--expect-single-language", "--decode", "3"], env=env)


def _arrays(d):
    g = lambda n: np.load(os.path.join(d, n), mmap_mode="r")
    return {k: g(k + ".npy") for k in ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "direction", "pack_start")}


def replay(a):
    _pkg()
    from transformers import AutoTokenizer
    from gemma_mt import format as fmt
    from gemma_mt.languages import ALL_ISOS
    meta = json.load(open(os.path.join(a.like, "meta.json")))
    src_tok = AutoTokenizer.from_pretrained(meta["tokenizer"]["requested"], use_fast=True)
    new_tok = AutoTokenizer.from_pretrained(a.tokenizer, use_fast=True)
    seq_len = meta["seq_len"]
    os.makedirs(a.out, exist_ok=True)
    splits = [s for s in ("train", "val") if os.path.isdir(os.path.join(a.like, s))]
    pat = re.compile(r"^<bos><start_of_turn>user\nTranslate the text to [^:\n]+:  (.*)<end_of_turn>\n<start_of_turn>model\n$", re.S)
    report = {}
    for sp in splits:
        d = os.path.join(a.like, sp); od = os.path.join(a.out, sp); os.makedirs(od, exist_ok=True)
        A = _arrays(d); toks = np.memmap(os.path.join(d, "tokens.u32.bin"), dtype=np.uint32, mode="r")
        n = len(A["sample_len"]); out_tokens = open(os.path.join(od, "tokens.u32.bin"), "wb")
        new_len = np.zeros(n, np.uint32); new_pl = np.zeros(n, np.uint32); new_tl = np.zeros(n, np.uint32)
        bad = 0; pos = 0; new_off = np.zeros(n, np.uint64)
        for i in range(n):
            o, L, pl, tl = int(A["sample_offset"][i]), int(A["sample_len"][i]), int(A["prompt_len"][i]), int(A["target_len"][i])
            ids = toks[o:o + L]
            ptxt = src_tok.decode(ids[:pl].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
            ttxt = src_tok.decode(ids[pl:pl + tl].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
            m = pat.match(ptxt)
            iso = ALL_ISOS[int(A["iso_id"][i])]
            if not m or not ttxt.endswith("<end_of_turn>") or not ttxt.startswith(iso + " "):
                bad += 1; arr = np.asarray(ids, np.uint32); npl, ntl = pl, tl                 # keep the sample unchanged and count it
            else:
                ex = fmt.build_example(m.group(1), iso, ttxt[len(iso) + 1:-len("<end_of_turn>")], new_tok)
                arr = np.asarray(ex.input_ids, np.uint32); npl, ntl = ex.prompt_len, ex.target_len
            out_tokens.write(arr.tobytes()); new_off[i] = pos; pos += len(arr); new_len[i] = len(arr); new_pl[i] = npl; new_tl[i] = ntl
        out_tokens.close()
        ps = np.asarray(A["pack_start"]); over = 0
        for p in range(len(ps) - 1):
            if int(new_len[int(ps[p]):int(ps[p + 1])].sum()) > seq_len:
                over += 1
        for nme, arr in (("sample_offset", new_off), ("sample_len", new_len), ("prompt_len", new_pl), ("target_len", new_tl)):
            np.save(os.path.join(od, nme + ".npy"), arr)
        for nme in ("iso_id", "direction", "pack_start"):
            np.save(os.path.join(od, nme + ".npy"), np.asarray(A[nme]))
        report[sp] = {"samples": n, "undecodable_kept_unchanged": bad, "packs_over_seq_len": over, "tokens_old": int(np.asarray(A["sample_len"]).sum()), "tokens_new": int(new_len.sum()),
                      "target_tokens_old": int(np.asarray(A["target_len"]).sum()), "target_tokens_new": int(new_tl.sum())}
    from gemma_mt.data.prepare_sft import tokenizer_fingerprint
    # the trainer refuses packs whose recorded fingerprint differs from the tokenizer it loads, so record the NEW tokenizer's (T1 finding, 2026-10-28)
    meta2 = dict(meta); meta2["tokenizer"] = dict(meta["tokenizer"], requested=a.tokenizer, replayed_from=a.like, fingerprint=tokenizer_fingerprint(new_tok))
    if new_tok.pad_token_id is not None and "pad_id" in meta["tokenizer"]:
        meta2["tokenizer"]["pad_id"] = int(new_tok.pad_token_id)
    meta2["replay_report"] = report
    json.dump(meta2, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))
    if any(r["packs_over_seq_len"] for r in report.values()):
        raise SystemExit("some packs exceed seq_len after replay: the new tokenizer lengthens text")


def selftest(a):
    replay(argparse.Namespace(like=a.like, out=a.out, tokenizer=a.tokenizer))
    for sp in ("train", "val"):
        p1, p2 = os.path.join(a.like, sp), os.path.join(a.out, sp)
        if os.path.isdir(p1):
            assert open(os.path.join(p1, "tokens.u32.bin"), "rb").read() == open(os.path.join(p2, "tokens.u32.bin"), "rb").read(), f"{sp}: replay with the same tokenizer is not identical"
    print("selftest OK: replay with the same tokenizer reproduces the packed tokens exactly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--windows", required=True); p.add_argument("--out", required=True); p.add_argument("--tokenizer", default="google/gemma-3-270m-it")
    p.add_argument("--workers", type=int, default=5); p.add_argument("--val-fraction", type=float, default=0.002); p.add_argument("--no-audit", action="store_true")
    r = sub.add_parser("replay"); r.add_argument("--like", required=True); r.add_argument("--out", required=True); r.add_argument("--tokenizer", required=True)
    t = sub.add_parser("selftest"); t.add_argument("--like", required=True); t.add_argument("--out", required=True); t.add_argument("--tokenizer", default="google/gemma-3-270m-it")
    a = ap.parse_args()
    {"prepare": prepare, "replay": replay, "selftest": selftest}[a.cmd](a)
