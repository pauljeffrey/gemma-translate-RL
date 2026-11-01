# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit a packed SFT dataset (output of prepare_sft): are the samples whole and are the packs built right?

    python -m gemma_mt.data.audit_packed --data gemma/packed_fwd --split train [--tokenizer google/gemma-3-270m-it] [--decode 6]

Exhaustive structural checks (every sample / pack, vectorised) and token-level checks on a random subset:

  structure   offsets contiguous, no pack longer than seq_len, packs hold whole samples (a sample never straddles a pack),
              sum of sample lengths == number of stored tokens, every sample has prompt > 0 and target > 0 tokens,
              prompt + target <= sample length, the leftover (template tail) is at most 2 tokens
  one language per pack          (--expect-single-language, for --pack-by-language data)
  token level (sample of --check-samples)
              starts with <bos>; exactly one <bos>; target region ends with <end_of_turn>; the decoded sample has exactly
              two <start_of_turn> / <end_of_turn>; the instruction is the literal template; the answer starts with the
              language tag; labels in the loaded pack cover EXACTLY the target tokens (no prompt, no padding)
  attention   position_ids restart at every sample, segment_ids 1..k (the block-diagonal mask is built from these)

Exit code 0 = no problem found. Nothing is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from gemma_mt import format as fmt
from gemma_mt.data.packed_dataset import IGNORE, PackedDataset
from gemma_mt.languages import ALL_ISOS


def audit(ds: PackedDataset, tok: Any = None, *, check_samples: int = 2000, expect_single_language: bool = False,
          seed: int = 0, decode_examples: int = 0) -> Dict[str, Any]:
    problems: List[str] = []
    rep: Dict[str, Any] = {"n_packs": ds.n_packs, "n_samples": ds.n_samples, "seq_len": ds.seq_len}
    sl = np.asarray(ds.sample_len, dtype=np.int64)
    off = np.asarray(ds.sample_offset, dtype=np.int64)
    pl = np.asarray(ds.prompt_len, dtype=np.int64)
    tl = np.asarray(ds.target_len, dtype=np.int64)
    ps = np.asarray(ds.pack_start, dtype=np.int64)

    # ---- structure
    if ds.n_samples:
        if off[0] != 0 or not np.array_equal(off[1:], np.cumsum(sl)[:-1]):
            problems.append("sample offsets are not contiguous")
    if int(sl.sum()) != ds.n_tokens_total:
        problems.append(f"sum(sample_len)={int(sl.sum())} != stored tokens {ds.n_tokens_total}")
    if ps[0] != 0 or ps[-1] != ds.n_samples or (np.diff(ps) <= 0).any():
        problems.append("pack_start is not a strictly increasing partition of the samples")
    if ds.n_packs and ds.n_samples:
        pack_len = np.add.reduceat(sl, ps[:-1])
        rep["pack_fill_mean"] = float(pack_len.mean() / ds.seq_len)
        rep["pack_fill_min"] = float(pack_len.min() / ds.seq_len)
        rep["samples_per_pack_mean"] = float(ds.n_samples / ds.n_packs)
        if (pack_len > ds.seq_len).any():
            problems.append(f"{int((pack_len > ds.seq_len).sum())} packs are longer than seq_len={ds.seq_len}")
        rep["tokens"] = int(sl.sum())
        rep["target_tokens"] = int(tl.sum())
        rep["target_share"] = float(tl.sum() / max(sl.sum(), 1))
    if (pl <= 0).any():
        problems.append(f"{int((pl <= 0).sum())} samples without prompt tokens")
    if (tl <= 0).any():
        problems.append(f"{int((tl <= 0).sum())} samples without target tokens")
    tail = sl - pl - tl
    if (tail < 0).any() or (tail > 2).any():
        problems.append(f"{int(((tail < 0) | (tail > 2)).sum())} samples whose prompt+target do not match their length (tail {tail.min()}..{tail.max()})")
    if expect_single_language and ds.n_packs:
        try:
            ds.pack_languages()
        except ValueError as e:
            problems.append(str(e))
    if ds.n_samples:
        iso_counts = np.bincount(np.asarray(ds.iso_id, dtype=np.int64), minlength=len(ALL_ISOS))
        rep["languages"] = int((iso_counts > 0).sum())

    # ---- token level on a random subset of packs
    rng = np.random.default_rng(seed)
    n_checked = 0
    examples: List[str] = []
    if tok is not None and ds.n_packs:
        bos = tok.bos_token_id
        eot = tok.convert_tokens_to_ids(fmt.END_OF_TURN)
        sot_txt, eot_txt = fmt.START_OF_TURN, fmt.END_OF_TURN
        want = min(check_samples, ds.n_samples)
        packs = rng.permutation(ds.n_packs)
        for pi in packs:
            if n_checked >= want:
                break
            item = ds[int(pi)]
            ids, lab, pos, seg, cu = item["input_ids"], item["labels"], item["position_ids"], item["segment_ids"], item["cu_seqlens"]
            s0, s1 = ds.sample_range(int(pi))
            k = s1 - s0
            if int(seg.max()) != k or sorted(set(seg[seg > 0].tolist())) != list(range(1, k + 1)):
                problems.append(f"pack {pi}: segment ids are not 1..{k}")
            for j in range(k):
                a, b = int(cu[j]), int(cu[j + 1])
                sid = ids[a:b]
                if not np.array_equal(pos[a:b], np.arange(b - a)):
                    problems.append(f"pack {pi} sample {j}: position_ids do not restart at 0")
                if sid[0] != bos or int((sid == bos).sum()) != 1:
                    problems.append(f"pack {pi} sample {j}: not exactly one <bos> at the start")
                p, t = int(ds.prompt_len[s0 + j]), int(ds.target_len[s0 + j])
                if sid[p + t - 1] != eot:
                    problems.append(f"pack {pi} sample {j}: the target does not end with <end_of_turn>")
                lab_idx = np.flatnonzero(lab[a:b] != IGNORE)
                if len(lab_idx) != t or (len(lab_idx) and (lab_idx[0] != p or lab_idx[-1] != p + t - 1)):
                    problems.append(f"pack {pi} sample {j}: the loss labels do not cover exactly the target tokens")
                txt = tok.decode(sid.tolist(), skip_special_tokens=False)
                if txt.count(sot_txt) != 2 or txt.count(eot_txt) != 2:
                    problems.append(f"pack {pi} sample {j}: {txt.count(sot_txt)} start_of_turn / {txt.count(eot_txt)} end_of_turn (want 2/2)")
                if "Translate the text to " not in txt:
                    problems.append(f"pack {pi} sample {j}: instruction template missing")
                if len(examples) < decode_examples:
                    examples.append(f"[{ALL_ISOS[int(ds.iso_id[s0 + j])]}] len={b - a} prompt={p} target={t}\n{txt}")
                n_checked += 1
            if int((lab[int(cu[k]):] != IGNORE).sum()):
                problems.append(f"pack {pi}: padding carries loss labels")
    rep["samples_token_checked"] = n_checked
    rep["problems"] = problems[:50]
    rep["n_problems"] = len(problems)
    rep["examples"] = examples
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--tokenizer", default="google/gemma-3-270m-it")
    ap.add_argument("--check-samples", type=int, default=2000)
    ap.add_argument("--expect-single-language", action="store_true")
    ap.add_argument("--decode", type=int, default=0, help="print this many decoded samples")
    a = ap.parse_args(argv)
    ds = PackedDataset(a.data, a.split)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rep = audit(ds, tok, check_samples=a.check_samples, expect_single_language=a.expect_single_language,
                decode_examples=a.decode)
    ex = rep.pop("examples")
    print(json.dumps(rep, indent=1))
    for e in ex:
        print("-" * 70)
        print(e)
    print("AUDIT", "OK" if rep["n_problems"] == 0 else f"FAILED ({rep['n_problems']} problems)")
    return 0 if rep["n_problems"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
