#!/usr/bin/env python3
"""Rewrite a packed TRANSLATION dataset so that the language tag in front of every translation is ONE dedicated token (tag mode "single_token", gemma_mt/tags.py) instead of its 1-3
text pieces. Growth 1 (owner 2026-11-01).

    python retag_packs.py --src work/packed_full --out work/packed_full_tag [--tokenizer google/gemma-3-270m-it] [--splits train val] [--verify 200]

What changes: the tokens `tag pieces` at the start of each target span (position sample_offset + prompt_len) become the tag token id of the sample's language; target_len and sample_len
shrink by (pieces - 1); sample_offset is recomputed; tokens.u32.bin is rewritten. What does NOT change: the number and order of samples, pack_start (sample indices), iso_id,
direction, prompt_len, every other token (compared sample by sample in --verify), the language of every pack and therefore the sha256 of the pack structure (pack_start + iso_id + direction):
a schedule over pack indices (and its sidecar) stays valid. Packs only get shorter (<= 4,096 tokens, asserted).

Every sample must START with the text pieces of its tag (tokenizer(iso)); a sample that does not is an error (`--allow-mismatch N` tolerates up to N samples and leaves them as they are, which
mixes the formats: do not use it for training data). Do NOT run it on instruction packs (they carry no tag).
meta.json gets `tag_mode: "single_token"`, `tag_map_sha256` (iso -> pieces -> token id) and updated n_tokens / target_tokens statistics.
"""
import argparse, hashlib, json, os, shutil, sys, time
import numpy as np

NAMES = ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "direction", "pack_start")


def tag_maps(tokenizer, iso_table):
    """(pieces, token_id) per language of the table: the text pieces the tag had, and the single tag token that replaces them."""
    from gemma_mt import tags
    from gemma_mt.languages import ALL_ISOS
    assert list(iso_table) == list(ALL_ISOS[: len(iso_table)]), "the iso_table of the packs is not a prefix of gemma_mt.languages.ALL_ISOS"
    pieces, ids = [], []
    for iso in iso_table:
        p = [int(i) for i in (tokenizer(iso, add_special_tokens=False)["input_ids"])]
        t = int(tokenizer.convert_tokens_to_ids(tags.tag_string(iso, tokenizer)))
        unk = getattr(tokenizer, "unk_token_id", None)
        assert p and t != unk, (iso, p, t)
        pieces.append(p); ids.append(t)
    return pieces, ids


def tag_map_sha256(iso_table, pieces, ids):
    return hashlib.sha256(json.dumps([[i, p, t] for i, p, t in zip(iso_table, pieces, ids)], separators=(",", ":")).encode()).hexdigest()


def retag_split(src, dst, pieces, tag_ids, chunk_tokens=1 << 26, allow_mismatch=0, log=print):
    """One split directory (train/ or val/): arrays + tokens.u32.bin -> dst. Returns a statistics dict."""
    os.makedirs(dst, exist_ok=True)
    A = {n: np.load(os.path.join(src, n + ".npy")) for n in NAMES}
    so, sl, pl, tl, iso, ps = A["sample_offset"].astype(np.int64), A["sample_len"].astype(np.int64), A["prompt_len"].astype(np.int64), A["target_len"].astype(np.int64), A["iso_id"].astype(np.int64), A["pack_start"].astype(np.int64)
    n = len(sl)
    tok = np.memmap(os.path.join(src, "tokens.u32.bin"), dtype=np.uint32, mode="r")
    assert n == 0 or (so[0] == 0 and (so[1:] == so[:-1] + sl[:-1]).all() and so[-1] + sl[-1] == len(tok)), "samples are not contiguous in tokens.u32.bin"
    npieces = np.array([len(p) for p in pieces], dtype=np.int64); kmax = int(npieces.max())
    # 1. the target must start with the text pieces of its tag
    start = so + pl
    ok = np.ones(n, dtype=bool)
    P = np.full((len(pieces), kmax), -1, dtype=np.int64)
    for i, p in enumerate(pieces): P[i, : len(p)] = p
    for k in range(kmax):
        has = npieces[iso] > k
        idx = np.flatnonzero(has)
        got = tok[start[idx] + k].astype(np.int64)
        ok[idx] &= got == P[iso[idx], k]
    bad = np.flatnonzero(~ok)
    if len(bad) > allow_mismatch:
        by = np.bincount(iso[bad], minlength=len(pieces)); worst = np.argsort(-by)[:5]
        raise SystemExit(f"{len(bad)} of {n} samples do not start with the text pieces of their tag (worst languages: {[(int(w), int(by[w])) for w in worst]}); first sample {int(bad[0])}")
    shrink = np.where(ok, npieces[iso] - 1, 0)                 # tokens removed per sample (0 for a tolerated mismatch)
    # 2. deletions (the 2nd.. pieces) and replacements (the first piece -> the tag token), as sorted global token positions
    rep_pos = start[ok]; rep_val = np.asarray(tag_ids, dtype=np.int64)[iso[ok]].astype(np.uint32)
    dels = [start[ok & (npieces[iso] > k)] + k for k in range(1, kmax)]
    del_pos = np.sort(np.concatenate(dels)) if dels else np.zeros(0, dtype=np.int64)
    # 3. stream the tokens
    out_tok = os.path.join(dst, "tokens.u32.bin"); n_out = 0
    with open(out_tok, "wb") as f:
        for lo in range(0, len(tok), chunk_tokens):
            hi = min(len(tok), lo + chunk_tokens)
            buf = np.array(tok[lo:hi], dtype=np.uint32)
            a, b = np.searchsorted(rep_pos, [lo, hi]); buf[rep_pos[a:b] - lo] = rep_val[a:b]
            a, b = np.searchsorted(del_pos, [lo, hi])
            if b > a:
                keep = np.ones(hi - lo, dtype=bool); keep[del_pos[a:b] - lo] = False; buf = buf[keep]
            buf.tofile(f); n_out += len(buf)
    assert n_out == len(tok) - int(shrink.sum()), (n_out, len(tok), int(shrink.sum()))
    # 4. arrays
    new_sl = sl - shrink; new_tl = tl - shrink
    new_so = np.zeros(n, dtype=np.int64)
    if n:
        new_so[1:] = np.cumsum(new_sl)[:-1]
    assert n == 0 or (new_so[-1] + new_sl[-1] == n_out)
    sums = np.add.reduceat(new_sl, ps[:-1]) if len(ps) > 1 else np.zeros(0, dtype=np.int64)
    assert len(sums) == 0 or int(sums.max()) <= 4096, f"a pack has {int(sums.max())} tokens"
    assert int(ps[-1]) == n
    out = {n_: A[n_] for n_ in NAMES}
    out["sample_offset"] = new_so.astype(A["sample_offset"].dtype); out["sample_len"] = new_sl.astype(A["sample_len"].dtype); out["target_len"] = new_tl.astype(A["target_len"].dtype)
    for n_, v in out.items():
        np.save(os.path.join(dst, n_ + ".npy"), v)
    per_lang_target = np.bincount(iso, weights=new_tl, minlength=len(pieces)).astype(np.int64)
    log(f"  {os.path.basename(src)}: {n} samples, {len(ps) - 1} packs, tokens {len(tok)} -> {n_out} ({-100 * (n_out - len(tok)) / max(1, len(tok)):.2f}% fewer), tolerated mismatches {len(bad)}")
    return dict(n_samples=n, n_packs=len(ps) - 1, tokens_before=int(len(tok)), tokens_after=int(n_out), target_tokens=int(new_tl.sum()), per_language_target_tokens=per_lang_target, mismatches=int(len(bad)),
                max_pack_tokens=int(sums.max()) if len(sums) else 0)


def verify(src, dst, pieces, tag_ids, tokenizer, n_check, iso_table, log=print, seed=0):
    """Decode n_check random samples before and after; everything but the tag must be identical."""
    A = {n: np.load(os.path.join(src, n + ".npy")) for n in NAMES}; B = {n: np.load(os.path.join(dst, n + ".npy")) for n in NAMES}
    ta = np.memmap(os.path.join(src, "tokens.u32.bin"), dtype=np.uint32, mode="r"); tb = np.memmap(os.path.join(dst, "tokens.u32.bin"), dtype=np.uint32, mode="r")
    n = len(A["sample_len"]); rng = np.random.default_rng(seed)
    for n_name in ("iso_id", "direction", "pack_start", "prompt_len"):
        assert np.array_equal(A[n_name], B[n_name]), n_name
    shown = 0
    for i in sorted(rng.choice(n, size=min(n_check, n), replace=False).tolist()):
        a = ta[int(A["sample_offset"][i]): int(A["sample_offset"][i]) + int(A["sample_len"][i])].astype(np.int64)
        b = tb[int(B["sample_offset"][i]): int(B["sample_offset"][i]) + int(B["sample_len"][i])].astype(np.int64)
        k = int(A["iso_id"][i]); p = int(A["prompt_len"][i]); m = len(pieces[k])
        assert a[:p].tolist() == b[:p].tolist(), f"sample {i}: prompt differs"
        assert a[p: p + m].tolist() == pieces[k] and b[p] == tag_ids[k], f"sample {i}: tag"
        assert a[p + m:].tolist() == b[p + 1:].tolist(), f"sample {i}: the tokens after the tag differ"
        assert int(B["target_len"][i]) == int(A["target_len"][i]) - (m - 1) and int(B["sample_len"][i]) == len(b)
        if shown < 3:
            shown += 1
            log(f"  sample {i} [{iso_table[k]}] before: {tokenizer.decode(a[p: p + m + 4].tolist())!r}  after: {tokenizer.decode(b[p: p + 5].tolist())!r}")
    log(f"  verified {min(n_check, n)} random samples: prompt, tokens after the tag, lengths, iso/direction/pack_start identical")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tokenizer", default="google/gemma-3-270m-it")
    ap.add_argument("--splits", nargs="+", default=["train", "val"]); ap.add_argument("--verify", type=int, default=200); ap.add_argument("--allow-mismatch", type=int, default=0)
    ap.add_argument("--chunk-tokens", type=int, default=1 << 26); ap.add_argument("--fake-tokenizer", action="store_true", help="tests: the byte stand-in tokenizer")
    a = ap.parse_args(); t0 = time.time()
    sys.path.insert(0, os.environ.get("GEMMA_MT_PKG", "/workspace/work/code"))
    if a.fake_tokenizer:
        from gemma_mt.fake_tokenizer import FakeGemmaTokenizer; tok = FakeGemmaTokenizer()
    else:
        from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained(a.tokenizer, token=os.environ.get("HF_TOKEN"))
    meta = json.load(open(os.path.join(a.src, "meta.json")))
    assert meta.get("tag_mode", "text") == "text", "already retagged"
    iso_table = meta["iso_table"]; pieces, tag_ids = tag_maps(tok, iso_table)
    stats = {}
    for sp in a.splits:
        if not os.path.isdir(os.path.join(a.src, sp)): continue
        stats[sp] = retag_split(os.path.join(a.src, sp), os.path.join(a.out, sp), pieces, tag_ids, a.chunk_tokens, a.allow_mismatch)
        if a.verify: verify(os.path.join(a.src, sp), os.path.join(a.out, sp), pieces, tag_ids, tok, a.verify, iso_table)
    new = json.loads(json.dumps(meta))
    new["tag_mode"] = "single_token"; new["tag_map_sha256"] = tag_map_sha256(iso_table, pieces, tag_ids)
    new["tag_map"] = {iso: {"pieces": p, "token_id": t} for iso, p, t in zip(iso_table, pieces, tag_ids)}
    new["retag"] = {"source": os.path.abspath(a.src), "seconds": round(time.time() - t0, 1), "note": "tag pieces replaced by one tag token; n_tokens/target_tokens updated, packing_efficiency/padding_fraction NOT recomputed"}
    for sp, st in stats.items():
        if sp in new.get("splits", {}):
            s = new["splits"][sp]; s["n_tokens"] = st["tokens_after"]; s["target_tokens"] = st["target_tokens"]
            for iso, d in s.get("per_language", {}).items():
                d["target_tokens"] = int(st["per_language_target_tokens"][iso_table.index(iso)])
    os.makedirs(a.out, exist_ok=True); json.dump(new, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"retagged {list(stats)} -> {a.out} in {time.time() - t0:.0f}s; tag_map_sha256 {new['tag_map_sha256'][:16]}")


if __name__ == "__main__":
    main()
