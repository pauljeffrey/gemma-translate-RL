# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the packed SFT dataset.

    python -m gemma_mt.data.prepare_sft --input data/raw --out data/packed --seq-len 2048

Input rows (parquet / jsonl): `english`, `target`, `iso`  (+ optional `id`). See gemma_mt/data/io.py.

What happens, in order
  1. Every row is validated and normalised (iso alias -> canonical tag, whitespace stripped, empties dropped,
     unknown language dropped, exact duplicates dropped).
  2. The row goes to `train` or `val` by a HASH of its split key (default: the English text), so the same English
     sentence never lands in both splits, whatever the language, and the split does not depend on row order.
     Optionally a few extra keys per rare language are forced into `val` (`min_val_per_language`).
  3. `format.build_example` renders the chat template and tokenizes: prompt tokens carry label -100, the target
     `{iso} {translation}<end_of_turn>` carries the loss. Samples longer than `max_sample_tokens` are DROPPED
     (never truncated: a truncated target would teach the model to stop mid-sentence or never to stop).
     With `directions` containing "x2en" every pair ALSO yields the reverse sample
     (`Translate the text to English:  {african_text}` -> `eng {english}<end_of_turn>`); both samples of a pair
     get the same split (the split key does not depend on the direction), so a validation sentence is never
     trained on in the other direction. `x2en_fraction` keeps a deterministic random fraction of the reverse samples.
  4. Samples are packed into sequences of `seq_len` tokens with best-fit-decreasing inside shuffled buffers; a
     sample is never split across sequences.
  5. Written per split:
        tokens.u32.bin        flat uint32 token ids (Gemma's 262k vocab does not fit uint16), samples back to back
                              in pack order, NO padding stored
        sample_offset.npy     uint64  start of the sample in tokens.u32.bin
        sample_len.npy        uint32  total tokens of the sample (prompt + target + tail)
        prompt_len.npy        uint32  masked prompt tokens at the start of the sample
        target_len.npy        uint32  tokens that carry loss (`{iso} {translation}<end_of_turn>`)
        iso_id.npy            uint16  index into gemma_mt.languages.ALL_ISOS of the AFRICAN language of the pair
        direction.npy         uint8   0 = en2x, 1 = x2en (absent in older datasets = all en2x)
        pack_start.npy        uint64  n_packs + 1 entries: pack p owns samples pack_start[p] .. pack_start[p+1]
     and one `meta.json` (tokenizer name + fingerprint, counts, packing efficiency, per-language counts, config).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import os
import shutil
import sys
import time
from array import array
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from gemma_mt import format as fmt
from gemma_mt.data.io import iter_rows
from gemma_mt.data.packing import pack_buffered, packing_stats
from gemma_mt.languages import ALL_ISOS, canonical_iso, iso_to_id

DTYPE = "uint32"
FORMAT_VERSION = 1
SPLITS = ("train", "val")


@dataclass
class PrepareConfig:
    input: str
    out_dir: str
    tokenizer: str = "google/gemma-3-270m-it"
    tokenizer_revision: Optional[str] = None
    seq_len: int = 2048
    max_sample_tokens: Optional[int] = None      # default: seq_len (a sample must fit one pack)
    val_fraction: float = 0.01
    split_key: str = "english"                   # "english" | "id" | "pair"
    min_val_per_language: int = 0
    min_samples_for_forced_val: int = 200        # only force val keys for languages with at least this many rows
    seed: int = 0
    buffer_size: int = 100_000
    dedupe: bool = True
    directions: Tuple[str, ...] = ("en2x",)      # ("en2x", "x2en") = bidirectional
    x2en_fraction: float = 1.0                   # share of the pairs that also give a reverse sample
    decontaminate_isos: Tuple[str, ...] = ()     # languages whose pairs are reused in several windows (pcm x2): train windows that share an
                                                 # English sentence with a validation window of the same language are dropped
    workers: int = 0                             # >0: tokenise in this many processes (the real tokenizer only; ~1,000 windows/s per process)
    pack_by_language: bool = False               # every pack holds ONE language (needed by the language-stratified sampler, train_sft `language_alpha`)

    def __post_init__(self):
        self.directions = tuple(self.directions)
        self.decontaminate_isos = tuple(self.decontaminate_isos)
        if not self.directions or any(d not in fmt.DIRECTIONS for d in self.directions):
            raise ValueError(f"directions must be a non-empty subset of {fmt.DIRECTIONS}")
        if not 0.0 <= self.x2en_fraction <= 1.0:
            raise ValueError("x2en_fraction must be in [0, 1]")
        if self.max_sample_tokens is None:
            self.max_sample_tokens = self.seq_len
        if self.max_sample_tokens > self.seq_len:
            raise ValueError("max_sample_tokens cannot exceed seq_len (a sample must fit in one pack)")
        if self.split_key not in ("english", "id", "pair"):
            raise ValueError("split_key must be english | id | pair")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")


# ----------------------------------------------------------------------------------------------- row handling
def tokenizer_fingerprint(tok: Any) -> str:
    """Hash of everything that decides token ids: vocabulary, special ids, chat template. Changing any of them
    changes the fingerprint, so a packed dataset can be checked against the tokenizer used for training."""
    h = hashlib.sha256()
    vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
    for token, i in sorted(vocab.items(), key=lambda kv: (kv[1], kv[0])):
        h.update(f"{i}\t{token}\n".encode("utf-8"))
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        h.update(f"{name}={getattr(tok, name, None)}\n".encode())
    h.update(str(getattr(tok, "chat_template", "")).encode("utf-8"))
    return h.hexdigest()


def normalise_row(row: Dict[str, Any]) -> Tuple[Optional[Tuple[str, str, str, Optional[str]]], Optional[str]]:
    """-> ((english, target, iso, id), None) or (None, reason)."""
    en, tr, tag = row.get("english"), row.get("target"), row.get("iso")
    if not isinstance(en, str) or not isinstance(tr, str) or tag is None:
        return None, "missing_field"
    en, tr = en.strip(), tr.strip()
    if not en or not tr:
        return None, "empty_text"
    iso = canonical_iso(str(tag))
    if iso is None:
        return None, "unknown_iso"
    rid = row.get("id")
    return (en, tr, iso, None if rid is None else str(rid)), None


def split_fraction(key: str, seed: int) -> float:
    """Deterministic uniform number in [0, 1) from (seed, key)."""
    d = hashlib.blake2b(f"{seed}|{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "big") / 2.0 ** 64


def split_key_of(en: str, tr: str, iso: str, rid: Optional[str], mode: str) -> str:
    if mode == "english":
        return en
    if mode == "id":
        return rid if rid is not None else f"{iso}\x1f{en}\x1f{tr}"
    return f"{iso}\x1f{en}"                         # "pair": same English may be val in one language, train in another


def _pair_key(en: str, tr: str, iso: str) -> str:
    return f"{iso}\x1f{en}\x1f{tr}"


def _dedupe_hash(en: str, tr: str, iso: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{iso}\x1f{en}\x1f{tr}".encode("utf-8"), digest_size=8).digest(), "big")


def _clean_rows(cfg: PrepareConfig, dropped: Dict[str, int]) -> Iterator[Tuple[str, str, str, Optional[str]]]:
    seen: set = set()
    for row in iter_rows(cfg.input):
        norm, reason = normalise_row(row)
        if norm is None:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        if cfg.dedupe:
            h = _dedupe_hash(norm[0], norm[1], norm[2])
            if h in seen:
                dropped["duplicate"] = dropped.get("duplicate", 0) + 1
                continue
            seen.add(h)
        yield norm


def forced_val_keys(cfg: PrepareConfig) -> set:
    """Pass 1 (no tokenization): for languages with fewer than `min_val_per_language` hash-selected validation keys,
    add the keys with the smallest hash fraction. Deterministic, order independent."""
    if cfg.min_val_per_language <= 0:
        return set()
    k = cfg.min_val_per_language
    per_lang_count: Dict[str, int] = {}
    heaps: Dict[str, list] = {}                       # iso -> max-heap (negated u) of the k smallest (u, key)
    natural: Dict[str, int] = {}
    for en, tr, iso, rid in _clean_rows(cfg, {}):
        key = split_key_of(en, tr, iso, rid, cfg.split_key)
        u = split_fraction(key, cfg.seed)
        per_lang_count[iso] = per_lang_count.get(iso, 0) + 1
        if u < cfg.val_fraction:
            natural[iso] = natural.get(iso, 0) + 1
        h = heaps.setdefault(iso, [])
        item = (-u, key)
        if len(h) < k:
            heapq.heappush(h, item)
        elif item > h[0]:
            heapq.heapreplace(h, item)
    forced = set()
    for iso, h in heaps.items():
        if per_lang_count[iso] >= cfg.min_samples_for_forced_val and natural.get(iso, 0) < k:
            forced.update(key for _, key in h)
    return forced


# ----------------------------------------------------------------------------------------------- storage
class _SampleStore:
    """Append-only on-disk store of tokenized samples (before packing)."""

    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, "raw_tokens.u32")
        self._f = open(self.path, "wb")
        self.offset = array("Q")
        self.length = array("I")
        self.prompt = array("I")
        self.target = array("I")
        self.iso = array("H")
        self.direction = array("B")
        self._pos = 0

    def add(self, ex: fmt.Example, iso_id: int, direction: int = 0) -> None:
        arr = np.asarray(ex.input_ids, dtype=np.uint32)
        self._f.write(arr.tobytes())
        self.offset.append(self._pos)
        self.length.append(len(arr))
        self.prompt.append(ex.prompt_len)
        self.target.append(ex.target_len)
        self.iso.append(iso_id)
        self.direction.append(direction)
        self._pos += len(arr)

    def close(self) -> None:
        self._f.close()

    def __len__(self) -> int:
        return len(self.length)


def _write_split(store: _SampleStore, out_dir: str, cfg: PrepareConfig, split: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    store.close()
    n = len(store)
    lengths = np.frombuffer(store.length, dtype=np.uint32) if n else np.zeros(0, np.uint32)
    seed = cfg.seed + (0 if split == "train" else 1)
    if cfg.pack_by_language and n:
        iso_arr = np.frombuffer(store.iso, dtype=np.uint16)
        bins = []
        for lang in np.unique(iso_arr):                       # sorted language ids -> deterministic pack order
            idx = np.flatnonzero(iso_arr == lang)
            sub = pack_buffered(lengths[idx].tolist(), cfg.seq_len, cfg.buffer_size, seed + int(lang))
            bins += [[int(idx[j]) for j in b] for b in sub]
    else:
        bins = pack_buffered(lengths.tolist(), cfg.seq_len, cfg.buffer_size, seed)
    stats = packing_stats(lengths.tolist(), bins, cfg.seq_len)

    order = [i for b in bins for i in b]
    pack_start = np.zeros(len(bins) + 1, dtype=np.uint64)
    pack_start[1:] = np.cumsum([len(b) for b in bins], dtype=np.uint64)
    off_old = np.frombuffer(store.offset, dtype=np.uint64) if n else np.zeros(0, np.uint64)
    plen = np.frombuffer(store.prompt, dtype=np.uint32) if n else np.zeros(0, np.uint32)
    tlen = np.frombuffer(store.target, dtype=np.uint32) if n else np.zeros(0, np.uint32)
    iso = np.frombuffer(store.iso, dtype=np.uint16) if n else np.zeros(0, np.uint16)
    direction = np.frombuffer(store.direction, dtype=np.uint8) if n else np.zeros(0, np.uint8)

    new_len = lengths[order] if n else lengths
    new_off = np.zeros(len(order), dtype=np.uint64)
    if len(order):
        new_off[1:] = np.cumsum(new_len[:-1].astype(np.uint64))
    raw = np.memmap(store.path, dtype=np.uint32, mode="r") if n else None
    out_path = os.path.join(out_dir, "tokens.u32.bin")
    with open(out_path, "wb") as f:
        for i in order:
            o = int(off_old[i])
            f.write(np.asarray(raw[o: o + int(lengths[i])], dtype=np.uint32).tobytes())
    del raw
    np.save(os.path.join(out_dir, "sample_offset.npy"), new_off)
    np.save(os.path.join(out_dir, "sample_len.npy"), new_len.astype(np.uint32))
    np.save(os.path.join(out_dir, "prompt_len.npy"), plen[order].astype(np.uint32) if n else plen)
    np.save(os.path.join(out_dir, "target_len.npy"), tlen[order].astype(np.uint32) if n else tlen)
    np.save(os.path.join(out_dir, "iso_id.npy"), iso[order].astype(np.uint16) if n else iso)
    np.save(os.path.join(out_dir, "direction.npy"), direction[order].astype(np.uint8) if n else direction)
    np.save(os.path.join(out_dir, "pack_start.npy"), pack_start)
    os.remove(store.path)

    per_lang: Dict[str, Dict[str, int]] = {}
    per_dir = {d: {"samples": 0, "tokens": 0, "target_tokens": 0} for d in fmt.DIRECTIONS}
    for i in range(n):
        d = per_lang.setdefault(ALL_ISOS[int(iso[i])], {"samples": 0, "target_tokens": 0, "en2x": 0, "x2en": 0})
        dname = fmt.DIRECTIONS[int(direction[i])]
        d["samples"] += 1
        d["target_tokens"] += int(tlen[i])
        d[dname] += 1
        per_dir[dname]["samples"] += 1
        per_dir[dname]["tokens"] += int(lengths[i])
        per_dir[dname]["target_tokens"] += int(tlen[i])
    return {
        "n_samples": n, "n_packs": len(bins), "n_tokens": stats["n_tokens"], "target_tokens": int(tlen.sum()) if n else 0,
        "packing_efficiency": stats["efficiency"], "padding_fraction": stats["padding_fraction"],
        "mean_samples_per_pack": stats["mean_samples_per_bin"], "per_language": per_lang, "per_direction": per_dir,
    }


# ----------------------------------------------------------------------------------------------- main entry
def load_tokenizer(name: str, revision: Optional[str] = None):  # pragma: no cover - needs network / cache
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)


_W_TOK: Any = None


def _init_worker(name: str, revision: Optional[str]) -> None:        # pragma: no cover - runs in child processes
    global _W_TOK
    _W_TOK = load_tokenizer(name, revision)


def _tokenise_chunk(items: List[Tuple[str, str, str, str]]) -> List[Any]:   # pragma: no cover - runs in child processes
    """items: (english, target, iso, direction) -> Example, or the string "format_error" for each."""
    out: List[Any] = []
    for en, tr, iso, direction in items:
        src, tgt_tag, tgt = fmt.directed_pair(en, tr, iso, direction)
        try:
            out.append(fmt.build_example(src, tgt_tag, tgt, _W_TOK))
        except ValueError:
            out.append("format_error")
    return out


def prepare(cfg: PrepareConfig, tokenizer: Any = None) -> Dict[str, Any]:
    """Run the whole pipeline. `tokenizer` may be injected (tests use `FakeGemmaTokenizer`)."""
    t0 = time.time()
    tok = tokenizer if tokenizer is not None else load_tokenizer(cfg.tokenizer, cfg.tokenizer_revision)
    fmt.assert_template_matches_reference(tok)
    stop = fmt.resolve_stop_tokens(tok)
    warnings = list(stop.warnings)
    mism = fmt.check_segment_consistency(tok)
    if mism:
        warnings.append("joint-vs-segment tokenization differs on probes (training uses the segment form, which "
                        "matches inference): " + "; ".join(mism))

    os.makedirs(cfg.out_dir, exist_ok=True)
    tmp = os.path.join(cfg.out_dir, "_tmp_store")
    shutil.rmtree(tmp, ignore_errors=True)
    forced = forced_val_keys(cfg)
    dropped: Dict[str, int] = {}
    stores = {s: _SampleStore(os.path.join(tmp, s)) for s in SPLITS}
    n_rows = 0
    val_sentences: set = set()
    if cfg.decontaminate_isos:
        from gemma_mt.rl.segments import split_sentences

        scan_drop: Dict[str, int] = {}
        for en, tr, iso, rid in _clean_rows(cfg, scan_drop):
            if iso in cfg.decontaminate_isos:
                key = split_key_of(en, tr, iso, rid, cfg.split_key)
                if key in forced or split_fraction(key, cfg.seed) < cfg.val_fraction:
                    val_sentences.update((iso, x.strip().casefold()) for x in split_sentences(en))
    def _consume(ex, iso, d_i, direction, is_val):
        if isinstance(ex, str):
            k = "format_error" if direction == fmt.EN2X else "format_error_x2en"
            dropped[k] = dropped.get(k, 0) + 1
            return
        if ex.n_tokens > cfg.max_sample_tokens:
            k = "too_long" if direction == fmt.EN2X else "too_long_x2en"
            dropped[k] = dropped.get(k, 0) + 1
            return
        stores["val" if is_val else "train"].add(ex, iso_to_id(iso), d_i)

    pool = None
    pend_items: List[Tuple[str, str, str, str]] = []
    pend_meta: List[Tuple[str, int, str, bool]] = []
    inflight: "collections.deque" = collections.deque()
    if cfg.workers > 0 and tokenizer is None:
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(max_workers=cfg.workers, initializer=_init_worker,
                                   initargs=(cfg.tokenizer, cfg.tokenizer_revision))

    def _drain(limit: int) -> None:
        while len(inflight) >= limit:
            fut, metas = inflight.popleft()
            for ex, (iso_, d_i_, dir_, val_) in zip(fut.result(), metas):
                _consume(ex, iso_, d_i_, dir_, val_)

    def _submit() -> None:
        if not pend_items:
            return
        inflight.append((pool.submit(_tokenise_chunk, list(pend_items)), list(pend_meta)))
        pend_items.clear()
        pend_meta.clear()
        _drain(3 * cfg.workers)                           # bounded memory: results are consumed in submission order

    for en, tr, iso, rid in _clean_rows(cfg, dropped):
        n_rows += 1
        key = split_key_of(en, tr, iso, rid, cfg.split_key)
        is_val = key in forced or split_fraction(key, cfg.seed) < cfg.val_fraction
        if val_sentences and not is_val and iso in cfg.decontaminate_isos:
            from gemma_mt.rl.segments import split_sentences

            if any((iso, x.strip().casefold()) in val_sentences for x in split_sentences(en)):
                dropped["train_overlaps_val"] = dropped.get("train_overlaps_val", 0) + 1
                continue
        for d_i, direction in enumerate(fmt.DIRECTIONS):
            if direction not in cfg.directions:
                continue
            if (direction == fmt.X2EN and cfg.x2en_fraction < 1.0
                    and split_fraction("x2en|" + _pair_key(en, tr, iso), cfg.seed) >= cfg.x2en_fraction):
                dropped["x2en_not_sampled"] = dropped.get("x2en_not_sampled", 0) + 1
                continue
            if pool is None:
                src, tgt_tag, tgt = fmt.directed_pair(en, tr, iso, direction)
                try:
                    ex = fmt.build_example(src, tgt_tag, tgt, tok)
                except ValueError:
                    ex = "format_error"
                _consume(ex, iso, d_i, direction, is_val)
            else:
                pend_items.append((en, tr, iso, direction))
                pend_meta.append((iso, d_i, direction, is_val))
                if len(pend_items) >= 2000:
                    _submit()
    if pool is not None:
        _submit()
        while inflight:
            _drain(1)
        pool.shutdown()

    split_meta = {}
    for s in SPLITS:
        split_meta[s] = _write_split(stores[s], os.path.join(cfg.out_dir, s), cfg, s)
    shutil.rmtree(tmp, ignore_errors=True)

    meta = {
        "format_version": FORMAT_VERSION,
        "dtype": DTYPE,
        "seq_len": cfg.seq_len,
        "instruction_template": fmt.INSTRUCTION_TEMPLATE,
        "end_of_turn": fmt.END_OF_TURN,
        "ignore_index": fmt.IGNORE_INDEX,
        "tokenizer": {
            "name": getattr(tok, "name_or_path", cfg.tokenizer),
            "requested": cfg.tokenizer,
            "revision": cfg.tokenizer_revision,
            "fingerprint": tokenizer_fingerprint(tok),
            "vocab_size": len(tok),
            "bos_id": tok.bos_token_id, "eos_id": tok.eos_token_id, "pad_id": tok.pad_token_id,
            "end_of_turn_id": stop.end_of_turn_id,
        },
        "iso_table": list(ALL_ISOS),
        "directions": list(cfg.directions),
        "pivot_tag": fmt.PIVOT_ISO,
        "rows_after_cleaning": n_rows,
        "dropped": dropped,
        "splits": split_meta,
        "config": asdict(cfg),
        "warnings": warnings,
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(cfg.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="file or directory of .parquet/.jsonl (columns english,target,iso)")
    ap.add_argument("--out", required=True, help="output directory (train/, val/, meta.json)")
    ap.add_argument("--tokenizer", default="google/gemma-3-270m-it")
    ap.add_argument("--tokenizer-revision", default=None)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-sample-tokens", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--split-key", default="english", choices=["english", "id", "pair"])
    ap.add_argument("--min-val-per-language", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buffer-size", type=int, default=100_000)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--directions", nargs="+", default=["en2x"], choices=list(fmt.DIRECTIONS),
                    help="en2x = English -> African; x2en = African -> English (reverse sample of every pair)")
    ap.add_argument("--x2en-fraction", type=float, default=1.0, help="share of the pairs that also give a reverse sample")
    ap.add_argument("--workers", type=int, default=0, help="tokenisation processes (0 = in this process)")
    ap.add_argument("--decontaminate", nargs="*", default=[], metavar="ISO",
                    help="languages whose pairs appear in several windows (e.g. pcm): drop train windows sharing an English sentence with validation")
    ap.add_argument("--pack-by-language", action="store_true",
                    help="pack each language separately (one language per pack) so training can sample languages in proportion")
    a = ap.parse_args(argv)
    cfg = PrepareConfig(input=a.input, out_dir=a.out, tokenizer=a.tokenizer, tokenizer_revision=a.tokenizer_revision,
                        seq_len=a.seq_len, max_sample_tokens=a.max_sample_tokens, val_fraction=a.val_fraction,
                        split_key=a.split_key, min_val_per_language=a.min_val_per_language, seed=a.seed,
                        buffer_size=a.buffer_size, dedupe=not a.no_dedupe, directions=tuple(a.directions),
                        x2en_fraction=a.x2en_fraction, pack_by_language=a.pack_by_language,
                        decontaminate_isos=tuple(a.decontaminate), workers=a.workers)
    meta = prepare(cfg)
    for s in SPLITS:
        m = meta["splits"][s]
        print(f"{s:5s}: {m['n_samples']:>9,} samples  {m['n_packs']:>8,} packs  {m['n_tokens']:>12,} tokens  "
              f"({m['target_tokens']:,} target)  packing efficiency {m['packing_efficiency']:.3f}  "
              + "  ".join(f"{d}={v['samples']:,}" for d, v in m["per_direction"].items() if v["samples"]))
    print("dropped:", meta["dropped"] or "nothing")
    for w in meta["warnings"]:
        print("WARNING:", w, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
