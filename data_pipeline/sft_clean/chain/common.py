"""Shared helpers of the post-GPU chain (qe_score -> gate -> windows -> pack -> sampler). CPU-testable; no GPU import at module level."""
import glob, hashlib, os, re, sys
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import xxhash

# --- window definitions: IDENTICAL to old_box_snapshot_2026-10-21/sft_build/gemma_windows.py
BINS = {"b64": (24, 65), "b256": (65, 257), "b512": (257, 513), "b1024": (513, 1025)}
CYCLE = ["b64", "b256", "b512", "b1024"]
TERM = re.compile(r"[.!?።…][\"'”’»)\]}]*\s*$")
SEQ_LEN = 4096
STAGE_A = dict(alpha=0.5, min_share=0.002, max_epochs=4.0, per_step=32)          # old run: 16,827 steps = one natural pass over all packs (gemma_sft_training_plan section 3)
STAGE_B = dict(min_epochs=1.0, max_epochs=2.0, priority_weight=4.0, per_step=32, budget_tokens=3_000_000_000, step_tokens=131_072,
               priority=["afr", "efi", "ewe", "fon", "hau", "ibo", "kin", "nya", "orm", "pcm", "sna", "som", "sot", "swh", "tsn", "twi", "wol", "xho", "yor", "zul"])
SPECIALIST_20 = "amh tir fon dyu kbp nus mos kam bam umb kea kmb vmw kab seh wol ewe luo tum yor".split()       # 20 languages, pre-registration Amendment 1
NEW_LANGS = "nde ful knc tzm".split()                                                                         # added by Amendment 6 (2026-10-30): SFT-only, built by extra_sources.py, appended after the 20 (ids in languages.ALL_ISOS are unchanged: the four were already registered)
SPECIALIST = SPECIALIST_20 + NEW_LANGS                                                                         # S1 / S2 train on these 24 (Amendment 6); G, W, C evaluation groups are unchanged


# --- disclosure tags (Amendment 6): windows carry owner_exception=<tag> for sources kept by an explicit owner decision (AfroLingu-MT, NLLB-mined Fulah, Bible-derived sets, TWB terms, ...)
OWNER_TAGS = [(re.compile(r"afrolingu", re.I), "afrolingu-mt"), (re.compile(r"^extra-nllb_mined"), "nllb_mined"), (re.compile(r"^extra-nllb_seed"), "nllb_seed"), (re.compile(r"^extra-bible"), "bible_derived"),
              (re.compile(r"^extra-twb"), "twb_gamayun_terms"), (re.compile(r"^extra-kanurimt"), "kanurimt_compilation")]


def owner_tag(source_dataset):
    """disclosure tag of the row's source ('' = none); matched on the full source_dataset string"""
    s_ = source_dataset or ""
    for rx, tag in OWNER_TAGS:
        if rx.search(s_):
            return tag
    return ""


def bin_of(n):
    for b in CYCLE:
        lo, hi = BINS[b]
        if n < hi:
            return b
    return "b1024"


def key_of(en, iso, tg):
    """xxh64 of (english casefold, iso, target casefold): the dedupe key of build_pairs.py (same value as the `key` column of out/kept)."""
    return xxhash.xxh64_intdigest((en.casefold() + "\x01" + iso + "\x01" + tg.casefold()).encode("utf-8", "surrogatepass"))


def shard_files(dirs):
    out = []
    for d in dirs:
        out += sorted(glob.glob(os.path.join(d, "*.parquet")))
    return out


def shard_id(path):
    """stable id of a shard file: <parent>__<name>"""
    return os.path.basename(os.path.dirname(path.rstrip("/"))) + "__" + os.path.basename(path)[:-8]


# --- commercial-use rule (Amendment 4, 2026-10-29): rows of licence class non_commercial are removed from every training and calibration role
NC_CLASS = "non_commercial"


def is_nc(licence):
    return licence == NC_CLASS


def assert_clean_src(dirs):
    """refuse directories that must never feed the chain (quarantine folders, raw rebuild shards with pre-dedupe duplicates)"""
    for d in dirs:
        low = d.lower().rstrip("/")
        if "quarantine" in low or low.endswith("/kept") or low.endswith("/meta"):
            raise SystemExit(f"{d}: not a chain input (quarantine/kept/meta); use data/, data_devhalf/, specialist/")


def parse_nc_allow(items):
    """--nc-allow repo[=tag] ...  ->  {repo: tag}; tag defaults to the last path component (howard-nlp/ibom-mt -> ibom-mt). Owner exceptions of 2026-10-29: howard-nlp/ibom-mt (efi, ibb)."""
    out = {}
    for it in items or []:
        repo, _, tag = it.partition("=")
        out[repo] = tag or repo.split("/")[-1]
    return out


def nc_exception_tag(source_dataset, allow):
    """tag of the owner exception a row's source falls under, else ''"""
    return allow.get((source_dataset or "").split(":")[0], "")


def nc_blocked(licence, source_dataset, allow):
    """True when the row must be removed: class non_commercial and its source is not on the allow-list"""
    return is_nc(licence) and not nc_exception_tag(source_dataset, allow)


# --- QE workload cap (2026-10-30, RTX 4500 Ada measured 129 pairs/s): a deterministic, MONOTONE per-language sample of the pairs to score.
# A pair is IN the cap when (key & 0xFFFFFFFF) < threshold[language]; threshold = min(2^32, ceil(N * 2^32 / count[language])) with count = the language's unique scoreable pairs, stored once
# (cap_counts.json). Raising N raises every threshold, so the pairs in the cap of a larger N are a superset (--skip-scored then scores only the new keys); a language with count <= N is in full.
CAP_M = 1 << 32


def cap_thresholds(counts, n):
    """{iso: threshold in [0, 2^32]}; n <= 0 means no cap"""
    if n <= 0:
        return None
    return {iso: min(CAP_M, -(-int(n) * CAP_M // max(int(c), 1))) for iso, c in counts.items()}


def in_cap(key, iso, thr):
    """True when the pair is inside the cap (thr None = no cap; a language missing from the counts is kept in full)"""
    return thr is None or (int(key) & (CAP_M - 1)) < thr.get(iso, CAP_M)


def cap_counts(src, exclude_nc=True, allow=()):
    """{iso: number of unique scoreable pairs} over the source directories (first directory wins for the language of a duplicated key; NC rows other than owner exceptions are not scoreable)"""
    seen_k, seen_i, names = [], [], {}
    for f in shard_files(src):
        pf = pq.ParquetFile(f)
        cols = [c for c in ("key", "english", "target", "iso", "licence", "source_dataset") if c in pf.schema_arrow.names]
        t = pf.read(columns=cols).to_pydict()
        n = len(t["iso"])
        k = np.array(t["key"], np.uint64) if "key" in t else np.array([key_of(t["english"][i], t["iso"][i], t["target"][i]) for i in range(n)], np.uint64)
        ok_ = np.ones(n, bool)
        if exclude_nc and "licence" in t:
            sd = t.get("source_dataset") or [""] * n
            ok_ = np.array([(x != NC_CLASS) or (s_.split(":")[0] in allow) for x, s_ in zip(t["licence"], sd)])
        seen_k.append(k[ok_]); seen_i.append(np.array([names.setdefault(x, len(names)) for x in np.array(t["iso"], dtype=object)[ok_]], np.int32))
    if not seen_k:
        return {}
    k, i = np.concatenate(seen_k), np.concatenate(seen_i)
    _, first = np.unique(k, return_index=True)
    inv = {v: u for u, v in names.items()}
    cnt = np.bincount(i[first], minlength=len(names))
    return {inv[j]: int(cnt[j]) for j in range(len(names))}
