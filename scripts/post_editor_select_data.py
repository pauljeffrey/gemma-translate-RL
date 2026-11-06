#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-editor, step 1: choose the HUMAN-translated en->X training pairs from the clean pair shards (references = human translations only).

    python scripts/post_editor_select_data.py --input s3://<S3_BUCKET>/<S3_PREFIX>/backup/hub --subdirs data specialist extra extra2k \
        --afriscience afriscience_clean_train.parquet --bench-dir /workspace/eval_385m/bench/test_sets --chars-per-token chars_per_token.json \
        --out OUT --budget 48000 --cap-per-lang 1400 [--protected-index I.npz --protected-manifest M.json]

`--input` is a local directory of parquet shards (columns english, target, iso, source_dataset, source_family, licence, [kind, decon_flag]) or `s3://bucket/prefix` (shards are
downloaded ONE AT A TIME to --tmp and deleted; boto3 + keys S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY, or AK / SK, or /root/.bucket_env). Nothing is kept on disk but the outputs.

Source policy (SOURCE_RULES below, first match on `source_family|source_dataset`): every source gets a class
    gen  human translation, known non-religious provenance (government, news, Wikipedia tools, UI strings, Tatoeba, science: AfriScience-MT)
    rel  human translation, religious text (JW300 / Bible: all of MT560, Helsinki fon, luo-opus, Bible sets, Tanzil)         -> capped to <= --max-rel-share of a language
    unv  human, provenance UNVERIFIED or aggregator (multi-open, OPUS-100, crowd sets of unknown origin)                       -> only fills languages that would otherwise be thin
    X    never: NLLB / mined / crawled / machine-translated English side / synthetic / benchmark dev or test / AfroLingu-MT (contains MAFAND train) / TranslatePsy / unknown source
Short pairs (< --short-words English words: UI strings, Tatoeba) are capped at --max-short-share. Flores/NTREX/MAFAND... text is excluded again by --bench-dir (exact match on
either side), and by the protected index when given (R1-R3 of decon.py); the shards already passed protected-2026-10-28.3.

Outputs (OUT): pairs.jsonl (split train|val, one pair per line), availability_by_language.csv (what exists), selection_by_language.csv (what was chosen), availability_by_source.csv,
selection_summary.json. Deterministic (hash order; no RNG state).
"""
import argparse, csv, glob, json, os, re, sys, time, unicodedata
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))

# --------------------------------------------------------------------------------------------- source policy
# (regex on "source_family|source_dataset", class, register, non_commercial_by_rule, note). FIRST match wins; unmatched = X.
SOURCE_RULES = [
    (r"flores_plus:dev:|flores-plus-dev-half", "gen", "flores_dev_train", False, "FLORES+ DEV TRAIN HALF (440 ids; in the SFT on purpose, owner 2026-11-03: allowed). NOT devtest, NOT the held-out dev half (data_devhalf holds the train half only)"),
    (r"flores_plus|flores-plus", "X", "benchmark", False, "FLORES+ devtest / held-out dev: benchmark text"),
    (r"African-Code-Switch", "X", "synthetic", False, "Gemma-4 translation of English"),
    (r"extra-nllb_seed|nllb-seed|oldi_seed", "gen", "wiki", False, "NLLB-Seed / OLDI seed: professionally human-translated Wikipedia sentences (not NLLB-mined); disclose the name"),
    (r"nllb|NLLB", "X", "mined", False, "NLLB-mined bitext"),
    (r"finetranslations", "X", "synthetic_en", False, "model-written English side"),
    (r"dochplt|DocHPLT|paracrawl|mined_web", "X", "mined", False, "crawl-mined"),
    (r"XLEnt|xlent", "X", "mined", False, "entity phrases mined from Wikipedia"),
    (r"afrolingu", "X", "aggregate", False, "AfroLingu-MT contains MAFAND train pairs (owner: MAFAND train NOT allowed)"),
    (r"AfricanFineTranslations|polynews", "X", "unclear", False, "provenance of the translations unclear"),
    (r"igbo_to_english_split|igbo-to-english|Amazon_Translator", "X", "quality", False, "English side does not translate the Igbo / machine translator output"),
    (r"nupe", "X", "unclear", False, "Quran-translation sentences, Nupe licence unknown"),
    (r"JW300|mehgoss-jworg|mt560|luo-opus|fon-helsinki|bible|ebible|Tanzil", "rel", "religious", False, "JW300 / Bible / Quran text (human, religious register)"),
    (r"ibom-mt", "gen", "religious_mixed", True, "IbomMT train (CC-BY-NC, owner exception): efi/ibb; human"),
    (r"AfriDocMT", "gen", "health_tech", True, "AfriDocMT sentence level (CC-BY-NC): professional translators, health/tech"),
    (r"TED2020", "gen", "talks", True, "TED subtitles (non-commercial)"),
    (r"vukuzenzele|mehgoss-vukuzenzele|autshumato|mehgoss-autshumato|kin-gazette|NMT_Rwandan-Gazette|govza|GoURMET|GlobalVoices", "gen", "gov_news", False, "government / news, human"),
    (r"pontoon|weblate|kabyle-tm|translatewiki|travis_foundation|gamayun", "gen", "ui_localisation", False, "software localisation / community translation; short strings"),
    (r"opus_wikimedia|opus-wikimedia|wikimedia", "gen", "wiki", False, "Wikipedia content translation, human"),
    (r"tatoeba", "gen", "crowd_short", False, "Tatoeba crowd translations; short sentences"),
    (r"ssa-mte|SSA-MTE", "gen", "news", False, "SSA-MTE TRAIN split (human references, CC-BY-SA); benchmark-adjacent, train split only"),
    (r"salt-train|Sunbird/salt", "gen", "salt_text", False, "SALT TRAIN split (Sunbird, human translators, CC-BY-SA-4.0); the SALT benchmark test split is protected"),
    (r"infopankki", "gen", "gov_info", False, "Infopankki municipal information (OPUS), human, CC-BY-4.0"),
    (r"chichewa-mt|chichewa-machine-translation", "gen", "agri_health", False, "Chichewa topic sentences (agriculture / health), human, Apache-2.0"),
    (r"multi-open|opus100|opus-100|okwu|swh-rogendo|xhosa-silvano|silvanosolutions|lug-pkyoyetera|luganda_english|wolof-sudoping|luhya|kweku|kanurimt|igbo-bundles|Tommy0201|iamwille|ccibeekeoc42",
     "unv", "unverified", False, "human but provenance unstated / aggregator / crowd of unknown origin"),
    (r"^afriscience", "sci", "science", False, "AfriScience-MT train (professional translators, Apache-2.0, simplified science summaries)"),
]
_RULES = [(re.compile(p, re.I), c, reg, nc, note) for p, c, reg, nc, note in SOURCE_RULES]
_CACHE = {}


def classify(source_family, source_dataset, licence=None):
    key = (str(source_family), str(source_dataset), str(licence))
    if key not in _CACHE:
        hay = f"{source_family}|{source_dataset}"
        res = ("X", "unknown_source", False, "source not in the policy table")
        for rx, c, reg, nc, note in _RULES:
            if rx.search(hay):
                res = (c, reg, nc, note); break
        c, reg, nc, note = res
        if str(licence) == "non_commercial":
            nc = True
        _CACHE[key] = (c, reg, bool(nc), note)
    return _CACHE[key]


# --------------------------------------------------------------------------------------------- text filters
_URL = re.compile(r"https?://|www\.|@\w+\.|<[a-z/][^>]*>|\{\w*\}|%\w|&\w+;", re.I)
_FUNC = re.compile(r"\b(the|and|of|to|in|is|are|was|were|for|that|with|a|an|it|he|she|we|you|they|this|not|be|have|will|can|i|my|your|our)\b", re.I)


_DETOK = [(re.compile(r"\s+([,.;:!?%)\]}])"), r"\1"), (re.compile(r"([(\[{])\s+"), r"\1"), (re.compile(r"\s+(['\u2019]s)\b"), r"\1"), (re.compile(r"\s{2,}"), " ")]


def detok(s):
    """OPUS / MT560 text is tokenised ('word , word .'): undo the spacing before punctuation and after opening brackets so the editor does not learn to write tokenised text."""
    for rx, rep in _DETOK:
        s = rx.sub(rep, s)
    return s.strip()


def norm_text(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", s).split())


def row_ok(en, tg, min_words, max_words, max_chars):
    """Python-level checks on the few rows that survive the vectorised filters."""
    if _URL.search(en) or _URL.search(tg):
        return False
    w = en.split()
    if not (min_words <= len(w) <= max_words) or len(en) > max_chars or len(tg) < 8 or len(tg) > 2 * max_chars:
        return False
    if norm_text(en) == norm_text(tg):
        return False
    if sum(c.isalpha() for c in en) < 0.6 * len(en.replace(" ", "")) or sum(c.isalpha() for c in tg) < 0.5 * len(tg.replace(" ", "")):
        return False
    if len(w) >= 6 and not _FUNC.search(en):                     # the English side must look English
        return False
    r = len(tg) / max(1, len(en))
    return 0.3 <= r <= 4.0


def load_bench_strings(root, min_tokens=3):
    """Normalised sentences (>= min_tokens tokens) of every benchmark file under `root` (jsonl / json / txt / tsv; every string value counts)."""
    out = set()

    def add(x):
        if isinstance(x, str):
            n = norm_text(x)
            if len(n.split()) >= min_tokens:
                out.add(n)
        elif isinstance(x, dict):
            for v in x.values(): add(v)
        elif isinstance(x, list):
            for v in x: add(v)
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            try:
                if f.endswith((".jsonl", ".json")):
                    txt = open(p, encoding="utf-8").read()
                    try:
                        add(json.loads(txt))
                    except json.JSONDecodeError:
                        for line in txt.splitlines():
                            if line.strip():
                                try: add(json.loads(line))
                                except json.JSONDecodeError: pass
                elif f.endswith((".txt", ".tsv", ".csv")):
                    for line in open(p, encoding="utf-8"):
                        for part in line.rstrip("\n").split("\t"): add(part)
            except (OSError, UnicodeDecodeError):
                continue
    return out


# --------------------------------------------------------------------------------------------- shard iteration
def _s3_client():
    sys.path.insert(0, "/workspace/s3lib")
    import boto3
    env = dict(os.environ)
    if os.path.exists("/root/.bucket_env"):
        for line in open("/root/.bucket_env"):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1); env.setdefault(k, v)
    ak, sk = env.get("S3_ACCESS_KEY_ID") or env.get("AK"), env.get("S3_SECRET_ACCESS_KEY") or env.get("SK")
    if not ak or not sk:
        sys.exit("no bucket keys (S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY or AK / SK)")
    return boto3.client("s3", endpoint_url=env.get("S3_ENDPOINT") or os.environ["S3_ENDPOINT_URL"], aws_access_key_id=ak, aws_secret_access_key=sk, region_name=os.environ.get("S3_REGION", "us-east-1"))


def iter_shards(inp, subdirs, tmp, sl=(0, 1), skip=None):
    """yield (relative name, local path, cleanup callable)."""
    if inp.startswith("s3://"):
        bucket, _, prefix = inp[5:].partition("/")
        s3 = _s3_client(); os.makedirs(tmp, exist_ok=True)
        for sd in subdirs:
            keys = [o["Key"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix.rstrip('/')}/{sd}/")
                    for o in page.get("Contents", []) if o["Key"].endswith(".parquet")]
            keys = [k for k in keys if not (skip and re.search(skip, k))]
            for n_k, k in enumerate(sorted(keys)):
                if n_k % sl[1] != sl[0]: continue
                local = os.path.join(tmp, "shard.parquet")
                s3.download_file(bucket, k, local)
                yield k[len(prefix.rstrip("/")) + 1:], local, (lambda p=local: os.remove(p))
    else:
        for sd in subdirs:
            files = [p for p in sorted(glob.glob(os.path.join(inp, sd, "**", "*.parquet"), recursive=True)) if not (skip and re.search(skip, p))]
            for n_p, p in enumerate(files):
                if n_p % sl[1] != sl[0]: continue
                yield os.path.relpath(p, inp), p, (lambda: None)


# --------------------------------------------------------------------------------------------- allocation (pure functions, unit-tested)
def plan_language(avail, T, max_rel_share=0.25, max_short_share=0.25, rel_only_cap=300, min_nonrel=100, unv_floor=None, max_unv_share=None, min_total=0, max_sci_share=0.40):
    """How many pairs of each (class, short) pool to take for one language at target size T. `avail` = {(cls, short): n}. Returns {(cls, short): k}.
    Owner rule (2026-11-03): the editor set is NOT built from religious text. A language that has non-religious human data (gen + unv >= min_nonrel pairs) is drawn from THAT: long gen first,
    then short gen (<= max_short_share of T), then unverified-provenance pairs; religious pairs are added ONLY to complete T and only up to max_rel_share of the final total
    (a language with little non-religious text therefore ends up smaller than T, not religious). A language with less than min_nonrel non-religious pairs is
    RELIGIOUS-ONLY (flagged in the coverage table): all its non-religious pairs plus religious pairs up to rel_only_cap. (unv_floor / max_unv_share / min_total are accepted and ignored: old names.)"""
    g = lambda c, s: int(avail.get((c, s), 0))
    T = int(T); take = defaultdict(int)
    nonrel_all = g("gen", False) + g("gen", True) + g("unv", False) + g("unv", True) + g("sci", False) + g("sci", True)
    if nonrel_all < min_nonrel:                                   # religious-only language
        for key in (("gen", False), ("gen", True), ("unv", False), ("unv", True), ("sci", False), ("sci", True)):
            take[key] = g(*key)
        left = max(0, min(int(rel_only_cap), T) - sum(take.values()))
        for key in (("rel", False), ("rel", True)):
            k = min(g(*key), left); take[key] = k; left -= k
        return {k: v for k, v in take.items() if v > 0}
    short_cap = int(max_short_share * T); short_used = 0
    sci_reserve = min(g("sci", False) + g("sci", True), int(max_sci_share * T))      # AfriScience-MT text (unseen by the SFT / RL models: realistic drafts): at most max_sci_share of the language's rows
    rem = T - sci_reserve
    def grab(c, s, limit):
        nonlocal rem, short_used
        k = max(0, min(g(c, s), limit, rem))
        if s: k = max(0, min(k, short_cap - short_used))
        take[(c, s)] += k; rem -= k; short_used += k if s else 0
    grab("gen", False, rem); grab("gen", True, rem); grab("unv", False, rem); grab("unv", True, rem)
    other = sum(take.values())
    rem = T - other
    sci_room = min(sci_reserve, int(other * max_sci_share / (1.0 - max_sci_share))) if other else 0                  # keeps the register mixed: science <= 40% of (other + science)
    grab("sci", False, sci_room); grab("sci", True, max(0, sci_room - take[("sci", False)]))
    nonrel = sum(take.values())
    rel_room = max(0, min(T - nonrel, int(nonrel * max_rel_share / (1.0 - max_rel_share))))        # religious <= max_rel_share of the final total, and only to complete T
    for key in (("rel", False), ("rel", True)):
        if key[1] and short_used + take[key] >= short_cap: continue
        k = min(g(*key), rel_room); take[key] += k; rel_room -= k
    return {k: v for k, v in take.items() if v > 0}


def allocate(avail_by_lang, budget, cap, floor, cap_override=None, **plan_kw):
    """UniMax-style: effective availability E_l = size of the plan at T = cap_l; languages in `cap_override` {iso: cap} (e.g. the 6 AfriScience languages at 3,000) are RESERVED their E_l first; the rest of the
    budget is water-filled over the other languages (cap per language, tail languages with <= floor keep all); then the plan is recomputed at the target of each language. Returns {iso: {(cls, short): k}}."""
    from gemma_mt.post_edit import unimax_water_fill
    cap_override = cap_override or {}
    eff = {l: sum(plan_language(a, cap_override.get(l, cap), **plan_kw).values()) for l, a in avail_by_lang.items()}
    targets = {l: eff[l] for l in cap_override if l in eff}
    rest = {l: v for l, v in eff.items() if l not in targets}
    targets.update(unimax_water_fill(rest, max(0, budget - sum(targets.values())), cap, floor))
    return {l: plan_language(avail_by_lang[l], t, **plan_kw) for l, t in targets.items()}


# --------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True); ap.add_argument("--subdirs", nargs="+", default=["data", "specialist", "extra", "extra2k"])
    ap.add_argument("--afriscience", default=None, help="afriscience_clean_train.parquet (source, target, target_lang)")
    ap.add_argument("--out", required=True); ap.add_argument("--tmp", default="/tmp/pe_select_tmp")
    ap.add_argument("--budget", type=int, default=48000, help="total pairs (train + val)"); ap.add_argument("--cap-per-lang", type=int, default=1400)
    ap.add_argument("--floor", type=int, default=250, help="a language with <= floor eligible pairs keeps all of them")
    ap.add_argument("--val-min", type=int, default=8); ap.add_argument("--val-max", type=int, default=30); ap.add_argument("--val-share", type=float, default=0.04)
    ap.add_argument("--max-rel-share", type=float, default=0.25, help="religious share cap per language WHERE non-religious human data exists"); ap.add_argument("--max-overall-rel-share", type=float, default=0.30); ap.add_argument("--cap-override", default="", help="iso,iso,...=cap e.g. amh,hau,lug,nso,yor,zul=3000 (the AfriScience-MT languages)"); ap.add_argument("--max-sci-share", type=float, default=0.40); ap.add_argument("--val-sci", type=int, default=300, help="validation pairs of AfriScience text per language (unseen in-domain validation); the same English sentences in every language"); ap.add_argument("--min-nonrel", type=int, default=100, help="fewer non-religious human pairs than this = religious-only language"); ap.add_argument("--max-short-share", type=float, default=0.25); ap.add_argument("--rel-only-cap", type=int, default=300)
    ap.add_argument("--min-pairs", type=int, default=250, help="language floor: every language with enough human pairs gets at least this many (religious text may exceed its share cap to reach it)")
    ap.add_argument("--unv-floor", type=int, default=600); ap.add_argument("--max-unv-share", type=float, default=0.35)
    ap.add_argument("--short-words", type=int, default=8); ap.add_argument("--min-words", type=int, default=5); ap.add_argument("--max-words", type=int, default=48); ap.add_argument("--max-chars", type=int, default=360)
    ap.add_argument("--reservoir", type=int, default=5000, help="candidates kept per (language, class, short)")
    ap.add_argument("--no-prefer-unseen", dest="prefer_unseen", action="store_false", help="by default pairs that A0 / the stage-A checkpoint never saw (stage-B-new sources, AfriScience) are taken first")
    ap.add_argument("--no-flores-dev", action="store_true", help="also drop the FLORES+ dev TRAIN half rows (the policy table admits them since the owner allowed them on 2026-11-03)")
    ap.add_argument("--no-nc", action="store_true", help="drop non-commercial sources (IbomMT, AfriDocMT, TED, Tanzil); also applied to loaded states")
    ap.add_argument("--exclude-ids", default=None, help="file with one pair_id per line (or a pairs.jsonl): these pairs are never selected (e.g. the pairs the editor E1 was trained on)")
    ap.add_argument("--isos", nargs="*", default=None, help="restrict to these languages (default: all tags of the table except the pivot)")
    ap.add_argument("--chars-per-token", default=None, help="chars_per_token.json (target chars per Gemma token per language) for the token estimates")
    ap.add_argument("--bench-dir", default=None); ap.add_argument("--protected-index", default=None); ap.add_argument("--protected-manifest", default=None)
    ap.add_argument("--max-shards", type=int, default=0, help="debug: stop after this many shards")
    ap.add_argument("--skip-regex", default=r"nllb|aya_aligned|dochplt|paracrawl|extra2_hold", help="shard paths matching this are never downloaded (their rows are excluded by the policy anyway)")
    ap.add_argument("--profile-csv", default=None, help="profile.csv of the whole bucket (counts per source incl. the excluded ones): fills the 'why no human pairs' column of coverage_rl.csv")
    ap.add_argument("--slice", default=None, help="i/K: process every K-th shard (parallel workers); with --save-state the state is pickled and the run stops before selection")
    ap.add_argument("--save-state", default=None); ap.add_argument("--load-states", nargs="*", default=None, help="merge the pickled states of the slices and select (no shard is read)")
    a = ap.parse_args(argv)
    import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
    from gemma_mt.languages import ALL_ISOS, PIVOT_ISO, canonical_iso
    from gemma_mt.post_edit import stable_unit
    isos = set(a.isos) if a.isos else {i for i in ALL_ISOS if i != PIVOT_ISO}
    os.makedirs(a.out, exist_ok=True)
    cpt = json.load(open(a.chars_per_token)) if a.chars_per_token else {}
    t0 = time.time()
    res = defaultdict(lambda: pd.DataFrame())                     # (iso, cls, short) -> reservoir
    tally = defaultdict(lambda: [0, 0.0, 0.0])                    # (iso, cls, reg, family, nc, short) -> n, en words, tgt chars
    keyparts = []                                                 # (iso code, cls code, key) of survivors for distinct counts
    dropped = defaultdict(int)
    excl = defaultdict(int)                                       # (iso, source_family, reason) -> rows the policy excluded (the "why has this language no human pairs" table)
    cls_code = {"gen": 0, "rel": 1, "unv": 2, "sci": 3}
    ISO_ID = {i: n for n, i in enumerate(ALL_ISOS)}

    def process(df, rel_name, sft_input):
        """df columns: english, target, iso, source_dataset, source_family, licence (+ decon_flag)."""
        df = df[df["iso"].isin(isos)]
        if df.empty: return
        trip = df[["source_family", "source_dataset", "licence"]].drop_duplicates()
        cmap = {tuple(r): classify(*r) for r in trip.itertuples(index=False, name=None)}
        info = [cmap[(f, d, l)] for f, d, l in zip(df["source_family"], df["source_dataset"], df["licence"])]
        df = df.assign(cls=[i[0] for i in info], reg=[i[1] for i in info], nc=[i[2] for i in info])
        for c, n in df["cls"].value_counts().items():
            if c == "X": dropped["policy_excluded"] += int(n)
        if (df["cls"] == "X").any():
            xdf = df[df["cls"] == "X"]
            for (iso_, fam_, reg_), n_ in xdf.groupby(["iso", "source_family", "reg"]).size().items(): excl[(iso_, fam_, reg_)] += int(n_)
        df = df[df["cls"] != "X"]
        if a.no_nc: df = df[~df["nc"]]
        if a.no_flores_dev: df = df[df["reg"] != "flores_dev_train"]       # coordinator brief: nothing from benchmark dev (set this to ignore the 2026-11-03 FLORES+ dev-train-half allowance)
        if "decon_flag" in df.columns: df = df[(df["decon_flag"].fillna("") == "")]
        if df.empty: return
        en, tg = pa.array(df["english"].tolist(), pa.string()), pa.array(df["target"].tolist(), pa.string())
        ew = pc.list_value_length(pc.utf8_split_whitespace(en)).to_numpy(zero_copy_only=False)
        el, tl = pc.utf8_length(en).to_numpy(zero_copy_only=False), pc.utf8_length(tg).to_numpy(zero_copy_only=False)
        m = (ew >= a.min_words) & (ew <= a.max_words) & (el <= a.max_chars) & (tl >= 8) & (tl <= 2 * a.max_chars) & (tl / np.maximum(el, 1) >= 0.3) & (tl / np.maximum(el, 1) <= 4.0)
        dropped["length_ratio"] += int((~m).sum())
        df = df[m].assign(ew=ew[m], tl=tl[m])
        if df.empty: return
        ok = [row_ok(e, t, a.min_words, a.max_words, a.max_chars) for e, t in zip(df["english"], df["target"])]
        dropped["row_filters"] += int(len(ok) - sum(ok))
        df = df[np.array(ok, dtype=bool)]
        if df.empty: return
        stage = "unseen" if not sft_input else ("A0-era" if (rel_name.split("/")[0] in ("data", "specialist", "data_devhalf") or "efik_nt" in rel_name) else "stageB-new")
        key = pd.util.hash_array((df["iso"] + "\x1f" + df["english"].map(norm_text)).to_numpy(dtype=object))
        rank = 0 if (stage != "A0-era" or not a.prefer_unseen) else 1                  # unseen / stage-B-new pairs first: A0 and the stage-A checkpoint never saw them (their drafts there are unseen-quality)
        df = df.assign(short=(df["ew"] < a.short_words).to_numpy(), key=key, okey=(key >> np.uint64(2)) + np.uint64(rank) * np.uint64(1 << 62), sft_input=sft_input, stage=stage)
        for (iso, cls, reg, fam, nc, short), g in df.groupby(["iso", "cls", "reg", "source_family", "nc", "short"]):
            t = tally[(iso, cls, reg, fam, bool(nc), bool(short))]; t[0] += len(g); t[1] += float(g["ew"].sum()); t[2] += float(g["tl"].sum())
        keyparts.append((df["iso"].map(ISO_ID).to_numpy(np.int32), df["cls"].map(cls_code).to_numpy(np.int8), df["key"].to_numpy(np.uint64)))
        cols = ["english", "target", "iso", "cls", "reg", "nc", "source_family", "source_dataset", "licence", "key", "okey", "short", "sft_input", "stage", "ew", "tl"]
        for (iso, cls, short), g in df.groupby(["iso", "cls", "short"]):
            k = (iso, cls, bool(short))
            cur = res[k]
            res[k] = pd.concat([cur, g[cols]]).sort_values("okey").drop_duplicates("key").head(a.reservoir)

    import pickle
    if a.load_states:
        for pth in a.load_states:
            st = pickle.load(open(pth, "rb"))
            parts = {}
            for k, v in st["res"].items():                          # states written before the class "sci" existed hold AfriScience rows as class gen / register science: re-label them
                if k[1] == "gen" and "science" in set(v["reg"]):
                    m_ = (v["reg"] == "science").to_numpy()
                    parts[k] = v[~m_]; parts[(k[0], "sci", k[2])] = v[m_].assign(cls="sci")
                else:
                    parts[k] = v
            for k, v in parts.items():
                if len(v) == 0: continue
                res[k] = pd.concat([res[k], v]).sort_values("okey").drop_duplicates("key").head(a.reservoir) if k in res else v
            for k, v in st["tally"].items():
                if k[2] == "science" and k[1] == "gen": k = (k[0], "sci") + tuple(k[2:])
                t = tally[k]; t[0] += v[0]; t[1] += v[1]; t[2] += v[2]
            keyparts.extend(st["keyparts"])
            for k, v in st["dropped"].items(): dropped[k] += v
            for k, v in st["excl"].items(): excl[k] += v
        print(f"merged {len(a.load_states)} states", flush=True)
        excl_ids = set()
        if a.exclude_ids:
            for l in open(a.exclude_ids, encoding="utf-8"):
                l = l.strip()
                if l: excl_ids.add(json.loads(l)["pair_id"] if l.startswith("{") else l)
        n_ex = n_nc = 0
        for k in list(res):
            v = res[k]; n0 = len(v)
            if a.no_nc: v = v[~v["nc"].astype(bool)]; n_nc += n0 - len(v)
            if excl_ids:
                ids = [f"{i}-{int(k):016x}" for i, k in zip(v["iso"], v["key"])]
                m_ = np.array([i not in excl_ids for i in ids], dtype=bool); n_ex += int((~m_).sum()); v = v[m_]
            res[k] = v
        print(f"merge filters: {n_nc} non-commercial and {n_ex} excluded-id candidates removed", flush=True)
    else:
        si, sk = (int(x) for x in a.slice.split("/")) if a.slice else (0, 1)
        n_sh = 0
        for n_file, (rel_name, path, cleanup) in enumerate(iter_shards(a.input, a.subdirs, a.tmp, (si, sk), a.skip_regex)):
            pf = pq.ParquetFile(path)
            cols = [c for c in ("english", "target", "iso", "source_dataset", "source_family", "licence", "decon_flag") if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=250000, columns=cols):
                df = batch.to_pandas()
                for c in ("source_family", "source_dataset", "licence"):
                    if c not in df.columns: df[c] = ""
                process(df, rel_name, not any(k in rel_name for k in ("salt_train", "infopankki", "chichewa_mt")))      # the v3 extras (SALT train, infopankki, chichewa) were in no SFT build
            cleanup(); n_sh += 1
            print(f"[{time.time()-t0:6.0f}s] {rel_name} done", flush=True)
            if a.max_shards and n_sh >= a.max_shards: break
        if a.afriscience and si == 0:
            d = pq.read_table(a.afriscience).to_pandas()
            d = pd.DataFrame({"english": d["source"], "target": d["target"], "iso": [canonical_iso(x) or x for x in d["target_lang"]], "source_dataset": "afriscience_mt:" + d["domain"].astype(str),
                              "source_family": "afriscience", "licence": d["licence"], "decon_flag": ""})
            process(d, "afriscience", False)
        if a.save_state:
            pickle.dump({"res": dict(res), "tally": dict(tally), "keyparts": keyparts, "dropped": dict(dropped), "excl": dict(excl)}, open(a.save_state, "wb"))
            print("state saved", a.save_state, flush=True); return

    # ---- distinct counts (cross-source duplicates removed) and availability tables
    if keyparts:
        I = np.concatenate([k[0] for k in keyparts]); C = np.concatenate([k[1] for k in keyparts]); K = np.concatenate([k[2] for k in keyparts])
        allk = np.rec.fromarrays([I, C, K], names="i,c,k"); uniq = np.unique(allk)
        distinct = defaultdict(int)
        for i, c in zip(uniq["i"], uniq["c"]): distinct[(ALL_ISOS[int(i)], int(c))] += 1
    else:
        distinct = {}
    inv = {v: k for k, v in cls_code.items()}
    with open(os.path.join(a.out, "availability_by_source.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "class", "register", "source_family", "non_commercial", "short", "pairs_gross", "en_words_mean", "tgt_chars_mean"])
        for (iso, cls, reg, fam, nc, short), (n, ew, tl) in sorted(tally.items()):
            w.writerow([iso, cls, reg, fam, int(nc), int(short), n, round(ew / n, 2), round(tl / n, 1)])
    avail = defaultdict(dict)
    for (iso, cls, short), df in res.items(): avail[iso][(cls, short)] = len(df)
    # the reservoir holds at most --reservoir per pool; the exact (gross) counts are in availability_by_source.csv
    cap_over = {}
    if a.cap_override:
        isos_, _, c_ = a.cap_override.partition('='); cap_over = {i: int(c_) for i in isos_.split(',') if i}
    rel_only_cap = a.rel_only_cap
    while True:
        plan = allocate(avail, a.budget, a.cap_per_lang, a.floor, cap_override=cap_over, max_rel_share=a.max_rel_share, max_short_share=a.max_short_share, rel_only_cap=rel_only_cap, min_nonrel=a.min_nonrel, max_sci_share=a.max_sci_share)
        tot = sum(sum(p.values()) for p in plan.values()); rel = sum(v for p in plan.values() for (c, sh), v in p.items() if c == "rel")
        print(f"plan: {tot} pairs, religious {rel} = {rel/max(1,tot):.3f} (rel_only_cap {rel_only_cap})", flush=True)
        if rel / max(1, tot) <= a.max_overall_rel_share or rel_only_cap <= 100: break
        rel_only_cap = int(rel_only_cap * 0.85)
    a.rel_only_cap_used = rel_only_cap

    # ---- take rows (hash order), second pass of exclusions on the chosen rows only
    bench = load_bench_strings(a.bench_dir) if a.bench_dir else None
    pix = None
    if a.protected_index:
        sys.path.insert(0, os.path.join(HERE, "..", "data_pipeline", "sft_clean"))
        import protected_index_loader as PL
        pix, _ = PL.load(a.protected_index, a.protected_manifest, verify=bool(a.protected_manifest))
    chosen, seen_en = [], set()
    removed = defaultdict(int)
    for iso, pl in sorted(plan.items()):
        for (cls, short), k in sorted(pl.items()):
            pool = res[(iso, cls, short)].sort_values("okey")
            got = 0
            for r in pool.itertuples(index=False):
                if got >= k: break
                ne = norm_text(r.english)
                if (iso, ne) in seen_en: continue
                if bench is not None and (ne in bench or norm_text(r.target) in bench):
                    removed["bench_dir_exact"] += 1; continue
                d_ = r._asdict() | {"iso": iso}
                if d_["reg"] in ("religious", "religious_mixed", "unverified", "gov_news", "crowd_short"):          # tokenised OPUS-style spacing occurs in these sources
                    d_["english"], d_["target"] = detok(d_["english"]), detok(d_["target"])
                chosen.append(d_); seen_en.add((iso, ne)); got += 1
    if pix is not None:
        flagged = set()
        rows = [{"i": i, "english": c["english"], "target": c["target"]} for i, c in enumerate(chosen)]
        for row, hits in PL.check_rows(pix, rows, ["english", "target"]):
            if any(hits): flagged.add(row["i"])
        removed["protected_index"] += len(flagged)
        chosen = [c for i, c in enumerate(chosen) if i not in flagged]

    # ---- split: val = smallest hash fraction per language; train excludes every English present in val (any language)
    by_iso = defaultdict(list)
    for c in chosen: by_iso[c["iso"]].append(c)
    val_en = set()
    sci_val_en = None
    if a.val_sci:                                                  # the SAME English sentences (smallest hash) are the science validation of every AfriScience language: unseen in-domain validation, never in training
        cnt = defaultdict(int)
        for c in chosen:
            if c["cls"] == "sci": cnt[norm_text(c["english"])] += 1
        common = sorted((e for e, n in cnt.items() if n >= 2), key=lambda e: stable_unit("scival", e))[: a.val_sci]
        sci_val_en = set(common)
        for c in chosen:
            if c["cls"] == "sci" and norm_text(c["english"]) in sci_val_en:
                c["split"] = "val"; val_en.add(norm_text(c["english"]))
    for iso, rows in by_iso.items():
        rows = [c for c in rows if c.get("split") != "val"] if sci_val_en else rows
        kv = max(a.val_min, min(a.val_max, int(round(a.val_share * len(rows))))) if len(rows) >= 3 * a.val_min else 0
        for c in sorted(rows, key=lambda c: stable_unit("val", norm_text(c["english"]))) [:kv]:
            c["split"] = "val"; val_en.add(norm_text(c["english"]))
    out = []
    for c in chosen:
        if c.get("split") != "val":
            c["split"] = "train"
            if norm_text(c["english"]) in val_en: removed["train_english_in_val"] += 1; continue
        out.append(c)
    with open(os.path.join(a.out, "pairs.jsonl"), "w", encoding="utf-8") as f:
        for n, c in enumerate(sorted(out, key=lambda c: (c["iso"], c["split"], int(c["key"])))):
            rec = {"pair_id": f"{c['iso']}-{int(c['key']):016x}", "iso": c["iso"], "english": c["english"], "reference": c["target"], "split": c["split"], "cls": c["cls"], "register": c["reg"],
                   "source_family": c["source_family"], "source_dataset": c["source_dataset"], "licence": c["licence"], "non_commercial": bool(c["nc"]), "short": bool(c["short"]),
                   "sft_input": bool(c["sft_input"]), "sft_stage": c["stage"], "en_words": int(c["ew"]), "domain": "science" if c["reg"] == "science" else ""}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- tables
    sel = defaultdict(lambda: defaultdict(int)); words = defaultdict(float); tcha = defaultdict(float)
    for c in out:
        s = sel[c["iso"]]; s["n"] += 1; s[c["cls"]] += 1; s["short"] += int(c["short"]); s["nc"] += int(c["nc"]); s["val"] += int(c["split"] == "val"); s["unseen"] += int(not c["sft_input"]); s["sci_val"] += int(c["cls"] == "sci" and c["split"] == "val")
        s["science"] += int(c["reg"] == "science"); s["stageB_new"] += int(c["stage"] == "stageB-new"); words[c["iso"]] += c["ew"]; tcha[c["iso"]] += c["tl"]
    gross = defaultdict(lambda: defaultdict(int)); lic = defaultdict(set)
    for (iso, cls, reg, fam, nc, short), (n, ew, tl) in tally.items():
        gross[iso][cls] += n; lic[iso].add(fam + (" [NC]" if nc else ""))
    with open(os.path.join(a.out, "availability_by_language.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "gen_gross", "rel_gross", "unv_gross", "sci_gross", "gen_distinct", "rel_distinct", "unv_distinct", "sci_distinct", "sources"])
        for iso in sorted(gross):
            w.writerow([iso, gross[iso]["gen"], gross[iso]["rel"], gross[iso]["unv"], gross[iso]["sci"], distinct.get((iso, 0), 0), distinct.get((iso, 1), 0), distinct.get((iso, 2), 0), distinct.get((iso, 3), 0), "; ".join(sorted(lic[iso]))])
    with open(os.path.join(a.out, "selection_by_language.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "chosen", "val", "gen", "rel", "unv", "rel_share", "short_share", "nc", "afriscience", "unseen_by_sft", "stageB_new_pairs", "en_words_mean", "est_tokens_en_plus_target"])
        for iso in sorted(sel):
            s = sel[iso]; n = s["n"]; tok = words[iso] * 1.5 + tcha[iso] / float(cpt.get(iso, 3.0))
            w.writerow([iso, n, s["val"], s["gen"], s["rel"], s["unv"], round(s["rel"] / n, 3), round(s["short"] / n, 3), s["nc"], s["science"], s["unseen"], s["stageB_new"], round(words[iso] / n, 1), int(tok)])

    # ---- coverage table: every language of the table (RL languages first), what exists, what was chosen, why not
    from gemma_mt.languages import LANGUAGES, SUPPORTED_RL
    try:
        from gemma_mt.eval import benchmarks as BM
        bnames = ("flores_plus", "ntrex", "mafand", "smol_sent", "salt", "glot_story", "ibom_mt", "flores_moz")
        def bench_refs(iso):
            out = []
            for b in bnames:
                try:
                    if BM.code_for(b, iso): out.append(b)
                except Exception:
                    pass
            return out
    except Exception:
        bench_refs = lambda iso: []
    prof_excl = defaultdict(dict)
    if a.profile_csv and os.path.exists(a.profile_csv):
        pdf = pd.read_csv(a.profile_csv)
        for (iso_, fam_), n_ in pdf.groupby(["iso", "source_family"]).n.sum().items():
            if classify(fam_, "", "")[0] == "X" or classify(fam_, fam_, "")[0] == "X": prof_excl[iso_][fam_] = int(n_)
    rl_current = set(SUPPORTED_RL); rl81 = rl_current - {"arz", "dag", "bbj", "ibb"}
    cov_rows = []
    for iso in [i for i in ALL_ISOS if i != PIVOT_ISO]:
        s_ = sel.get(iso, {}); n_ = s_.get("n", 0) if s_ else 0
        fams = {fam for (i2, cls, reg, fam, nc, sh) in tally if i2 == iso}
        regs = {(cls, reg) for (i2, cls, reg, fam, nc, sh) in tally if i2 == iso}
        cav = []
        rel_only = bool(n_) and s_["gen"] + s_["unv"] + s_["sci"] == 0
        if rel_only: cav.append("RELIGIOUS-ONLY (%d rows)" % s_["rel"])
        elif n_ and s_["rel"]: cav.append("religious share %.0f%%" % (100 * s_["rel"] / n_))
        if n_ and s_["unv"]: cav.append("%d unverified-provenance pairs" % s_["unv"])
        if n_ and s_["nc"]: cav.append("%d non-commercial pairs" % s_["nc"])
        if ("gen", "flores_dev_train") in regs: cav.append("FLORES+ dev train half present")
        if n_ and s_["science"]: cav.append("%d AfriScience (science register)" % s_["science"])
        excl_f = {fam: n for (i2, fam, reg), n in excl.items() if i2 == iso}
        for fam, n in prof_excl.get(iso, {}).items(): excl_f.setdefault(fam, n)
        status = "none" if n_ == 0 else ("covered" if n_ >= 0.95 * a.min_pairs else "thin (< %d)" % a.min_pairs)
        reason = ""
        if n_ == 0:
            reason = ("only excluded sources exist: " + ", ".join(f"{f} {n}" for f, n in sorted(excl_f.items(), key=lambda x: -x[1])[:4])) if excl_f else "no pair in any clean source (or all removed by length / protected-set filters)"
        elif n_ < a.min_pairs:
            reason = "only %d eligible human pairs after the filters" % n_
        cov_rows.append({"iso": iso, "name": LANGUAGES[iso].name if iso in LANGUAGES else "", "rl_current": int(iso in rl_current), "rl81": int(iso in rl81), "rl_consensus_only": int(bool(getattr(LANGUAGES.get(iso), "rl_consensus", False))),
                         "benchmarks_with_reference": "+".join(bench_refs(iso)), "gen_gross": gross[iso]["gen"] if iso in gross else 0, "rel_gross": gross[iso]["rel"] if iso in gross else 0, "unv_gross": gross[iso]["unv"] if iso in gross else 0, "sci_gross": gross[iso]["sci"] if iso in gross else 0,
                         "religious_only": int(rel_only), "religious_share": round(s_["rel"] / n_, 3) if n_ else "", "chosen": n_, "chosen_train": n_ - (s_.get("val", 0) if s_ else 0), "chosen_val": s_.get("val", 0) if s_ else 0, "chosen_gen": s_.get("gen", 0) if s_ else 0, "chosen_rel": s_.get("rel", 0) if s_ else 0, "chosen_unv": s_.get("unv", 0) if s_ else 0, "chosen_sci": s_.get("sci", 0) if s_ else 0, "chosen_sci_val": s_.get("sci_val", 0) if s_ else 0,
                         "status": status, "register_caveats": "; ".join(cav), "sources": "; ".join(sorted(fams)), "excluded_sources_rows": "; ".join(f"{f}:{n}" for f, n in sorted(excl_f.items(), key=lambda x: -x[1])[:5]), "reason_if_not_covered": reason})
    with open(os.path.join(a.out, "coverage_by_language.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cov_rows[0])); w.writeheader(); w.writerows(sorted(cov_rows, key=lambda r: (-r["rl_current"], r["iso"])))
    cov_summary = {"rl_current": len(rl_current), "rl81": len(rl81), "rl81_covered": sum(1 for r in cov_rows if r["rl81"] and r["status"] == "covered"), "rl81_thin": sum(1 for r in cov_rows if r["rl81"] and r["status"].startswith("thin")),
                   "rl81_none": [r["iso"] for r in cov_rows if r["rl81"] and r["status"] == "none"], "rl81_thin_list": [r["iso"] for r in cov_rows if r["rl81"] and r["status"].startswith("thin")],
                   "rl81_without_any_benchmark_reference": [r["iso"] for r in cov_rows if r["rl81"] and not r["benchmarks_with_reference"]]}
    summary = {"coverage": cov_summary, "pairs": len(out), "train": sum(c["split"] == "train" for c in out), "val": sum(c["split"] == "val" for c in out), "languages": len(sel), "classes": {k: sum(c["cls"] == k for c in out) for k in ("gen", "rel", "unv", "sci")},
               "non_commercial_pairs": sum(bool(c["nc"]) for c in out), "unseen_by_sft_pairs": sum(not c["sft_input"] for c in out), "stageB_new_pairs": sum(c["stage"] == "stageB-new" for c in out), "A0_era_pairs": sum(c["stage"] == "A0-era" for c in out), "removed_second_pass": dict(removed), "dropped_counts_first_pass": dict(dropped),
               "bench_dir_strings": len(bench) if bench is not None else None, "protected_index": a.protected_index, "seconds": round(time.time() - t0),
               "religious_share_overall": round(sum(c["cls"] == "rel" for c in out) / max(1, len(out)), 4), "religious_only_languages": sorted(r["iso"] for r in cov_rows if r["religious_only"]), "rel_only_cap_used": a.rel_only_cap_used,
               "params": {k: v for k, v in vars(a).items() if k not in ("input",)}}
    json.dump(summary, open(os.path.join(a.out, "selection_summary.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "params"}, indent=1, default=str))


if __name__ == "__main__":
    main()
