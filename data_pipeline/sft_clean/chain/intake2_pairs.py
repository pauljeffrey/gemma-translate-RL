#!/usr/bin/env python3
"""extra2 parallel-pair intake for the SabiYarn clean SFT rerun (2026-10-31). Runs ON THE INSTANCE (needs the protected index and the existing-key file).

    python intake2_pairs.py plan                              # builds work/_intake2/tasks.json (Hub listings) and the existing-key file if missing
    python intake2_pairs.py test --sources mt560_new mehgoss  # 1,000 rows per source through the full cleaning path (no index, nothing written to hub/)
    python intake2_pairs.py run [--sources A B] [--workers 24]   # stage every task, then finalize (cross-source dedupe + shards + stats)
    python intake2_pairs.py finalize                          # only the second step
    python intake2_pairs.py tokens                            # token estimates (Gemma tokenizer, 2,000 rows per source x language)

Pattern = extra_sources.py: standard pair schema (english, target, iso, source_dataset, licence, licence_raw, kind, source_family, flipped, key, aux), cleaning `ok()`, dedupe key
`common.key_of(english, iso, target)`, STRICT decontamination (`extra_sources.Filter`, both sides, any language, rules R1/R2/R3), per-language stats.

Order of the filters per row: clean (whitespace/NFC, ok(), source specific rules) -> dedupe against the key set of the chain inputs (hub/data, data_devhalf, specialist, extra:
work/existing_keys.npy, built once) and inside the task -> strict protected filter -> staging shard. The finalize step then removes pairs that an EARLIER source (priority order of SOURCES)
already kept, caps the sources that have a cap and writes hub/extra2/<source>/pairs-NNNN.parquet (200k rows per shard). Raw files are read from the Hub one at a time and deleted.

Licence rules: commercial-use or no-licence data only; `licence` = class (cc-by, cc0, mit, apache, share_alike, mpl, unspecified, other), `licence_raw` = the raw string (card or per-row field).
Rows whose per-row licence string says non-commercial are dropped (counted as dropped_licence). Splits named test* are never read. Nothing from FLORES / NTREX / SMOL / GATITOS / MAFAND /
TICO / AfriDocMT, nothing NLLB / CCMatrix / CCAligned / TranslatePsy.
"""
import argparse, collections, csv, glob, json, math, multiprocessing as mp, os, random, re, shutil, sys, time, unicodedata

import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from common import key_of                                               # noqa: E402
from extra_sources import ok, norm_ws, PLACEHOLDER, SCHEMA_COLS, URL   # noqa: E402

ROOT = os.environ.get("CLEAN_SFT_ROOT", "/workspace/work")
WORK = f"{ROOT}/work/_intake2"
STAGE = f"{WORK}/stage"
RAW = f"{WORK}/raw"
OUT = f"{ROOT}/hub/extra2"
INDEX = f"{ROOT}/work/_hub_protected/protected_index/protected_index.npz"
MANIFEST = f"{ROOT}/work/_hub_protected/protected_index/protected_manifest.json"
EXISTING = f"{ROOT}/work/existing_keys.npy"
TASKS = f"{WORK}/tasks.json"
PROGRESS = f"{ROOT}/logs/intake2_pairs.progress"
SHARD = 200_000
STAGE_ROWS = 100_000
CAP_DOCHPLT = 3_000_000
GEMMA_TOK = "google/gemma-3-270m-it"

# ---------------------------------------------------------------------------------------------------------------------------------------------- licences
def lic_class(raw):
    """class of a raw licence string; 'non_commercial' rows are dropped by the caller"""
    t = (raw or "").strip().lower()
    if not t or t in ("none", "null", "unknown", "nan"):
        return "unspecified"
    if re.search(r"(^|[^a-z])nc([^a-z]|$)|non-?commercial|noncommercial", t) or "all rights reserved" in t or "proprietary" in t:
        return "non_commercial"
    if re.search(r"gpl|(^|[^a-z])sa([^a-z]|$)|share-?alike|eupl|cddl|osl|epl", t):
        return "share_alike"
    if "mpl" in t or "mozilla" in t:
        return "mpl"
    if "apache" in t:
        return "apache"
    if "cc0" in t or "public domain" in t or "cc-zero" in t:
        return "cc0"
    if re.search(r"(^|[^a-z])(mit|bsd|isc|zlib|unlicense|x11|0bsd|wtfpl|python)([^a-z]|$)", t):
        return "mit"
    if re.search(r"cc-?by|creative commons attribution|odc-by", t):
        return "cc-by"
    return "other"


# ---------------------------------------------------------------------------------------------------------------------------------------------- Hub access
def _token():
    return os.environ.get("HF_TOKEN")


def hf_get(repo, filename, tag, revision=None):
    """download one repo file into a private raw dir (deleted by the caller via rm_raw); returns the local path"""
    from huggingface_hub import hf_hub_download
    d = f"{RAW}/{os.getpid()}_{tag}"
    for k in range(4):
        try:
            return hf_hub_download(repo, filename, repo_type="dataset", token=_token(), local_dir=d, revision=revision)
        except Exception as e:
            if k == 3 or "404" in str(e) or "401" in str(e) or "403" in str(e):
                raise
            time.sleep(8 * (k + 1))


def rm_raw(tag):
    shutil.rmtree(f"{RAW}/{os.getpid()}_{tag}", ignore_errors=True)


def hf_list(repo):
    from huggingface_hub import HfApi
    return [s.rfilename for s in HfApi(token=_token()).dataset_info(repo).siblings]


def iter_parquet(path, cols, batch=50_000):
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=batch, columns=cols):
        yield rb.to_pydict()


def iter_csv(path, delimiter=",", encoding="utf-8-sig", quoting_none=False):
    csv.field_size_limit(10 ** 8)
    with open(path, encoding=encoding, newline="") as f:
        for r in csv.DictReader(f, delimiter=delimiter, quoting=csv.QUOTE_NONE if quoting_none else csv.QUOTE_MINIMAL):
            yield r


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------------------------------------------------------------------------- adapters
# each adapter yields dicts {en, tg, iso} plus optional overrides {src (source_dataset), lic, lic_raw, kind, family, aux}
def gen_mt560(t, spec):
    p = hf_get(t["repo"], "data/train-00000-of-00001.parquet", t["id"])
    try:
        names = pq.ParquetFile(p).schema_arrow.names
        tg = [c for c in names if c != "eng"][0]
        for d in iter_parquet(p, ["eng", tg]):
            for e, g in zip(d["eng"], d[tg]):
                if e is not None and g is not None:
                    yield dict(en=e, tg=g, iso=t["iso"], src=t["repo"], lic_raw=f"cc-by-4.0 (HF card of {t['repo']}; OPUS MT560, mostly JW-derived religious text)")
    finally:
        rm_raw(t["id"])


MEHGOSS_ISO = {"af": "afr", "nr": "nbl", "nso": "nso", "ss": "ssw", "st": "sot", "tn": "tsn", "ts": "tso", "ve": "ven", "xh": "xho", "zu": "zul"}


def gen_mehgoss(t, spec):
    p = hf_get(t["repo"], t["file"], t["id"])
    try:
        for d in iter_parquet(p, ["lang", "text", "en_text", "source"]):
            for lg, tg, en, so in zip(d["lang"], d["text"], d["en_text"], d["source"]):
                iso = MEHGOSS_ISO.get(lg)
                if iso and tg is not None and en is not None:
                    yield dict(en=en, tg=tg, iso=iso, family=f"mehgoss-{so}", lic_raw=f"cc-by-4.0 (HF card of Mehgoss/sa-languages-translation; component `{so}`, upstream terms of that component apply)")
    finally:
        rm_raw(t["id"])


def gen_pairs_file(t, spec):
    """generic reader: t = {repo, file, fmt (parquet|csv|tsv|jsonl), en_col, tg_col, iso, [lic_col], [keep], [drop_values], [min_score], [noquote]}"""
    p = hf_get(t["repo"], t["file"], t["id"])
    try:
        fmt, ec, gc, iso = t["fmt"], t["en_col"], t["tg_col"], t["iso"]
        lc = t.get("lic_col")
        keep = t.get("keep")                                          # {column: [allowed values]}
        drop_src = t.get("drop_values")                               # {column: [substrings that exclude the row]}
        if fmt == "parquet":
            cols = [ec, gc] + ([lc] if lc else []) + list((keep or {})) + list((drop_src or {})) + (["score"] if t.get("min_score") else [])
            it = (dict(zip(d.keys(), vals)) for d in iter_parquet(p, list(dict.fromkeys(cols))) for vals in zip(*d.values()))
        elif fmt in ("csv", "tsv"):
            it = iter_csv(p, "\t" if fmt == "tsv" else ",", quoting_none=t.get("noquote", False))
        else:
            it = iter_jsonl(p)
        for r in it:
            en, tg = r.get(ec), r.get(gc)
            if en is None or tg is None:
                continue
            if keep and any(str(r.get(c)) not in vs for c, vs in keep.items()):
                continue
            if drop_src and any(any(s_ in str(r.get(c) or "").lower() for s_ in ss) for c, ss in drop_src.items()):
                continue
            if t.get("min_score"):
                try:
                    if float(r.get("score")) < t["min_score"]:
                        continue
                except (TypeError, ValueError):
                    continue
            o = dict(en=en, tg=tg, iso=iso)
            if lc:
                raw = str(r.get(lc) or "")
                o["lic_row"] = raw
                o["lic_raw"] = f"per-row licence field: {raw or 'not stated'} ({t['repo']})"
            yield o
    finally:
        rm_raw(t["id"])


def gen_csv_swapped(t, spec):                                         # gazette tsv has `kin\teng` header with quoted multi-line fields
    yield from gen_pairs_file(t, spec)


def gen_paracrawl(t, spec):
    p = hf_get(t["repo"], t["file"], t["id"])
    try:
        for d in iter_parquet(p, ["translation"]):
            for tr in d["translation"]:
                if tr and tr.get("en") is not None and tr.get(t["lang"]) is not None:
                    yield dict(en=tr["en"], tg=tr[t["lang"]], iso=t["iso"])
    finally:
        rm_raw(t["id"])


def gen_inioluwa(t, spec):
    yield from gen_pairs_file(t, spec)


WEBLATE_MAP = {"aa": "aar", "ach": "ach", "af": "afr", "ak": "twi", "tw": "twi", "am": "amh", "bem": "bem", "ber": "ber", "bm": "bam", "dua": "dua", "ee": "ewe", "ff": "ful",
               "guw": "guw", "ha": "hau", "hz": "her", "ig": "ibo", "kab": "kab", "kg": "kon", "ki": "kik", "lg": "lug", "ln": "lin", "mfe": "mfe", "mg": "mlg", "nd": "nde", "nr": "nbl",
               "ny": "nya", "om": "orm", "rn": "run", "rw": "kin", "sg": "sag", "sn": "sna", "so": "som", "ss": "ssw", "st": "sot", "sw": "swh", "ti": "tir", "tn": "tsn", "ts": "tso",
               "tum": "tum", "tzm": "tzm", "ve": "ven", "wo": "wol", "xh": "xho", "yo": "yor", "zgh": "zgh", "zu": "zul", "saq": "saq", "sus": "sus", "son": "son", "ssy": "ssy",
               "abr": "abr", "rcf": "rcf", "luo": "luo", "kj": "kua", "lua": "lua", "ndc": "ndc", "wal": "wal", "nus": "nus", "shi": "shi", "kbp": "kbp", "ewe": "ewe"}


def weblate_iso(cfg):
    """config name `en-<code>` / `en_GB-<code>` -> chain iso or None"""
    code = cfg.split("-", 1)[1] if "-" in cfg else ""
    if code.startswith("b+"):
        base = code.split("+")[1].lower()
    else:
        base = re.split(r"[-_@]", code)[0].lower()
    return WEBLATE_MAP.get(base)


def gen_weblate(t, spec):
    p = hf_get(t["repo"], t["file"], t["id"])
    try:
        for d in iter_parquet(p, ["source", "target", "license", "effective_license", "state", "is_approved"]):
            for en, tg, lc, elc, st, ap in zip(d["source"], d["target"], d["license"], d["effective_license"], d["state"], d["is_approved"]):
                if en is None or tg is None or ((st or 0) < 20 and not ap):
                    continue
                raw = (elc or lc or "")
                yield dict(en=en, tg=tg, iso=t["iso"], lic_raw=f"Weblate per-string licence: {raw or 'not stated'} (Tamazight-NLP/Weblate-Translations, card license: other)", lic_row=raw, aux=float("nan"))
    finally:
        rm_raw(t["id"])


def gen_dochplt(t, spec):
    """document pairs -> sentence pairs from the dataset's own 1:1 alignment entries (one id on each side) with bicleaner >= 0.5 and aligner >= 0.4, at most 40 pairs per document and
    `quota` pairs per file (spreads the 3 M cap over all files of the config: the files are sorted by domain). A document whose sentence lists differ in length is NOT paired by position."""
    p = hf_get(t["repo"], t["file"], t["id"])
    try:
        pf = pq.ParquetFile(p)
        quota, n = t["quota"], 0
        a_lang, b_lang = t["config"].split("-")
        en_is_src = a_lang == "en"
        for rg in range(pf.num_row_groups):
            if n >= quota:
                break
            tab = pf.read_row_group(rg, columns=["src_doc", "tgt_doc", "alignment"]).to_pylist()
            for doc in tab:
                S, T = doc["src_doc"], doc["tgt_doc"]
                si, ti = dict(zip(S["ids"], S["sentences"])), dict(zip(T["ids"], T["sentences"]))
                per_doc = 0
                for a in doc["alignment"]:
                    if len(a["src"]) != 1 or len(a["tgt"]) != 1:
                        continue
                    if (a["bicleaner-score"] or 0) < 0.5 or (a["aligner-score"] or 0) < 0.4:
                        continue
                    x, y = si.get(a["src"][0]), ti.get(a["tgt"][0])
                    if x is None or y is None:
                        continue
                    en, tg = (x, y) if en_is_src else (y, x)
                    yield dict(en=en, tg=tg, iso=t["iso"], aux=float(a["bicleaner-score"]))
                    per_doc += 1; n += 1
                    if per_doc >= 40 or n >= quota:
                        break
                if n >= quota:
                    break
    finally:
        rm_raw(t["id"])


def gen_autshumato(t, spec):
    """nwu-ctext/autshumato is a loading-script repo; its auto-converted parquet files (branch refs/convert/parquet of the same repo) hold the `translation` struct {en, <lang>}"""
    p = hf_get(t["repo"], t["file"], t["id"], revision="refs/convert/parquet")
    try:
        for d in iter_parquet(p, ["translation"]):
            for tr in d["translation"]:
                if tr and tr.get("en") is not None and tr.get(t["lang"]) is not None:
                    yield dict(en=tr["en"], tg=tr[t["lang"]], iso=t["iso"], family=f"autshumato-{t['config']}")
    finally:
        rm_raw(t["id"])


ADAPTERS = {"autshumato": gen_autshumato, "mt560": gen_mt560, "mehgoss": gen_mehgoss, "file": gen_pairs_file, "paracrawl": gen_paracrawl, "weblate": gen_weblate, "dochplt": gen_dochplt}


# ---------------------------------------------------------------------------------------------------------------------------------------------- source table (priority order)
# licence/licence_raw/kind/family are defaults of the source; a task or a row may override them (lic_raw / family / src)
SOURCES = collections.OrderedDict([
    ("mt560_new", dict(adapter="mt560", lic="cc-by", lic_raw="cc-by-4.0 (HF card)", kind="human", family="opus-mt560", repo="michsethowusu/english-<language>_sentence-pairs_mt560 (31 repos)")),
    ("mehgoss", dict(adapter="mehgoss", lic="cc-by", lic_raw="cc-by-4.0 (HF card)", kind="human", family="mehgoss", repo="Mehgoss/sa-languages-translation")),
    ("igbo_bundles", dict(adapter="file", lic="unspecified", lic_raw="no licence on the Hub", kind="human", family="igbo-bundles", repo="Tommy0201/* , iamwille/igbo_language_translation_dataset, ccibeekeoc42/english_to_igbo")),
    ("weblate", dict(adapter="weblate", lic="unspecified", lic_raw="per-string", kind="ui_strings", family="weblate", min_words=3, drop_placeholders=True, repo="Tamazight-NLP/Weblate-Translations")),
    ("kabyle_tm", dict(adapter="file", lic="cc-by", lic_raw="cc-by-4.0 (HF card)", kind="ui_strings", family="kabyle-tm", min_words=3, drop_placeholders=True, repo="taqbaylit/kabyle-english-TM")),
    ("nupe", dict(adapter="file", lic="unspecified", lic_raw="HF card license tag `cc` (variant not stated)", kind="human", family="nupe", min_words=3, repo="Inioluwa/nigerianLanguageTranslator")),
    ("twi_kweku", dict(adapter="file", lic="unspecified", lic_raw="no licence on the Hub", kind="human", family="twi-kweku", repo="kweku2088/twi_to_english")),
    ("dochplt", dict(adapter="dochplt", lic="cc0", lic_raw="cc0-1.0 (HF card of HPLT/DocHPLT)", kind="mined_doc_aligned", family="dochplt", cap=CAP_DOCHPLT, min_words=4, repo="HPLT/DocHPLT")),
    ("xhosa_silvano", dict(adapter="file", lic="unspecified", lic_raw="per-row licence column", kind="human", family="xhosa-silvano", repo="silvanosolutions/xhosa-nlp-dataset")),
    ("swh_rogendo", dict(adapter="file", lic="unspecified", lic_raw="no licence on the Hub", kind="human", family="swh-rogendo", min_words=3, repo="Rogendo/English-Swahili-Sentence-Pairs")),
    ("kin_gazette", dict(adapter="file", lic="unspecified", lic_raw="HF card license tag `cc` (variant not stated)", kind="human", family="kin-gazette", repo="DigitalUmuganda/NMT_Rwandan-Gazette_parallel_data_en_kin")),
    ("tatoeba_kab_shi", dict(adapter="file", lic="cc-by", lic_raw="cc-by-2.0 (HF card; Tatoeba sentences, CC BY 2.0 FR)", kind="human", family="tatoeba-abdelhaque", repo="abdelhaqueidali/{Kabyle-Latin-to-Tifinagh-Parallel-Corpus, Tashelhit-Tifinagh-Latin-Parallel-Tatoeba-Dataset}")),
    ("govza", dict(adapter="file", lic="cc-by", lic_raw="cc-by-4.0 (HF card of dsfsi/govza-sa-cabinet-statements-sentence-aligned)", kind="aligned_sentences", family="govza", min_words=3, repo="dsfsi/govza-sa-cabinet-statements-sentence-aligned")),
    ("lug_pky", dict(adapter="file", lic="apache", lic_raw="apache-2.0 (HF card)", kind="human", family="lug-pkyoyetera", repo="pkyoyetera/luganda_english_dataset")),
    ("paracrawl", dict(adapter="paracrawl", lic="cc0", lic_raw="cc0-1.0 (HF card of Helsinki-NLP/opus_paracrawl)", kind="mined_web", family="paracrawl", repo="Helsinki-NLP/opus_paracrawl")),
    ("autshumato", dict(adapter="autshumato", lic="cc-by", lic_raw="cc-by-2.5 (HF card of nwu-ctext/autshumato; Autshumato, South African government/university text)", kind="human", family="autshumato", repo="nwu-ctext/autshumato (refs/convert/parquet)")),
    ("okwu", dict(adapter="file", lic="other", lic_raw="cdla-permissive-2.0 (HF card of Okwu/african-language-parallel-corpus)", kind="human", family="okwu", repo="Okwu/african-language-parallel-corpus")),
])
MT560_ISO = {"alur": "alz", "bulu": "bum", "chokwe": "cjk", "chopi": "cce", "chuwabu": "chw", "coptic": "cop", "dangme": "ada", "egyptian-arabic": "arz", "gun": "guw", "konzo": "koo", "kwangali": "kwn", "lamba": "lam",
             "luba-katanga": "lub", "lukpa": "dop", "luvale": "lue", "mambwe-lungu": "mgr", "morisyen": "mfe", "ndonga": "ndo", "nyaneka": "nyk", "nyemba": "nba", "nyungwe": "nyu", "ruund": "rnd",
             "seychelles-french-creole": "crs", "songe": "sop", "southwestern-dinka": "dik", "tamashek": "taq", "tetela": "tll", "tonga": "toi", "tooro": "ttj", "zande": "zne", "zarma": "dje"}


def build_tasks():
    from huggingface_hub import HfApi
    api = HfApi(token=_token())
    T = collections.OrderedDict((k, []) for k in SOURCES)
    # 1. mt560: the repos with no tag yet (31 = the 99 repos minus the 68 the clean build used)
    for n, iso in MT560_ISO.items():
        r = f"michsethowusu/english-{n}_sentence-pairs_mt560"
        T["mt560_new"].append(dict(id=f"mt560_{iso}", source="mt560_new", repo=r, iso=iso, est=300))
    # 2. Mehgoss: every parquet except test splits
    for f in hf_list("Mehgoss/sa-languages-translation"):
        if f.endswith(".parquet") and "/test-" not in f:
            T["mehgoss"].append(dict(id="mehgoss_" + f.replace("/", "_")[:-8], source="mehgoss", repo="Mehgoss/sa-languages-translation", file=f, est=500 if "train" in f else 80))
    # 4. Igbo bundles in priority order (siblings duplicate each other: the key dedupe counts them once)
    P = "data/{}-00000-of-00001.parquet"
    ig = [("Tommy0201/JW300_Igbo_To_Eng", [P.format("train"), P.format("eval")], "parquet", "english", "igbo", "unspecified"),
          ("Tommy0201/WebScraped_Igbo_To_Eng", [P.format("train"), P.format("eval")], "parquet", "english", "igbo", "unspecified"),
          ("Tommy0201/IGBONLP_Igbo_To_Eng", [P.format("train"), P.format("eval")], "parquet", "english", "igbo", "unspecified"),
          ("Tommy0201/Igbo_To_Eng_Amazon_Translator", ["train.csv", "validation.csv"], "csv", "eng_output", "igbo_input", "unspecified"),
          ("Tommy0201/igbo_to_english_split", [P.format("train")], "parquet", "english", "igbo", "unspecified"),
          ("Tommy0201/igbo-to-english", [P.format("train")], "parquet", "english", "igbo", "unspecified"),
          ("iamwille/igbo_language_translation_dataset", [P.format("train"), P.format("valid")], "parquet", "english", "igbo", "mit"),
          ]
    for repo, files, fmt, en, tg, lic in ig:
        for f in files:
            raw = "MIT (HF card)" if lic == "mit" else "no licence on the Hub (card license: none); copy of a JW300 / web-scraped / IGBONLP / Amazon-translator bundle"
            T["igbo_bundles"].append(dict(id=f"igbo_{repo.split('/')[1]}_{f.rsplit('/', 1)[-1].rsplit('.', 1)[0]}", source="igbo_bundles", repo=repo, file=f, fmt=fmt, en_col=en, tg_col=tg, iso="ibo", est=200,
                                          lic_override=lic, lic_raw_override=raw))
    T["igbo_bundles"].append(dict(id="igbo_ccibeekeoc42_train", source="igbo_bundles", repo="ccibeekeoc42/english_to_igbo", file="train.jsonl", fmt="jsonl", en_col="English", tg_col="Igbo", iso="ibo", est=400,
                                  lic_override="mit", lic_raw_override="MIT (HF card)"))
    # 5. Weblate: en-<code> and en_GB-<code> configs of African languages
    names = sorted({f.split("/")[0] for f in hf_list("Tamazight-NLP/Weblate-Translations") if f.endswith(".parquet")})
    for c in names:
        if c.startswith("en-") or c.startswith("en_GB-"):
            iso = weblate_iso(c)
            if iso:
                T["weblate"].append(dict(id=f"weblate_{c}", source="weblate", repo="Tamazight-NLP/Weblate-Translations", file=f"{c}/train-00000-of-00001.parquet", iso=iso, est=20))
    # 6. Kabyle TM, Nupe (train only), Twi
    T["kabyle_tm"].append(dict(id="kabyle_tm", source="kabyle_tm", repo="taqbaylit/kabyle-english-TM", file="kab_en_translation_memory.parquet", fmt="parquet", en_col="source", tg_col="target", iso="kab", est=100,
                               keep={"source_lang": ["en"], "target_lang": ["kab"]}))
    T["nupe"].append(dict(id="nupe", source="nupe", repo="Inioluwa/nigerianLanguageTranslator", file="data/train-00000-of-00001.parquet", fmt="parquet", en_col="eng", tg_col="nupe", iso="nup", est=100))
    T["twi_kweku"].append(dict(id="twi_kweku", source="twi_kweku", repo="kweku2088/twi_to_english", file="twi_english.csv", fmt="csv", en_col="ENGLISH", tg_col="TWI", iso="twi", est=500))
    # 7. DocHPLT: afr, swh, xho (3 configs), all files, per-file quota
    files = [s for s in HfApi(token=_token()).dataset_info("HPLT/DocHPLT").siblings if s.rfilename.endswith(".parquet")]
    for cfg, iso in (("af-en", "afr"), ("en-sw", "swh"), ("en-xh", "xho")):
        fs = sorted(s.rfilename for s in files if s.rfilename.startswith(cfg + "/"))
        for f in fs:
            T["dochplt"].append(dict(id=f"dochplt_{cfg}_{f.split('train-')[1][:5]}", source="dochplt", repo="HPLT/DocHPLT", file=f, config=cfg, iso=iso, src_name=f"HPLT/DocHPLT:{cfg}",
                                     quota=int(math.ceil(2.0 * CAP_DOCHPLT / len(fs))), est=900))
    # 8. extras
    T["xhosa_silvano"].append(dict(id="xhosa_silvano", source="xhosa_silvano", repo="silvanosolutions/xhosa-nlp-dataset", file="parallel.parquet", fmt="parquet", en_col="english", tg_col="xhosa", iso="xho", est=200,
                                   lic_col="license", keep={"type": ["parallel"]}))
    T["swh_rogendo"].append(dict(id="swh_rogendo", source="swh_rogendo", repo="Rogendo/English-Swahili-Sentence-Pairs", file="ensw.csv", fmt="csv", en_col="English sentence", tg_col="Swahili Translation", iso="swh", est=200))
    T["kin_gazette"].append(dict(id="kin_gazette_train", source="kin_gazette", repo="DigitalUmuganda/NMT_Rwandan-Gazette_parallel_data_en_kin", file="gazette_train.tsv", fmt="tsv", en_col="eng", tg_col="kin", iso="kin", est=170))
    T["kin_gazette"].append(dict(id="kin_gazette_val", source="kin_gazette", repo="DigitalUmuganda/NMT_Rwandan-Gazette_parallel_data_en_kin", file="gazette_val.tsv", fmt="tsv", en_col="eng", tg_col="kin", iso="kin", est=5))
    T["tatoeba_kab_shi"].append(dict(id="tatoeba_kab", source="tatoeba_kab_shi", repo="abdelhaqueidali/Kabyle-Latin-to-Tifinagh-Parallel-Corpus", file="en_kab_tifinagh.tsv", fmt="tsv", en_col="en", tg_col="kab_latn", iso="kab", est=135, noquote=True))
    T["tatoeba_kab_shi"].append(dict(id="tatoeba_shi", source="tatoeba_kab_shi", repo="abdelhaqueidali/Tashelhit-Tifinagh-Latin-Parallel-Tatoeba-Dataset", file="en_shi_tifinagh.tsv", fmt="tsv", en_col="en", tg_col="shi_latn", iso="shi", est=22, noquote=True))
    for sp in ("train", "eval"):
        T["govza"].append(dict(id=f"govza_afr-eng_{sp}", source="govza", repo="dsfsi/govza-sa-cabinet-statements-sentence-aligned", file=f"afr-eng/{sp}/0000.parquet", fmt="parquet", en_col="eng", tg_col="afr", iso="afr",
                               est=20, min_score=0.9))
    T["lug_pky"].append(dict(id="lug_pky", source="lug_pky", repo="pkyoyetera/luganda_english_dataset", file="data/train-00000-of-00001-d6ac85019c81ba60.parquet", fmt="parquet", en_col="English", tg_col="Luganda", iso="lug", est=80))
    T["paracrawl"].append(dict(id="paracrawl_sw", source="paracrawl", repo="Helsinki-NLP/opus_paracrawl", file="en-sw/train-00000-of-00001.parquet", lang="sw", iso="swh", est=100))
    T["paracrawl"].append(dict(id="paracrawl_so", source="paracrawl", repo="Helsinki-NLP/opus_paracrawl", file="en-so/train-00000-of-00001.parquet", lang="so", iso="som", est=20))
    for pair, iso in (("pcm-eng", "pcm"), ("swa-eng", "swh"), ("yor-eng", "yor")):
        T["okwu"].append(dict(id=f"okwu_{pair}", source="okwu", repo="Okwu/african-language-parallel-corpus", file=f"data/{pair}/{pair}.csv", fmt="csv", en_col="english_translation", tg_col="source_text", iso=iso, est=30,
                              keep={"verified": ["true", "True"]}, drop_values={"source": ["smol", "gatitos", "flores", "ntrex", "mafand", "tico"]}))
    for cfg, lang, iso in (("autshumato-en-tn", "tn", "tsn"), ("autshumato-en-ts", "ts", "tso"), ("autshumato-en-ts-manual", "ts", "tso"), ("autshumato-en-zu", "zu", "zul")):
        T["autshumato"].append(dict(id=cfg, source="autshumato", repo="nwu-ctext/autshumato", file=f"{cfg}/train/0000.parquet", config=cfg, lang=lang, iso=iso, est=200))
    json.dump(T, open(TASKS, "w"), indent=0)
    return T


# ---------------------------------------------------------------------------------------------------------------------------------------------- task runner
FILTER = None            # strict protected-set filter (loaded in the parent before the fork; shared copy-on-write)
EXISTING_KEYS = None     # sorted uint64 (mmap)
CAPS = {}                # shared counters for capped sources


def write_stage(rows, d, tid, n):
    os.makedirs(d, exist_ok=True)
    t = {c: [r[c] for r in rows] for c in SCHEMA_COLS}
    pq.write_table(pa.table({**t, "key": pa.array(t["key"], pa.uint64()), "aux": pa.array(t["aux"], pa.float32())}), f"{d}/{tid}-{n:04d}.parquet", compression="zstd")


def run_task(task, limit=None, out_root=None, check=True):
    """stage one task. Returns a stats dict. limit = stop after that many RAW rows (adapter test)."""
    spec = SOURCES[task["source"]]
    sname = task["source"]
    adapter = ADAPTERS[spec["adapter"]]
    st = collections.defaultdict(collections.Counter)                  # (iso, source_dataset) -> Counter
    removed = collections.Counter()                                    # (iso, source_dataset, rule_side) -> n
    seen, buf, staged, n_stage, samples = set(), [], 0, 0, []
    sdir = f"{out_root or STAGE}/{sname}"
    t0 = time.time()
    cap_iso = task.get("iso") if spec.get("cap") else None

    def flush():
        nonlocal buf, staged
        if not buf:
            return
        if EXISTING_KEYS is not None:
            ks = np.array([r["key"] for r in buf], np.uint64)
            pos = np.searchsorted(EXISTING_KEYS, ks); pos[pos >= len(EXISTING_KEYS)] = len(EXISTING_KEYS) - 1
            hit = EXISTING_KEYS[pos] == ks
            for r, h in zip(buf, hit):
                if h:
                    st[(r["iso"], r["source_dataset"])]["dropped_dup_existing"] += 1
            buf = [r for r, h in zip(buf, hit) if not h]
        res = FILTER.check(buf) if (FILTER is not None and check and buf) else [None] * len(buf)
        out = []
        for r, h in zip(buf, res):
            k = (r["iso"], r["source_dataset"])
            if h:
                st[k]["removed_protected"] += 1; removed[(r["iso"], r["source_dataset"], h)] += 1
            else:
                st[k]["kept_stage"] += 1; out.append(r)
        buf = []
        for r in out:
            staged_rows.append(r)
        if cap_iso and out:
            with CAPS[cap_iso].get_lock():
                CAPS[cap_iso].value += len(out)

    staged_rows = []
    n_raw = 0
    try:
        for g in adapter(task, spec):
            if cap_iso and CAPS[cap_iso].value >= spec["cap"] * 1.1:
                break
            n_raw += 1
            if limit and n_raw > limit:
                break
            iso = g["iso"]
            sd = g.get("src") or task.get("src_name") or task.get("repo") or sname
            k = (iso, sd)
            st[k]["raw_rows"] += 1
            en, tg = norm_ws(unicodedata.normalize("NFC", str(g["en"]))), norm_ws(unicodedata.normalize("NFC", str(g["tg"])))
            if not ok(en, tg):
                st[k]["dropped_clean"] += 1; continue
            mw = spec.get("min_words", 0)
            if (mw and len(en.split()) < mw) or (spec.get("drop_placeholders") and (PLACEHOLDER.search(en) or PLACEHOLDER.search(tg))):
                st[k]["dropped_short_or_placeholder"] += 1; continue
            lic_raw = g.get("lic_raw") or task.get("lic_raw_override") or spec["lic_raw"]
            lic = task.get("lic_override") or spec["lic"]
            if g.get("lic_row") is not None:
                lic = lic_class(g["lic_row"])
                if lic == "non_commercial":
                    st[k]["dropped_licence"] += 1; continue
            kk = key_of(en, iso, tg)
            if kk in seen:
                st[k]["dropped_dup_within"] += 1; continue
            seen.add(kk)
            st[k]["after_dedupe_within"] += 1
            r = dict(english=en, target=tg, iso=iso, source_dataset=sd, licence=lic, licence_raw=lic_raw, kind=g.get("kind") or spec["kind"], source_family=g.get("family") or spec["family"],
                     flipped=False, key=kk, aux=g.get("aux", float("nan")))
            if len(samples) < 3 and random.random() < 0.01 or (limit and len(samples) < 3):
                samples.append((en[:140], tg[:140]))
            buf.append(r)
            if len(buf) >= 5000:
                flush()
            if len(staged_rows) >= STAGE_ROWS:
                if out_root is None or True:
                    write_stage(staged_rows, sdir, task["id"], n_stage); n_stage += 1; staged += len(staged_rows)
                staged_rows = []
        flush()
        if staged_rows:
            write_stage(staged_rows, sdir, task["id"], n_stage); n_stage += 1; staged += len(staged_rows)
        err = None
    except Exception as e:
        err = repr(e)[:400]
    stats = dict(task=task["id"], source=sname, raw=n_raw, staged=staged, seconds=round(time.time() - t0), error=err, samples=samples,
                 st={f"{k[0]}\t{k[1]}": dict(v) for k, v in st.items()}, removed={f"{k[0]}\t{k[1]}\t{k[2]}": v for k, v in removed.items()})
    os.makedirs(sdir, exist_ok=True)
    json.dump(stats, open(f"{sdir}/{task['id']}.stats.json", "w"))
    return stats


def _worker(task):
    return run_task(task)


def log_progress(msg):
    os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
    with open(PROGRESS, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def load_globals(need_filter=True):
    global FILTER, EXISTING_KEYS
    if os.path.exists(EXISTING):
        EXISTING_KEYS = np.load(EXISTING, mmap_mode="r")
    if need_filter:
        from extra_sources import Filter
        FILTER = Filter(INDEX, MANIFEST)


def cmd_run(a):
    T = json.load(open(TASKS))
    names = a.sources or list(SOURCES)
    load_globals()
    for n in names:
        if SOURCES[n].get("cap"):
            for t in T[n]:
                CAPS.setdefault(t["iso"], mp.Value("q", 0))
    tasks = [t for n in names for t in T[n] if not os.path.exists(f"{STAGE}/{n}/{t['id']}.stats.json")]
    tasks.sort(key=lambda t: -t.get("est", 1))
    log_progress(f"run: {len(tasks)} tasks, {a.workers} workers, protected index {FILTER.version}, existing keys {len(EXISTING_KEYS):,}")
    done, rows, t0 = 0, 0, time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers) as pool:
        for s in pool.imap_unordered(_worker, tasks):
            done += 1; rows += s["staged"]
            log_progress(f"[{done}/{len(tasks)}] {s['task']}: raw {s['raw']:,} staged {s['staged']:,} in {s['seconds']}s{' ERROR ' + s['error'] if s['error'] else ''} | total staged {rows:,} | {time.time() - t0:.0f}s")


def cmd_test(a):
    T = json.load(open(TASKS)) if os.path.exists(TASKS) else build_tasks()
    load_globals(need_filter=a.with_filter)
    test_root = f"{WORK}/test_stage"
    shutil.rmtree(test_root, ignore_errors=True)
    for n in (a.sources or list(SOURCES)):
        ts = T[n]
        pick = ts[:1] if a.first_only else ts[:: max(1, len(ts) // 3)][:3]
        for t in pick:
            for k in range(1):
                for cap in CAPS:
                    pass
            if SOURCES[n].get("cap"):
                CAPS.setdefault(t["iso"], mp.Value("q", 0))
            s = run_task(t, limit=a.limit, out_root=test_root, check=a.with_filter)
            tot = collections.Counter()
            for v in s["st"].values():
                tot.update(v)
            print(f"[{n}/{t['id']}] raw {s['raw']} {dict(tot)} err={s['error']} {s['seconds']}s")
            for e, g in s["samples"][:2]:
                print("     EN:", e, "\n     TG:", g)


def cmd_plan(a):
    os.makedirs(WORK, exist_ok=True)
    T = build_tasks()
    print({k: len(v) for k, v in T.items()})


# ---------------------------------------------------------------------------------------------------------------------------------------------- finalize
def cmd_finalize(a):
    T = json.load(open(TASKS))
    os.makedirs(OUT + "/_stats", exist_ok=True)
    seen = np.empty(0, np.uint64)
    st = collections.defaultdict(collections.Counter); removed = collections.Counter(); errors = {}
    for sname in SOURCES:
        sdir = f"{STAGE}/{sname}"
        cap = SOURCES[sname].get("cap")
        capn = collections.Counter()
        # stats of the tasks
        for t in T[sname]:
            p = f"{sdir}/{t['id']}.stats.json"
            if not os.path.exists(p):
                errors[t["id"]] = "no stats (task not run)"; continue
            s = json.load(open(p))
            if s["error"]:
                errors[t["id"]] = s["error"]
            for k, v in s["st"].items():
                iso, sd = k.split("\t"); st[(iso, sd)].update(v)
            for k, v in s["removed"].items():
                iso, sd, h = k.split("\t"); removed[(iso, sd, h)] += v
        # write final shards for this source
        odir = f"{OUT}/{sname}"
        shutil.rmtree(odir, ignore_errors=True); os.makedirs(odir)
        writer, n_in_shard, shard_i, rows_out, newkeys = None, 0, 0, 0, []
        src_seen = np.empty(0, np.uint64)
        order = {t["id"]: i for i, t in enumerate(T[sname])}
        files = sorted(glob.glob(f"{sdir}/*.parquet"), key=lambda f: (order.get(os.path.basename(f).rsplit("-", 1)[0], 10 ** 6), f))
        for f in files:
            tb = pq.read_table(f)
            ks = tb["key"].to_numpy()
            dup_prev = np.zeros(len(ks), bool)
            for arr in (seen, src_seen):
                if len(arr):
                    pos = np.searchsorted(arr, ks); pos[pos >= len(arr)] = len(arr) - 1
                    dup_prev |= arr[pos] == ks
            _, first = np.unique(ks, return_index=True)
            first_mask = np.zeros(len(ks), bool); first_mask[first] = True
            keep = (~dup_prev) & first_mask
            isos, sds = tb["iso"].to_pylist(), tb["source_dataset"].to_pylist()
            for i in np.nonzero(~keep)[0]:
                st[(isos[i], sds[i])]["dropped_dup_other_source"] += 1
            if cap:
                idx = np.nonzero(keep)[0]
                for i in idx:
                    if capn[isos[i]] >= cap:
                        keep[i] = False; st[(isos[i], sds[i])]["dropped_cap"] += 1
                    else:
                        capn[isos[i]] += 1
            if not keep.any():
                continue
            tb = tb.filter(pa.array(keep))
            newkeys.append(tb["key"].to_numpy())
            src_seen = np.sort(np.concatenate([src_seen, tb["key"].to_numpy()]))
            kk = tb["iso"].to_pylist(), tb["source_dataset"].to_pylist()
            for a_, b_ in zip(*kk):
                st[(a_, b_)]["kept"] += 1
            start = 0
            while start < tb.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(f"{odir}/pairs-{shard_i:04d}.parquet", tb.schema, compression="zstd"); n_in_shard = 0
                take = min(SHARD - n_in_shard, tb.num_rows - start)
                writer.write_table(tb.slice(start, take)); n_in_shard += take; start += take; rows_out += take
                if n_in_shard >= SHARD:
                    writer.close(); writer = None; shard_i += 1
        if writer is not None:
            writer.close()
        if newkeys:
            seen = np.sort(np.concatenate([seen] + newkeys))
        log_progress(f"finalize {sname}: {rows_out:,} rows in {shard_i + (1 if writer is not None or n_in_shard else 0)} shards")
    write_stats(st, removed, errors)


def write_stats(st, removed, errors):
    d = OUT + "/_stats"
    cols = ["raw_rows", "dropped_clean", "dropped_short_or_placeholder", "dropped_licence", "dropped_dup_within", "dropped_dup_existing", "removed_protected", "dropped_dup_other_source", "dropped_cap", "kept"]
    with open(d + "/per_language_source.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iso", "source", "raw_rows", "dropped_by_clean_rules", "dropped_short_or_placeholder", "dropped_licence", "dropped_dup_within_source", "dropped_dup_in_chain_inputs",
                    "removed_by_protected_filter", "dropped_dup_earlier_source", "dropped_cap", "kept"])
        for (iso, sd), c in sorted(st.items()):
            w.writerow([iso, sd, c["raw_rows"], c["dropped_clean"], c["dropped_short_or_placeholder"], c["dropped_licence"], c["dropped_dup_within"], c["dropped_dup_existing"], c["removed_protected"],
                        c["dropped_dup_other_source"], c["dropped_cap"], c["kept"]])
    with open(d + "/removed_by_protected_set.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "source", "first_rule_and_side", "rows"])
        for (iso, sd, h), n in sorted(removed.items(), key=lambda kv: -kv[1]):
            w.writerow([iso, sd, h, n])
    per, rawl, prot = collections.Counter(), collections.Counter(), collections.Counter()
    for (iso, sd), c in st.items():
        per[iso] += c["kept"]; rawl[iso] += c["raw_rows"]; prot[iso] += c["removed_protected"]
    ver = None
    try:
        ver = json.load(open(MANIFEST)).get("version")
    except Exception:
        pass
    json.dump({"kept_by_language": dict(sorted(per.items())), "kept_total": sum(per.values()), "raw_rows_by_language": dict(sorted(rawl.items())), "removed_protected_by_language": dict(sorted(prot.items())),
               "protected_index_version": ver, "errors": errors, "sources": {k: dict(repo=v["repo"], licence=v["lic"]) for k, v in SOURCES.items()}}, open(d + "/summary.json", "w"), indent=1)
    print("kept by language:", dict(sorted(per.items())), "| total", sum(per.values()), "| errors:", errors or "none")


# ---------------------------------------------------------------------------------------------------------------------------------------------- tokens
def cmd_tokens(a):
    """sample up to 2,000 rows per (source, language) from the final shards, count Gemma tokens of english + target, scale by the rows kept"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEMMA_TOK, token=_token())
    rng = random.Random(7)
    rows = []
    for sname in SOURCES:
        files = sorted(glob.glob(f"{OUT}/{sname}/*.parquet"))
        if not files:
            continue
        pool = collections.defaultdict(list); count = collections.Counter()
        for f in files:
            t = pq.read_table(f, columns=["english", "target", "iso"]).to_pydict()
            for e, g, i in zip(t["english"], t["target"], t["iso"]):
                count[i] += 1
                L = pool[i]
                if len(L) < 2000:
                    L.append((e, g))
                else:
                    j = rng.randrange(count[i])
                    if j < 2000:
                        L[j] = (e, g)
        for i, L in sorted(pool.items()):
            te = sum(len(x) for x in tok([e for e, _ in L], add_special_tokens=False)["input_ids"])
            tg = sum(len(x) for x in tok([g for _, g in L], add_special_tokens=False)["input_ids"])
            n = len(L)
            rows.append([sname, i, count[i], n, round(te / n, 2), round(tg / n, 2), int(te / n * count[i]), int(tg / n * count[i]), int((te + tg) / n * count[i])])
    with open(OUT + "/_stats/tokens_by_source_language.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source", "iso", "rows", "sample", "en_tokens_per_row", "target_tokens_per_row", "est_en_tokens", "est_target_tokens", "est_total_tokens"]); w.writerows(rows)
    per = collections.Counter()
    for r in rows:
        per[r[1]] += r[8]
    json.dump(dict(sorted(per.items())), open(OUT + "/_stats/tokens_by_language.json", "w"), indent=1)
    print("est. tokens (english+target) by language:", dict(sorted(per.items())), "total", sum(per.values()))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("plan")
    t = sp.add_parser("test"); t.add_argument("--sources", nargs="*"); t.add_argument("--limit", type=int, default=1000); t.add_argument("--with-filter", action="store_true"); t.add_argument("--first-only", action="store_true")
    r = sp.add_parser("run"); r.add_argument("--sources", nargs="*"); r.add_argument("--workers", type=int, default=24)
    sp.add_parser("finalize"); sp.add_parser("tokens")
    a = ap.parse_args()
    os.makedirs(RAW, exist_ok=True); os.makedirs(STAGE, exist_ok=True)
    {"plan": cmd_plan, "test": cmd_test, "run": cmd_run, "finalize": cmd_finalize, "tokens": cmd_tokens}[a.cmd](a)


if __name__ == "__main__":
    main()
