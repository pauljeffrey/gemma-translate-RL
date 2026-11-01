#!/usr/bin/env python3
"""Clean English -> African SFT pairs (step 1 of the rerun, 2026-10-28): rebuild the pair set from the reachable sources of the old build and drop every pair
that carries protected evaluation text (decon.py, strict rule: any R1/R2/R3 hit on either side).

Same semantics as the old chain where the old scripts are known:
  gemma_sft_pairs.py   English on one side, flipped when English is the target, NOT_AFRICAN tags dropped, ok() length/script filter, whitespace collapse, exact dedupe key
  gemma_merge_pairs.py licence class per source (report/license_scan.json), non_commercial dropped, `flipped` kept
  gemma_gate_pairs.py  canonical iso tags (gemma_mt.languages), tags outside the table dropped   (QE score + lowest-15% cut + min-rows are NOT applied here: deferred, needs the GPU)
Plus the new part: source-level quarantine, NLLB/TranslatePsy exclusion, row-level decontamination, per-row accounting.

Unit = one input file. Per unit this writes   out/kept/<unit>.parquet   (english, target, iso, source_dataset, licence, flipped, key)  decontaminated rows
                                              out/meta/<unit>.parquet   (key, iso, src, drop, tlen)  one line for EVERY ok() pair (kept or dropped; no text)
and appends a line to out/ledger.jsonl (statistics incl. rows removed per rule/set/side). A restart skips finished units.

Run:  taskset -c <4 cores> nice -n 19 python build_pairs.py --out /workspace/work/out --index /workspace/work/protected/protected_index.npz
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import multiprocessing as mp
import os
import re
import sys
import threading
import time
from typing import Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

import decon as D

try:
    import gemma_languages as GL            # copy of gemma_mt/gemma_mt/languages.py
except ImportError:                          # laptop
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gemma_mt", "gemma_mt"))
    import languages as GL

NOT_AFRICAN = {"eng", "fra", "spa", "por", "deu", "ita", "rus", "ara", "arb", "ary", "arz", "aeb", "acm", "apc", "afb", "ayl", "und", "hin", "ben", "tir_dup"}
URL = re.compile(r"https?://|www\.", re.I)
EXCLUDE_SRC = re.compile(r"nllb|translatepsy|aya_interlanguage|aya-translate|aya_collection|aya-collection|aya_dataset|ghananlpcommunity/pristine|Turaco-NLLB", re.I)
# source-level quarantine (lead's rule, audit 5.3 item 4): benchmark-named sources; mafand/salt only when the name says dev/test
QUARANTINE_SRC = re.compile(r"flores|ntrex|smol|tico|sib200|xsim|afrobench|(mafand|salt)[^|]*(dev|test|valid)|(^|[/_.\-])(dev|devtest|test|validation|eval)([/_.\-]|$)", re.I)
RULES = ("R1_exact", "R2_contiguous", "R3_8gram")
DL = "/workspace/work/dl"

IX: Optional[D.Index] = None


# ----------------------------------------------------------------------------------------------------------------- old filters (gemma_sft_pairs.py)
def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def ok(en: str, tg: str) -> bool:
    if not (3 <= len(en) <= 1500 and 3 <= len(tg) <= 2500):
        return False
    r = len(tg) / max(1, len(en))
    if not (0.25 <= r <= 4.5):
        return False
    if en.casefold() == tg.casefold() or URL.search(en) or URL.search(tg):
        return False
    letters = sum(c.isalpha() for c in en)
    if letters / len(en) < 0.4 or not re.search(r"[A-Za-z]{2}", en):
        return False
    return sum(c.isalpha() for c in tg) / len(tg) >= 0.35


def key_of(en: str, iso: str, tg: str) -> int:
    return xxhash.xxh64_intdigest((en.casefold() + "\x01" + iso + "\x01" + tg.casefold()).encode("utf-8", "surrogatepass"))


# ----------------------------------------------------------------------------------------------------------------- licences (gemma_merge_pairs.py)
def lic_class(raw) -> str:
    s = re.sub(r"[ _]+", "-", (raw or "").lower().strip())
    if not s or s in {"none", "unknown", "unspecified"}:
        return "unspecified"
    if re.search(r"(^|-)nc(-|$)|non-?commercial", s):
        return "non_commercial"
    if re.search(r"-sa|share-?alike|gfdl|odbl|gpl", s):
        return "share_alike"
    for pat, name in ((r"cc0|public.?domain", "cc0"), (r"apache", "apache"), (r"\bmit\b", "mit"), (r"cc-by", "cc-by"), (r"odc-by", "odc-by"), (r"mpl", "mpl")):
        if re.search(pat, s):
            return name
    return "other"


SCAN = {}


def load_scan(path: str):
    global SCAN
    if os.path.exists(path):
        SCAN = json.load(open(path))["datasets"]


def licence_of(src: str, declared: Optional[str] = None) -> str:
    if declared:
        return lic_class(declared)
    v = SCAN.get((src or "").split(":")[0])
    return lic_class(v.get("declared")) if v else "unspecified"


# ----------------------------------------------------------------------------------------------------------------- readers
def fetch(repo: str, fname: str, revision: Optional[str] = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fname, repo_type="dataset", revision=revision, token=os.environ["HF_TOKEN"], local_dir=os.path.join(DL, repo.replace("/", "__")))


def get(row: dict, key: str):
    v = row
    for k in key.split("."):
        if isinstance(v, str):
            try:
                v = json.loads(v.replace("'", '"'))
            except Exception:
                return None
        if not isinstance(v, dict):
            return None
        v = v.get(k)
    return v


def rows_of(path: str, cols: Optional[List[str]] = None, remote: Optional[str] = None) -> Iterator[dict]:
    """Rows of a data file. Parquet is streamed by batches with column projection (local file, or remote path 'datasets/<repo>/<file>' read by range requests)."""
    low = path.lower()
    if low.endswith(".parquet"):
        if remote:
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem(token=os.environ["HF_TOKEN"])
            f = fs.open(remote, "rb", block_size=1 << 20, cache_type="readahead")
            pf = pq.ParquetFile(f)
        else:
            f = None
            pf = pq.ParquetFile(path)
        use = [c for c in (cols or pf.schema_arrow.names) if c in pf.schema_arrow.names]
        for b in pf.iter_batches(batch_size=20000, columns=use):
            d = b.to_pydict()
            for i in range(b.num_rows):
                yield {c: d[c][i] for c in use}
        if f:
            f.close()
    elif low.endswith((".csv", ".tsv")):
        csv.field_size_limit(10 ** 8)
        with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t" if low.endswith(".tsv") else ","):
                yield r
    elif low.endswith(".jsonl"):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("{"):
                    try:
                        yield json.loads(line)
                    except Exception:
                        pass
    elif low.endswith(".json"):
        with open(path, encoding="utf-8", errors="replace") as fh:
            try:
                d = json.load(fh)
            except Exception:
                fh.seek(0)
                d = [json.loads(l) for l in fh if l.strip().startswith("{")]
        if isinstance(d, dict):
            d = d.get("data") or d.get("rows") or list(d.values())[0]
        for r in d:
            if isinstance(r, dict):
                yield r


# ----------------------------------------------------------------------------------------------------------------- units
MT560_ALIAS = {"ganda": "lug", "gikuyu": "kik", "dholuo": "luo", "northern-sotho": "nso", "nigerian-pidgin": "pcm", "tigrigna": "tir", "rundi": "run", "oshiwambo": "kua",
               "sidaama": "sid", "jula": "dyu", "kikongo": "kon", "moore": "mos", "merina-malagasy": "mlg", "cameroon-pidgin": "wes", "edo": "bin"}
PONTOON = {"af": "afr", "am": "amh", "ee": "ewe", "ha": "hau", "ig": "ibo", "rw": "kin", "ny": "nya", "om": "orm", "sn": "sna", "st": "sot",
           "sw": "swh", "tn": "tsn", "tw": "twi", "wo": "wol", "xh": "xho", "yo": "yor", "zu": "zul", "so": "som"}
VUK = {"afr": "afr", "nbl": "nbl", "nso": "nso", "sot": "sot", "ssw": "ssw", "tsn": "tsn", "tso": "tso", "ven": "ven", "xho": "xho", "zul": "zul"}


def make_units(api, only: Optional[List[str]] = None) -> List[dict]:
    U: List[dict] = []
    # 1. processed rows of the first SFT build (translation-* groups)
    info = api.dataset_info((os.environ['HF_ORG'] + "/sft-new"))
    for s in sorted(info.siblings, key=lambda s: s.rfilename):
        if re.match(r"data/translation-[^/]+/part-\d+\.parquet$", s.rfilename):
            U.append(dict(id="sftnew:" + s.rfilename[5:-8].replace("/", "_"), kind="sftnew", repo=(os.environ['HF_ORG'] + "/sft-new"), file=s.rfilename))
    # 2. public equivalents of the OPUS-MT560 bundles (michsethowusu/english-<language>_sentence-pairs_mt560, cards: OPUS MT560, cc-by-4.0)
    names = {l.name.lower().replace(" ", "-"): l.iso for l in GL.LANGUAGES.values()}
    for r in sorted(d.id for d in api.list_datasets(author="michsethowusu", search="mt560", limit=500)):
        if not r.endswith("_sentence-pairs_mt560"):
            continue
        n = r.split("/")[1][len("english-"):-len("_sentence-pairs_mt560")]
        iso = MT560_ALIAS.get(n) or names.get(n)
        if iso:
            U.append(dict(id="mt560:" + n, kind="tab", repo=r, pick=r"data/train.*\.parquet$", en="eng", tg=n_to_col(n, iso), iso=iso, licence="cc-by-4.0"))
    # 3. other reachable public pair sources (schemas probed 2026-10-28)
    U += [
        dict(id="luo_opus", kind="tab", repo="BrianMwangi/english-to-luo-opus", pick=r"\.csv$", en="english_clean", tg="luo_clean", iso="luo", licence="cc-by-4.0"),
        dict(id="pristine_twi", kind="tab", repo="fiifinketia/pristine-twi-english", pick=r"train.*\.parquet$", en="english_translation", tg="twi", iso="twi", licence="cc-by-4.0",
             force_quarantine="re-upload declares CC-BY-4.0 but the upstream ghananlpcommunity/pristine-twi-english is CC-BY-NC-4.0 (old licence scan: non_commercial, held): owner to decide"),
        dict(id="luhya", kind="tab", repo="mamakobe/luhya-multilingual-dataset", pick=r"data/train.*\.parquet$", en="english_text", tg="luhya_text", iso="luy", licence="cc-by-4.0"),
        dict(id="kweku_eng_twi", kind="tab", repo="KwekuAinoo/eng_twi", pick=r"\.csv$", en="ENGLISH", tg="TWI", iso="twi", licence=None),
        dict(id="wolof_sudoping", kind="tab", repo="sudoping01/english-wolof-translation", pick=r"\.json$", en=None, tg=None, iso="wol", licence=None, autocols=True),
        dict(id="wolof_bible", kind="tab", repo="cibfaye/wolof-english-bible", pick=r"data/train.*\.parquet$", en="translation.en", tg="translation.wo", iso="wol", licence=None),
        dict(id="fon_helsinki", kind="tab", repo="beethogedeon/Helsinki-en-fon-opus-20k", pick=r"data/train.*\.parquet$", en="source_text", tg="translated_text", iso="fon", licence=None, lang_col=("source_lang", "en")),
        dict(id="synthetic_external_lug", kind="remote", repo="yigagilbert/synthetic-parallel-external", pick=r"train.*\.parquet$", en="text_eng", tg="text_lug", iso="lug", licence="cc-by-4.0",
             cols=["text_eng", "text_lug", "source_dataset", "source_subset"]),
        dict(id="afrolingu_train", kind="afrolingu", repo="UBC-NLP/AfroLingu-MT", file="AfroLinfu-MT-train.jsonl"),
        # owner decision 2026-10-28: SSA-MTE TRAIN human references (dev/test stay protected); CC-BY-SA-4.0 -> licence class share_alike
        dict(id="ssa_mte_train", kind="ssamte", repo="McGill-NLP/SSA-MTE", licence="cc-by-sa-4.0"),
        # owner decision 2026-10-28 (update): IbomMT train is CC-BY-NC and IS used: main output, licence column = non_commercial (filterable), same decontamination; the only deliberate NC source
        dict(id="ibom_train:efi", kind="tab", repo="howard-nlp/ibom-mt", pick=r"^eng-efi/train\.csv$", en="eng", tg="efi", iso="efi", licence="cc-by-nc-4.0", allow_nc=True),
        dict(id="ibom_train:ibb", kind="tab", repo="howard-nlp/ibom-mt", pick=r"^eng-ibb/train\.csv$", en="eng", tg="ibb", iso="ibb", licence="cc-by-nc-4.0", allow_nc=True),
        dict(id="ibom_train:davlan_efi", kind="tab", repo="Davlan/ibom-mt-en-efi", pick=r"^train\.json$", en=None, tg=None, iso="efi", licence="cc-by-nc-4.0", allow_nc=True, autocols=True),
    ]
    for loc, iso in PONTOON.items():
        U.append(dict(id=f"pontoon:{loc}", kind="tab", repo="ayymen/Pontoon-Translations", pick=rf"^en-{loc}\.tsv$", en="source_string", tg="target_string", iso=iso, licence=None))
    for code, iso in VUK.items():
        pick = rf"^afr-eng/train.*\.parquet$" if code == "afr" else rf"^eng-{code}/train.*\.parquet$"
        U.append(dict(id=f"vukuzenzele:{code}", kind="tab", repo="dsfsi/vukuzenzele-sentence-aligned", pick=pick, en="eng", tg=code, iso=iso, licence=None))
    if only:
        U = [u for u in U if any(u["id"].startswith(o) for o in only)]
    return U


SSA_LANGS = {"Amharic": "amh", "Ewe": "ewe", "Hausa": "hau", "Igbo": "ibo", "Kikuyu": "kik", "Kinyarwanda": "kin", "Lingala": "lin", "Luo": "luo", "Nyanja": "nya",
             "Sesotho": "sot", "Swahili": "swh", "Twi": "twi", "Wolof": "wol", "Yoruba": "yor", "Zulu": "zul", "Emakhuwa": "vmw"}
_EN_WORDS = set("the of and to in is that for it as with was on be by this are or from at an have not but they you he she we his her their will would can has been which there when what who".split())


def looks_english(s: str) -> bool:
    """Cheap English check (no LID model on the box): >= 2 English function words and >= 15% of the tokens. Used only where the source language of a file is not guaranteed (SSA-MTE has French / Portuguese originals)."""
    t = re.findall(r"[a-z']+", s.lower())
    if len(t) < 3:
        return False
    h = sum(1 for w in t if w in _EN_WORDS)
    return h >= 2 and h / len(t) >= 0.15


def n_to_col(n: str, iso: str) -> str:
    """Column of the target language in an mt560 repo = the language slug as used by michsethowusu: probed at read time (any non-'eng' string column)."""
    return "*"


def gen_pairs(u: dict, api) -> Iterator[Tuple[str, str, str, str, Optional[str], bool]]:
    """-> (english, target, raw_iso, source_dataset, declared_licence, flipped)"""
    k = u["kind"]
    if k == "sftnew":
        p = fetch(u["repo"], u["file"])
        try:
            for r in rows_of(p, ["source_lang", "targ_lang", "source_dataset", "input", "response"]):
                s, g, a, b = r["source_lang"], r["targ_lang"], r["input"] or "", r["response"] or ""
                if s == "eng" and g not in NOT_AFRICAN:
                    yield a, b, g, r["source_dataset"] or u["repo"], None, False
                elif g == "eng" and s not in NOT_AFRICAN:
                    yield b, a, s, r["source_dataset"] or u["repo"], None, True
        finally:
            os.remove(p)
    elif k == "ssamte":
        seen = set()
        for lang, iso in SSA_LANGS.items():
            p = fetch(u["repo"], f"train/{lang}.json")
            try:
                for r in rows_of(p.replace(".json", ".jsonl") if False else p):
                    en, ref = r.get("src_sent"), r.get("ref")
                    if not isinstance(en, str) or not isinstance(ref, str) or (r.get("input_id"), iso) in seen:
                        continue
                    seen.add((r.get("input_id"), iso))
                    if not looks_english(en):
                        continue
                    yield en, ref, iso, u["repo"], u.get("licence"), False
            finally:
                os.remove(p)
    elif k == "afrolingu":
        p = fetch(u["repo"], u["file"])
        try:
            for r in rows_of(p):
                a_, _, b_ = str(r.get("langcode") or "").partition("-")
                if a_ == "eng" and b_:
                    yield r.get("input") or "", r.get("output") or "", b_, u["repo"], None, False
                elif b_ == "eng" and a_:
                    yield r.get("output") or "", r.get("input") or "", a_, u["repo"], None, True
        finally:
            os.remove(p)
    elif k in ("tab", "remote"):
        info = api.dataset_info(u["repo"])
        files = sorted(s.rfilename for s in info.siblings if re.search(u["pick"], s.rfilename) and s.rfilename.lower().endswith((".parquet", ".csv", ".tsv", ".json", ".jsonl")))
        if not files:
            raise RuntimeError("no file matches " + u["pick"])
        for f in files:
            if QUARANTINE_SRC.search("/" + f.split("/")[-1]) and not re.search(r"train", f):
                continue
            if k == "remote":
                it = rows_of(f, u.get("cols"), remote=f"datasets/{u['repo']}/{f}")
                p = None
            else:
                p = fetch(u["repo"], f)
                it = rows_of(p)
            try:
                en_c, tg_c = u.get("en"), u.get("tg")
                for i, r in enumerate(it):
                    if i == 0 and (tg_c == "*" or u.get("autocols")):
                        keys = [c for c in r if isinstance(r[c], str)]
                        en_c = "eng" if "eng" in r else next((c for c in keys if c.lower().startswith(("en", "eng"))), keys[0])
                        tg_c = next(c for c in keys if c != en_c and c.lower() not in ("score", "id", "__index_level_0__"))
                        u["_cols"] = (en_c, tg_c)
                    a, b = get(r, en_c), get(r, tg_c)
                    if not isinstance(a, str) or not isinstance(b, str):
                        continue
                    if u.get("lang_col") and r.get(u["lang_col"][0]) != u["lang_col"][1]:
                        continue
                    sub = r.get("source_dataset") or ""
                    sub = f"{u['repo']}:{sub}:{r.get('source_subset') or ''}" if (sub or r.get("source_subset")) else u["repo"]
                    yield a, b, u["iso"], sub, u.get("licence"), False
            finally:
                if p and os.path.exists(p):
                    os.remove(p)
    else:
        raise ValueError(k)


PROV: Dict[str, dict] = {}
PROV_RE = re.compile(r"nllb|no language left behind|translate\s?psy", re.I)


def prov_flag(src: str) -> Optional[str]:
    """Repo-card scan: a source whose card mentions NLLB or TranslatePsy goes to quarantine_nc/ (owner rule 2026-10-23/28), never into the main output."""
    repo = (src or "").split(":")[0]
    if repo not in PROV:
        info = {"flag": None, "card": "n/a"}
        if re.match(r"^[\w.\-]+/[\w.\-]+$", repo):
            try:
                from huggingface_hub import hf_hub_download
                t = open(hf_hub_download(repo, "README.md", repo_type="dataset", token=os.environ["HF_TOKEN"], local_dir=os.path.join(DL, "_cards", repo.replace("/", "__")))).read()
                m = PROV_RE.search(t)
                info = {"flag": ("card mentions " + m.group(0)) if m else None, "card": "read", "mentions": len(PROV_RE.findall(t))}
            except Exception as e:  # noqa: BLE001
                info = {"flag": None, "card": type(e).__name__}
        PROV[repo] = info
    return PROV[repo]["flag"]


# ----------------------------------------------------------------------------------------------------------------- decontamination workers
def _check_chunk(args):
    ens, tgs = args
    a = IX.check_texts(ens)
    b = IX.check_texts(tgs)
    out = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x[0] or x[1] or x[2] or y[0] or y[1] or y[2]:
            out.append((i, x, y))
    return out


def run_unit(u: dict, api, pool, out: str, qsize: int = 4000) -> dict:
    st = collections.Counter()
    by_iso_ok, by_iso_drop = collections.Counter(), collections.Counter()
    by_iso_src_ok, by_iso_src_drop = collections.Counter(), collections.Counter()
    rule_set_side = collections.Counter()      # "rule|set|side" -> rows
    first_rule_iso = collections.Counter()      # "R1|iso" first-hit rule by language
    only_r3_iso = collections.Counter()
    lang_unknown = collections.Counter()
    pairs: List[tuple] = []
    for en, tg, raw_iso, src, decl, flipped in gen_pairs(u, api):
        st["rows_in"] += 1
        if not isinstance(en, str) or not isinstance(tg, str):
            continue
        if QUARANTINE_SRC.search(src):
            st["quarantined_source_dropped"] += 1
            continue
        iso = GL.canonical_iso(raw_iso)
        if iso is None:
            lang_unknown[raw_iso] += 1
            continue
        lic = licence_of(src, decl)
        reason = ""
        if EXCLUDE_SRC.search(src):
            reason = "source name: NLLB / TranslatePsy / held-out Aya translation"
        elif prov_flag(src):
            reason = "provenance: " + prov_flag(src)
        elif u.get("force_quarantine"):
            reason = "licence doubt: " + u["force_quarantine"]
        elif lic == "non_commercial" and not u.get("allow_nc"):
            reason = "non-commercial licence (" + str(decl or lic) + ")"
        en, tg = norm_ws(en), norm_ws(tg)
        if not ok(en, tg):
            st["fail_ok_filter"] += 1
            continue
        pairs.append((en, tg, iso, src, lic, bool(flipped), reason))
    st["pairs_ok"] = sum(1 for p_ in pairs if not p_[6])
    # decontaminate in chunks across the workers
    chunks = [([p[0] for p in pairs[i:i + qsize]], [p[1] for p in pairs[i:i + qsize]]) for i in range(0, len(pairs), qsize)]
    drop = np.zeros(len(pairs), np.uint8)
    for ci, res in enumerate(pool.imap(_check_chunk, chunks, chunksize=1)):
        base = ci * qsize
        for i, x, y in res:
            j = base + i
            code = 0
            sides = (8 if (x[0] or x[1] or x[2]) else 0) | (16 if (y[0] or y[1] or y[2]) else 0)
            for ridx, rname in enumerate(RULES):
                m_x, m_y = x[ridx], y[ridx]
                if m_x or m_y:
                    code |= 1 << ridx
                for sidemask, side in ((m_x, "en"), (m_y, "tg")):
                    for nm in IX.names(sidemask):
                        rule_set_side[f"{'Q_' if pairs[j][6] else ''}{rname}|{nm}|{side}"] += 1
            drop[j] = code | sides
    iso_i = collections.Counter()
    keys = np.empty(len(pairs), np.uint64)
    meta_iso, meta_src, meta_tlen, meta_lic, meta_q = [], [], [], [], []
    kept = collections.defaultdict(list)
    qk = collections.defaultdict(list)
    for j, (en, tg, iso, src, lic, fl, reason) in enumerate(pairs):
        keys[j] = key_of(en, iso, tg)
        meta_lic.append(lic); meta_q.append(1 if reason else 0)
        if reason:
            st['quarantine_pairs_ok'] += 1
        if not reason:
            by_iso_ok[iso] += 1
            by_iso_src_ok[f"{iso}|{src}"] += 1
        meta_iso.append(iso); meta_src.append(src); meta_tlen.append(len(tg))
        d = int(drop[j])
        if d & 7:
            if reason:
                st["quarantine_removed_by_filter"] += 1
                continue
            by_iso_drop[iso] += 1
            by_iso_src_drop[f"{iso}|{src}"] += 1
            first = "R1_exact" if d & 1 else ("R2_contiguous" if d & 2 else "R3_8gram")
            first_rule_iso[f"{first}|{iso}"] += 1
            if not (d & 3):
                only_r3_iso[iso] += 1
            continue
        tgt = qk if reason else kept
        tgt["english"].append(en); tgt["target"].append(tg); tgt["iso"].append(iso); tgt["source_dataset"].append(src)
        tgt["licence"].append(lic); tgt["flipped"].append(fl); tgt["key"].append(int(keys[j]))
        if reason:
            qk["reason"].append(reason)
    uid = u["id"].replace("/", "_").replace(":", "_")
    os.makedirs(f"{out}/kept", exist_ok=True)
    os.makedirs(f"{out}/meta", exist_ok=True)
    if kept["english"]:
        t = pa.table({**{c: kept[c] for c in ("english", "target", "iso", "source_dataset", "licence", "flipped")}, "key": pa.array(kept["key"], pa.uint64())})
        pq.write_table(t, f"{out}/kept/{uid}.parquet", compression="zstd")
    if qk["english"]:
        os.makedirs(f"{out}/quarantine_nc", exist_ok=True)
        pq.write_table(pa.table({**{c: qk[c] for c in ("english", "target", "iso", "source_dataset", "licence", "flipped", "reason")}, "key": pa.array(qk["key"], pa.uint64())}),
                       f"{out}/quarantine_nc/{uid}.parquet", compression="zstd")
    if len(pairs):
        pq.write_table(pa.table({"key": pa.array(keys, pa.uint64()), "iso": meta_iso, "src": meta_src, "lic": meta_lic, "quar": pa.array(meta_q, pa.uint8()),
                                 "drop": pa.array(drop, pa.uint8()), "tlen": pa.array(meta_tlen, pa.uint32())}), f"{out}/meta/{uid}.parquet", compression="zstd")
    st["kept"] = len(kept["english"])
    st["quarantine_kept"] = len(qk["english"])
    return {"unit": u["id"], "repo": u.get("repo"), "stats": dict(st), "cols": u.get("_cols"), "by_iso_ok": dict(by_iso_ok), "by_iso_drop": dict(by_iso_drop),
            "by_iso_src_ok": dict(by_iso_src_ok), "by_iso_src_drop": dict(by_iso_src_drop), "rule_set_side": dict(rule_set_side),
            "first_rule_iso": dict(first_rule_iso), "only_r3_iso": dict(only_r3_iso), "lang_unknown": dict(lang_unknown), "time": time.strftime("%H:%M:%S")}


# ----------------------------------------------------------------------------------------------------------------- push loop
def push_loop(api, repo: str, out: str, stop: threading.Event, every: int = 600):
    def once():
        try:
            api.upload_folder(folder_path=out, path_in_repo="rebuild", repo_id=repo, repo_type="dataset", allow_patterns=["kept/*", "meta/*", "quarantine_nc/*", "ledger.jsonl", "prov_flags.json"],
                              commit_message="clean sft shards (progress)")
        except Exception as e:  # noqa: BLE001
            print("push failed (will retry):", type(e).__name__, str(e)[:120], flush=True)
    while not stop.wait(every):
        once()
    once()


def main():
    global IX
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--scan", default="/workspace/work/license_scan.json")
    ap.add_argument("--repo", default=(os.environ['HF_ORG_LEGACY'] + "/sabiyarn-clean-sft-2026-10-28"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    load_scan(a.scan)
    units = make_units(api, a.only)
    if a.list:
        for u in units:
            print(u["id"], u.get("repo"), u.get("file", u.get("pick")))
        return
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(DL, exist_ok=True)
    led = f"{a.out}/ledger.jsonl"
    done = set()
    if os.path.exists(led):
        done = {json.loads(l)["unit"] for l in open(led) if l.strip()}
    IX = D.Index.load(a.index)
    print(f"index: {len(IX.s_keys):,} sentence hashes, {len(IX.g_keys):,} 8-gram hashes, sets {len(IX.set_names)}; units {len(units)}, done {len(done)}", flush=True)
    ctx = mp.get_context("fork")
    pool = ctx.Pool(a.workers)
    stop = threading.Event()
    th = None
    if not a.no_push:
        api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
        th = threading.Thread(target=push_loop, args=(api, a.repo, a.out, stop), daemon=True)
        th.start()
    for i, u in enumerate(units):
        if u["id"] in done:
            continue
        t0 = time.time()
        try:
            rec = run_unit(u, api, pool, a.out)
        except Exception as e:  # noqa: BLE001
            print(f"[{i+1}/{len(units)}] {u['id']} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            with open(f"{a.out}/failed.jsonl", "a") as fh:
                fh.write(json.dumps({"unit": u["id"], "error": f"{type(e).__name__}: {str(e)[:300]}"}) + "\n")
            continue
        with open(led, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        json.dump(PROV, open(f"{a.out}/prov_flags.json", "w"), indent=1)
        s = rec["stats"]
        print(f"[{i+1}/{len(units)}] {u['id']}: in {s.get('rows_in',0):,} ok {s.get('pairs_ok',0):,} kept {s.get('kept',0):,} "
              f"(removed {s.get('pairs_ok',0)-s.get('kept',0):,}) {time.time()-t0:.0f}s", flush=True)
    stop.set()
    if th:
        th.join()
    print("all units finished", flush=True)


if __name__ == "__main__":
    main()
