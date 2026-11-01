#!/usr/bin/env python3
"""intake2_instr: instruction / chat / QA intake for the clean SFT rerun (2026-10-31). Runs ON THE INSTANCE (CPU only, never touches the GPU).

Stream (nothing raw is kept) African-language instruction / chat / QA datasets from the Hub, convert each row to a chat
conversation (`messages` = JSON list of {role: user|assistant, optional first system}), clean, decontaminate against the protected
index (STRICT: a row is dropped when ANY turn has ANY hit of R1/R2/R3, exactly the semantics of extra_sources.Filter / decon.py),
exact-dedupe on the normalised conversation (within a source, and across sources: the first source in the priority order keeps the row),
cap 400,000 rows per (source, language) and write ~100k-row shards:

    /workspace/work/instr/<source_name>/part-NNNN.parquet
    columns: id, source_dataset, iso, messages, task_type, licence, licence_raw, synthetic, n_chars

    python intake2_instr.py list                                   # sources in priority order
    python intake2_instr.py test --sources a,b --n 1000            # 1,000 raw rows per source (first file), full cleaning + decon, prints reject reasons + 3 samples, writes nothing final
    python intake2_instr.py run  --sources a,b --workers 24       # real run (resumable: a finished source has _state/<source>.done)
    python intake2_instr.py tokens                                 # Gemma token estimate per (source, language), 2,000-row sample per language
    python intake2_instr.py aya                                    # alignment verdict of the Aya collection language splits vs English (200 rows)

Licence classes: commercial_ok | share_alike | unspecified (= no licence declared; disclose). Non-commercial and NLLB/TranslatePsy-derived data are
never added here. `synthetic` = machine-translated or LLM-generated on at least the instruction side (see each source's note in SOURCES).
"""
import argparse, collections, csv, glob, io, json, multiprocessing as mp, os, re, shutil, sys, threading, time, traceback, unicodedata

import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import xxhash

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE), "/workspace/work/code", "/workspace/work/code/sft_clean", "/workspace/work/code/sft_clean/chain"):
    if p not in sys.path:
        sys.path.insert(0, p)

NEW_OK = set()          # filled by intake2_sources.register (African languages outside languages.py, listed as NEW in the report)
OUT = os.environ.get("INSTR_OUT", "/workspace/work/instr")
TMP = os.path.join(OUT, "_tmp")
STATE = os.path.join(OUT, "_state")
CAND = os.path.join(OUT, "_cand")
PROT_INDEX = "/workspace/work/work/_hub_protected/protected_index/protected_index.npz"
PROT_MANIFEST = "/workspace/work/work/_hub_protected/protected_index/protected_manifest.json"
PROGRESS = "/workspace/work/logs/intake2_instr.progress"
CAP = 400_000
SHARD = 100_000
MIN_USER, MIN_ASSIST, MAX_CHARS = 8, 2, 12_000
BATCH = 4000

# ------------------------------------------------------------------------------------------------------------------------ languages
try:
    from gemma_mt import languages as GL
    OUR = set(GL.ALL_ISOS)
except Exception:                                    # pragma: no cover
    OUR = set()

ISO_MAP = {  # 2-letter / 3-letter variants / names -> our tag (languages.py) ; anything else African is kept as its own iso3 (listed as NEW in the report)
    "af": "afr", "am": "amh", "ha": "hau", "ig": "ibo", "yo": "yor", "sw": "swh", "swa": "swh", "swc": "swc", "so": "som", "rw": "kin", "zu": "zul", "xh": "xho",
    "sn": "sna", "ny": "nya", "chichewa": "nya", "nyanja": "nya", "st": "sot", "tn": "tsn", "mg": "mlg", "plt": "mlg", "mlg": "mlg", "om": "orm", "gaz": "orm", "ti": "tir",
    "wo": "wol", "lg": "lug", "ln": "lin", "ff": "ful", "fuv": "ful", "fub": "ful", "ful": "ful", "fuf": "ful", "aka": "twi", "ak": "twi", "tw": "twi", "ee": "ewe", "rn": "run",
    "ss": "ssw", "ts": "tso", "ve": "ven", "nr": "nbl", "nd": "nde", "kg": "kon", "ki": "kik", "ku": "kur", "sg": "sag", "bm": "bam", "kau": "knc", "kr": "knc",
    "pcm": "pcm", "yor": "yor", "hau": "hau", "ibo": "ibo", "kin": "kin", "ewe": "ewe", "wol": "wol", "tsn": "tsn", "sot": "sot", "nso": "nso", "xho": "xho", "zul": "zul",
    "afrikaans": "afr", "amharic": "amh", "hausa": "hau", "igbo": "ibo", "yoruba": "yor", "swahili": "swh", "somali": "som", "kinyarwanda": "kin", "shona": "sna", "oromo": "orm",
    "tigrinya": "tir", "wolof": "wol", "luganda": "lug", "lingala": "lin", "zulu": "zul", "xhosa": "xho", "setswana": "tsn", "tswana": "tsn", "sesotho": "sot", "malagasy": "mlg",
    "twi": "twi", "akan": "twi", "ewe_": "ewe", "bambara": "bam", "pidgin": "pcm", "nigerian pidgin": "pcm", "kirundi": "run", "tumbuka": "tum", "bemba": "bem",
    "fon": "fon", "fongbe": "fon", "efik": "efi", "tsonga": "tso", "venda": "ven", "swati": "ssw", "kikuyu": "kik", "kongo": "kon", "sepedi": "nso", "northern sotho": "nso",
    "egyptian arabic": "arz", "moroccan arabic": "ary", "tunisian arabic": "aeb", "arz": "arz", "ary": "ary", "aeb": "aeb",
}
NOT_AFRICAN = {"eng", "en", "fra", "fr", "deu", "spa", "por", "ita", "zho", "jpn", "ara", "rus", "hin", "ind", "tur", "vie", "kor", "nld", "pol", "english", "french"}


def norm_iso(code):
    if code is None:
        return None
    c = str(code).strip().lower().replace("_latn", "").replace("_ethi", "").replace("_arab", "")
    c = ISO_MAP.get(c, c)
    return c


# ------------------------------------------------------------------------------------------------------------------------ text cleaning / checks
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​  ﻿]")
WS_TRAIL = re.compile(r"[ \t]+\n")
MANYNL = re.compile(r"\n{4,}")
EN_STOP = frozenset("the and of is was are that with for this you have it not but from they will be by an at which or what your can been were their would there about into them than more".split())
JUNK = re.compile(r"Error code: \d{3}|invalid_request_error|credit balance is too low|<\|im_(?:start|end)\|>|<start_of_turn>|<\|eot_id\|>|<\|begin_of_text\|>|Traceback \(most recent call last\)")
TOK = re.compile(r"[^\W\d_]+", re.UNICODE)
ETH = re.compile(r"[ሀ-᎟ⶀ-⷟꬀-꬯]")
ARB = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
LAT = re.compile(r"[A-Za-zÀ-ɏḀ-ỿɐ-ʯ]")
SCRIPT = {"amh": ETH, "tir": ETH, "tig": ETH, "sid": ETH, "wal": ETH, "ary": ARB, "arz": ARB, "aeb": ARB, "ara": ARB}
NO_SCRIPT_CHECK = {"shi", "tzm", "ber", "zgh", "nqo", "kab", "eng"}


def clean(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = CTRL.sub("", s)
    s = s.replace("#VALUE!", "")
    s = WS_TRAIL.sub("\n", s)
    s = MANYNL.sub("\n\n\n", s)
    return s.strip()


def en_share(t):
    toks = TOK.findall(t[:3000].lower())
    if len(toks) < 6:
        return 0.0
    return sum(1 for w in toks if w in EN_STOP) / len(toks)


def script_ok(iso, text):
    """cheap script check of the assistant text: Ethiopic/Arabic languages must be mostly that script, everything else mostly Latin (Tifinagh/N'Ko languages are not checked)."""
    if iso in NO_SCRIPT_CHECK:
        return True
    letters = [c for c in text[:2000] if c.isalpha()]
    if len(letters) < 12:
        return True
    rx = SCRIPT.get(iso)
    if rx is not None:
        return sum(1 for c in letters if rx.match(c)) / len(letters) >= 0.5
    return sum(1 for c in letters if LAT.match(c)) / len(letters) >= 0.7


def norm_key(t):
    t = unicodedata.normalize("NFKC", t).casefold()
    return re.sub(r"[\W_]+", " ", t).strip()


def conv_key(msgs):
    """exact-dedupe key: xxh64 of the normalised (user text, assistant text) of the whole conversation (system turn excluded)"""
    u = "\x02".join(norm_key(m[1]) for m in msgs if m[0] == "user")
    a = "\x02".join(norm_key(m[1]) for m in msgs if m[0] == "assistant")
    return xxhash.xxh64_intdigest((u + "\x01" + a).encode("utf-8", "surrogatepass"))


def fix_turns(turns):
    """turns = [(role, text)] -> valid alternating list starting (after an optional system) with user and ending with assistant; None when impossible"""
    out = []
    sysm = None
    for r, t in turns:
        r = {"human": "user", "gpt": "assistant", "model": "assistant", "bot": "assistant", "system": "system", "user": "user", "assistant": "assistant"}.get(str(r).lower())
        t = clean(t)
        if r is None or not t:
            if r is None:
                return None
            continue
        if r == "system":
            if not out and sysm is None:
                sysm = t
            continue
        if out and out[-1][0] == r:
            out[-1] = (r, out[-1][1] + "\n\n" + t)
        else:
            out.append((r, t))
    while out and out[0][0] != "user":
        out = out[1:]
    while out and out[-1][0] != "assistant":
        out = out[:-1]
    if len(out) < 2:
        return None
    return ([("system", sysm)] if sysm else []) + out


# ------------------------------------------------------------------------------------------------------------------------ row shapes (row dict, path) -> list of (role, text)
def S_instr(ins="instruction", inp="input", out="output", join="\n\n", prefix=None):
    def f(r, path):
        u = clean(r.get(ins))
        x = clean(r.get(inp)) if inp else ""
        if x:
            u = (u + join + x) if u else x
        return [("user", u), ("assistant", clean(r.get(out)))]
    return f


def S_qa(q="question", a="answer", ctx=None):
    def f(r, path):
        u = clean(r.get(q))
        if ctx and clean(r.get(ctx)):
            u = clean(r.get(ctx)) + "\n\n" + u
        return [("user", u), ("assistant", clean(r.get(a)))]
    return f


def S_msgs(col="messages", rk="role", ck="content"):
    def f(r, path):
        m = r.get(col)
        if isinstance(m, str):
            try:
                m = json.loads(m)
            except Exception:
                return None
        if not isinstance(m, (list, tuple)):
            return None
        out = []
        for t in m:
            if not isinstance(t, dict):
                return None
            role = t.get(rk) if rk in t else t.get("from")
            cont = t.get(ck) if ck in t else t.get("value")
            out.append((role, cont))
        return out
    return f


CHATML = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.S)


def S_chatml(col="text", strip_think=True):
    def f(r, path):
        t = r.get(col)
        if not isinstance(t, str):
            return None
        out = []
        for role, body in CHATML.findall(t):
            if strip_think:
                body = re.sub(r"<think>.*?</think>", "", body, flags=re.S)
            out.append((role, body))
        return out or None
    return f


def S_pair(i="input", o="output"):
    return S_qa(i, o)


# ------------------------------------------------------------------------------------------------------------------------ sources
class Src:
    def __init__(self, name, repo, files, shape, iso, licence, licence_raw, synthetic=False, task="instruction", cap=CAP, note="", max_rows_file=None, rowfilter=None, task_fn=None, synth_fn=None, order=100, vs_old=False):
        self.name, self.repo, self.files, self.shape, self.iso = name, repo, files, shape, iso
        self.licence, self.licence_raw, self.synthetic, self.task, self.cap, self.note = licence, licence_raw, synthetic, task, cap, note
        self.max_rows_file, self.rowfilter, self.task_fn, self.synth_fn, self.order, self.vs_old = max_rows_file, rowfilter, task_fn, synth_fn, order, vs_old


SOURCES = collections.OrderedDict()


def add(*a, **k):
    s = Src(*a, **k)
    SOURCES[s.name] = s
    return s


def rx_files(pattern):
    r = re.compile(pattern)
    return lambda paths: [p for p in paths if r.search(p)]


def col_iso(col, mapping=None):
    def f(path, row):
        v = row.get(col)
        return norm_iso(mapping.get(v, v) if mapping else v)
    return f


def dir_iso(depth=0, mapping=None):
    def f(path, row):
        d = path.split("/")[depth]
        return norm_iso(mapping.get(d, d) if mapping else d)
    return f


def file_iso(mapping):
    def f(path, row):
        b = os.path.basename(path).split(".")[0].lower()
        for k, v in mapping.items():
            if k in b:
                return v
        return None
    return f


try:
    from intake2_instr_sources import register as _register_sources   # sources live in a second file next to this one (keeps this file reviewable)
    _register_sources(sys.modules[__name__])
except ImportError:
    pass

# ------------------------------------------------------------------------------------------------------------------------ Hub access
def hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def list_files(repo):
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token())
    return [(f.path, getattr(f, "size", 0) or 0) for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True) if hasattr(f, "size")]


def fetch(repo, path, wdir):
    from huggingface_hub import hf_hub_download
    for k in range(5):
        try:
            return hf_hub_download(repo, path, repo_type="dataset", local_dir=wdir, token=hf_token())
        except Exception as e:
            msg = str(e)
            if "401" in msg or "403" in msg or "gated" in msg.lower() or "404" in msg:
                raise
            time.sleep(5 * (k + 1))
    raise RuntimeError("download failed " + repo + "/" + path)


def iter_file_rows(local, limit=None, cols=None):
    n = 0
    low = local.lower()
    if low.endswith(".parquet"):
        pf = pq.ParquetFile(local)
        names = pf.schema_arrow.names
        use = [c for c in cols if c in names] if cols else None
        for rb in pf.iter_batches(batch_size=2000, columns=use):
            for r in rb.to_pylist():
                yield r
                n += 1
                if limit and n >= limit:
                    return
    elif low.endswith(".jsonl"):
        with open(local, encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield json.loads(ln)
                except Exception:
                    continue
                n += 1
                if limit and n >= limit:
                    return
    elif low.endswith(".json"):
        with open(local, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        try:
            d = json.loads(raw)
        except Exception:                                  # .json file that is really JSON lines
            d = []
            for ln in raw.splitlines():
                try:
                    d.append(json.loads(ln))
                except Exception:
                    pass
        del raw
        if isinstance(d, dict):
            for k in ("data", "train", "rows", "examples"):
                if isinstance(d.get(k), list):
                    d = d[k]
                    break
            else:
                d = list(d.values()) if all(isinstance(v, dict) for v in d.values()) else [d]
        for r in d:
            yield r
            n += 1
            if limit and n >= limit:
                return
    elif low.endswith((".csv", ".tsv")):
        csv.field_size_limit(10 ** 9)
        with open(local, encoding="utf-8-sig", errors="replace", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t" if low.endswith(".tsv") else ","):
                yield r
                n += 1
                if limit and n >= limit:
                    return
    else:
        raise ValueError("unsupported file type " + local)


# ------------------------------------------------------------------------------------------------------------------------ decontamination
_IX = None


def load_index():
    global _IX
    if _IX is None:
        import protected_index_loader as PL
        _IX, man = PL.load(PROT_INDEX, PROT_MANIFEST)
        print("protected index", man.get("version"), flush=True)
    return _IX


def protected_hits(ix, texts):
    """per text: None or the first rule that hits (R1 exact sentence, R2 contained protected sentence, R3 shared 8-gram)"""
    res = ix.check_texts(texts)
    out = []
    for z in res:
        h = None
        for rule, m in zip(("R1", "R2", "R3"), z[:3]):
            if m and h is None:
                h = rule
        out.append(h)
    return out


# ------------------------------------------------------------------------------------------------------------------------ one file -> candidate parquet
SCHEMA = pa.schema([("id", pa.string()), ("source_dataset", pa.string()), ("iso", pa.string()), ("messages", pa.string()), ("task_type", pa.string()), ("licence", pa.string()),
                    ("licence_raw", pa.string()), ("synthetic", pa.bool_()), ("n_chars", pa.int32()), ("dkey", pa.uint64())])


SCHEMA_L = pa.schema([(f.name, pa.large_string() if pa.types.is_string(f.type) else f.type) for f in SCHEMA])      # merge works on 64-bit offsets (a 600k-row Tumbuka source has > 2 GB of messages)


SCHEMA_OUT = pa.schema([f for f in SCHEMA if f.name != "dkey"])


def convert_row(src, r, path, idx):
    """-> (record | None, reject_reason)"""
    if src.rowfilter is not None and not src.rowfilter(r, path):
        return None, "rowfilter"
    iso = src.iso(path, r) if callable(src.iso) else src.iso
    if not iso:
        return None, "no_iso"
    if iso in NOT_AFRICAN:
        return None, "not_african"
    if iso not in OUR and iso not in NEW_OK:
        return None, "lang_not_in_scope"
    turns = src.shape(r, path)
    if not turns:
        return None, "bad_shape"
    msgs = fix_turns(turns)
    if not msgs:
        return None, "empty_turns"
    body = [m for m in msgs if m[0] != "system"]
    if len(body[0][1]) < MIN_USER:
        return None, "short_user"
    if any(m[0] == "assistant" and len(m[1]) < MIN_ASSIST for m in body):
        return None, "short_assistant"
    n = sum(len(m[1]) for m in msgs)
    if n > MAX_CHARS:
        return None, "too_long"
    a_all = " ".join(m[1] for m in body if m[0] == "assistant")
    if any(JUNK.search(m[1]) for m in msgs):
        return None, "junk_marker"
    if not script_ok(iso, a_all):
        return None, "wrong_script"
    if iso != "eng" and en_share(a_all) >= 0.25:
        return None, "assistant_english"
    task = src.task_fn(r, path) if src.task_fn else src.task
    syn = src.synth_fn(r, path) if src.synth_fn else src.synthetic
    rid = r.get("id") if isinstance(r.get("id"), (str, int)) else None
    rec = dict(id=f"{src.name}:{path.replace('/', '|')}:{idx}" + (f":{rid}" if rid is not None else ""), source_dataset=src.repo, iso=iso,
               messages=json.dumps([{"role": m[0], "content": m[1]} for m in msgs], ensure_ascii=False), task_type=task, licence=src.licence,
               licence_raw=src.licence_raw, synthetic=bool(syn), n_chars=n, dkey=conv_key(msgs))
    rec["_texts"] = [m[1] for m in msgs]
    return rec, None


def process_file(src, path, fidx, limit=None, ix=None, outdir=None, keep_samples=0):
    """download one repo file, convert + clean + decontaminate its rows, write the surviving candidates to a parquet; returns (stats, samples)"""
    ix = ix or load_index()
    wdir = os.path.join(TMP, f"{src.name}_{os.getpid()}")
    os.makedirs(wdir, exist_ok=True)
    stats = collections.Counter()
    by_lang = collections.Counter()
    samples = []
    local = None
    try:
        local = fetch(src.repo, path, wdir)
        recs, pending = [], []
        seen = set()

        def flush():
            if not pending:
                return
            texts = [t for rec in pending for t in rec["_texts"]]
            hits = protected_hits(ix, texts)
            k = 0
            for rec in pending:
                n = len(rec["_texts"])
                h = next((x for x in hits[k:k + n] if x), None)
                k += n
                if h:
                    stats["reject_protected_" + h] += 1
                    continue
                del rec["_texts"]
                recs.append(rec)
            pending.clear()

        for i, r in enumerate(iter_file_rows(local, limit=limit or src.max_rows_file)):
            stats["raw"] += 1
            rec, why = convert_row(src, r, path, i)
            if rec is None:
                stats["reject_" + why] += 1
                continue
            if rec["dkey"] in seen:
                stats["reject_dup_in_file"] += 1
                continue
            seen.add(rec["dkey"])
            pending.append(rec)
            if len(pending) >= BATCH:
                flush()
        flush()
        for rec in recs:
            by_lang[rec["iso"]] += 1
        if keep_samples:
            import random
            rr = random.Random(0)
            samples = rr.sample(recs, min(keep_samples, len(recs)))
        if outdir and recs:
            os.makedirs(outdir, exist_ok=True)
            tbl = pa.Table.from_pylist([{k: v for k, v in rec.items()} for rec in recs], schema=SCHEMA)
            pq.write_table(tbl, os.path.join(outdir, f"{fidx:05d}.parquet"), compression="zstd")
        stats["kept_candidates"] = len(recs)
    finally:
        shutil.rmtree(wdir, ignore_errors=True)
    return dict(stats), dict(by_lang), samples


def _worker(args):
    name, path, fidx = args
    src = SOURCES[name]
    try:
        st, bl, _ = process_file(src, path, fidx, ix=_IX, outdir=os.path.join(CAND, name))
        return name, path, st, bl, None
    except Exception as e:
        return name, path, {}, {}, f"{type(e).__name__}: {str(e)[:300]}"


# ------------------------------------------------------------------------------------------------------------------------ merge: cross-source dedupe, cap, shards
def load_seen():
    p = os.path.join(STATE, "seen_keys.npy")
    return np.load(p) if os.path.exists(p) else np.empty(0, np.uint64)


def save_seen(a):
    os.makedirs(STATE, exist_ok=True)
    tmp = os.path.join(STATE, "seen_keys.npy.tmp.npy")
    np.save(tmp, np.unique(a))
    os.replace(tmp, os.path.join(STATE, "seen_keys.npy"))


def merge_source(name, log=print):
    src = SOURCES[name]
    files = sorted(glob.glob(os.path.join(CAND, name, "*.parquet")))
    seen = np.sort(load_seen())
    op = os.path.join(STATE, "old_keys.npy")
    old = np.sort(np.load(op)) if (src.vs_old and os.path.exists(op)) else np.empty(0, np.uint64)
    stats = collections.Counter()
    per_lang = collections.defaultdict(list)       # iso -> list of tables
    for fp in files:
        t = pq.read_table(fp).cast(SCHEMA_L)
        stats["candidates"] += t.num_rows
        per_lang_ = collections.defaultdict(list)
        isos = t.column("iso").to_pylist()
        for i, iso in enumerate(isos):
            per_lang_[iso].append(i)
        for iso, idx in per_lang_.items():
            per_lang[iso].append(t.take(pa.array(idx)))
    out_tables, kept_lang, dropped = [], collections.Counter(), collections.Counter()
    new_keys = []
    for iso, tabs in sorted(per_lang.items()):
        t = pa.concat_tables(tabs)
        keys = t.column("dkey").to_numpy()
        # first occurrence inside the source keeps; then drop keys already kept by an earlier source
        _, first = np.unique(keys, return_index=True)
        mask = np.zeros(len(keys), bool)
        mask[first] = True
        stats["dup_within_source"] += int((~mask).sum())
        in_seen = np.isin(keys, seen)
        stats["dup_vs_earlier_sources"] += int((mask & in_seen).sum())
        mask &= ~in_seen
        if src.vs_old and len(old):
            in_old = np.isin(keys, old)
            stats["dup_vs_old_build_sft_all_new"] += int((mask & in_old).sum())
            mask &= ~in_old
        idx = np.nonzero(mask)[0]
        if len(idx) > src.cap:                       # deterministic, unbiased cap: lowest xxh64(id) first
            ids = t.column("id").to_pylist()
            h = np.array([xxhash.xxh64_intdigest(ids[i].encode('utf-8', 'surrogatepass')) for i in idx], np.uint64)
            keep = np.sort(idx[np.argsort(h, kind="stable")[:src.cap]])
            dropped[iso] += len(idx) - len(keep)
            idx = keep
        kept_lang[iso] += len(idx)
        if len(idx):
            tt = t.take(pa.array(idx))
            out_tables.append(tt)
            new_keys.append(tt.column("dkey").to_numpy())
    od = os.path.join(OUT, name)
    shutil.rmtree(od, ignore_errors=True)
    n_out = 0
    if out_tables:
        os.makedirs(od, exist_ok=True)
        allt = pa.concat_tables(out_tables)
        # stable order: by iso then id so shards are reproducible
        order = np.lexsort((np.array(allt.column("id").to_pylist(), dtype=object), np.array(allt.column("iso").to_pylist(), dtype=object)))
        allt = allt.take(pa.array(order))
        final = allt.drop(["dkey"])
        for k, s in enumerate(range(0, final.num_rows, SHARD)):
            pq.write_table(final.slice(s, SHARD).cast(SCHEMA_OUT), os.path.join(od, f"part-{k:04d}.parquet"), compression="zstd")
        n_out = final.num_rows
        save_seen(np.concatenate([seen] + new_keys))
    stats["kept"] = n_out
    stats["capped_off"] = sum(dropped.values())
    return dict(stats), dict(kept_lang), dict(dropped)


def progress(msg):
    os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
    with open(PROGRESS, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


# ------------------------------------------------------------------------------------------------------------------------ commands
def select(names):
    if names in (None, "", "all"):
        return list(SOURCES)
    out = []
    for n in names.split(","):
        if n not in SOURCES:
            raise SystemExit("unknown source " + n)
        out.append(n)
    return out


def cmd_test(a):
    ix = load_index()
    os.makedirs(os.path.join(OUT, "_work", "samples"), exist_ok=True)
    for name in select(a.sources):
        src = SOURCES[name]
        t0 = time.time()
        try:
            paths = src.files([p for p, _ in list_files(src.repo)])
            if not paths:
                print(f"### {name}: NO FILES MATCH", flush=True)
                continue
            path = paths[0] if not a.file else next((p for p in paths if a.file in p), paths[0])
            st, bl, samples = process_file(src, path, 0, limit=a.n, ix=ix, keep_samples=3)
        except Exception as e:
            print(f"### {name}: FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        print(f"### {name} ({src.repo}) file={path} n_files={len(paths)} {time.time() - t0:.1f}s", flush=True)
        print("   ", {k: v for k, v in sorted(st.items())}, "langs:", dict(collections.Counter(bl).most_common(8)), flush=True)
        for s in samples:
            m = json.loads(s["messages"])
            print("    [%s|%s] " % (s["iso"], s["task_type"]) + " || ".join(f"{x['role'][0]}: {x['content'][:160]!r}" for x in m[:4]), flush=True)
        with open(os.path.join(OUT, "_work", "samples", name + ".jsonl"), "w") as f:
            for s in samples:
                f.write(json.dumps({k: v for k, v in s.items() if k != "_texts"}, ensure_ascii=False, default=str) + "\n")


def cmd_run(a):
    names = select(a.sources)
    names = sorted(names, key=lambda n: (SOURCES[n].order, n))           # priority order = merge order (first source keeps cross-source duplicates)
    os.makedirs(STATE, exist_ok=True)
    ix = load_index()
    global _IX
    _IX = ix
    ctx = mp.get_context("fork")
    prog = collections.Counter()
    hb_stop = threading.Event()

    def heartbeat():
        while not hb_stop.wait(180):
            progress(f"stage1 running: {dict(prog)}")
    threading.Thread(target=heartbeat, daemon=True).start()
    todo, tasks, plan = [], [], {}
    for name in names:
        if os.path.exists(os.path.join(STATE, name + ".done")) and not a.force:
            progress(f"{name}: already done, skipped")
            continue
        src = SOURCES[name]
        try:
            allf = list_files(src.repo)
        except Exception as e:
            progress(f"{name}: LIST FAILED {str(e)[:150]}")
            open(os.path.join(STATE, name + ".failed"), "w").write(str(e))
            continue
        paths = src.files([p for p, _ in allf])
        if not paths:
            progress(f"{name}: no matching files")
            continue
        shutil.rmtree(os.path.join(CAND, name), ignore_errors=True)
        plan[name] = dict(files=len(paths), t0=time.time(), tot=collections.Counter(), errs=[], done=0)
        todo.append(name)
        tasks += [(name, p, i) for i, p in enumerate(paths)]
    prog["sources"] = len(todo); prog["files"] = len(tasks)
    progress(f"stage1 start: {len(todo)} sources, {len(tasks)} files, {a.workers} workers")
    # biggest sources first so that the pool never idles at the end on one huge file
    with ctx.Pool(min(a.workers, max(1, len(tasks)))) as pool:
        for nm, path, st, bl, err in pool.imap_unordered(_worker, tasks, chunksize=1):
            pl = plan[nm]
            pl["done"] += 1; prog["files_done"] += 1
            if err:
                pl["errs"].append((path, err)); prog["file_errors"] += 1
            else:
                pl["tot"].update(st); prog["raw"] += st.get("raw", 0); prog["cand"] += st.get("kept_candidates", 0)
            if pl["done"] == pl["files"]:
                progress(f"{nm}: stage1 done raw={pl['tot'].get('raw', 0)} cand={pl['tot'].get('kept_candidates', 0)} file_errors={len(pl['errs'])}")
    hb_stop.set()
    for name in todo:                                                      # stage 2: cross-source dedupe, cap, shards (sequential, priority order)
        src, pl = SOURCES[name], plan[name]
        mst, kept, capped = merge_source(name)
        shutil.rmtree(os.path.join(CAND, name), ignore_errors=True)
        rep = dict(source=name, repo=src.repo, seconds=round(time.time() - pl["t0"]), files=pl["files"], file_errors=pl["errs"][:10], n_file_errors=len(pl["errs"]), read=dict(pl["tot"]), merge=mst,
                   kept_by_lang=kept, capped_off=capped, licence=src.licence, licence_raw=src.licence_raw, synthetic=src.synthetic, note=src.note)
        json.dump(rep, open(os.path.join(STATE, name + ".report.json"), "w"), ensure_ascii=False, indent=1)
        if pl["errs"] and not mst.get("kept"):
            progress(f"{name}: FAILED all files ({len(pl['errs'])} errors) e.g. {pl['errs'][0][1][:140]}")
        else:
            open(os.path.join(STATE, name + ".done"), "w").write(time.strftime("%F %T"))
            progress(f"{name}: merged kept={mst.get('kept', 0)} langs={len(kept)} dup_within={mst.get('dup_within_source', 0)} dup_earlier={mst.get('dup_vs_earlier_sources', 0)} capped={mst.get('capped_off', 0)} file_errors={len(pl['errs'])}")


def _oldkeys_file(args):
    repo, path = args
    wdir = os.path.join(TMP, f"old_{os.getpid()}")
    os.makedirs(wdir, exist_ok=True)
    keys = []
    try:
        local = fetch(repo, path, wdir)
        pf = pq.ParquetFile(local)
        names = pf.schema_arrow.names
        cols = [c for c in ("instruction", "input", "response", "task") if c in names]
        for rb in pf.iter_batches(batch_size=20000, columns=cols):
            d = rb.to_pydict()
            n = len(d.get("instruction", []))
            for i in range(n):
                if d.get("task") and d["task"][i] == "translation":
                    continue
                ins, inp, resp = d["instruction"][i] or "", (d.get("input") or [""] * n)[i] or "", (d.get("response") or [""] * n)[i] or ""
                u = clean(ins) + (("\n\n" + clean(inp)) if clean(inp) else "")
                keys.append(xxhash.xxh64_intdigest((norm_key(u) + "\x01" + norm_key(clean(resp))).encode("utf-8", "surrogatepass")))
    except Exception as e:
        return path, np.empty(0, np.uint64), f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        shutil.rmtree(wdir, ignore_errors=True)
    return path, np.array(keys, np.uint64), None


def cmd_oldkeys(a):
    """hash set of the (user, assistant) text of the OLD build (<HF_ORG_LEGACY>/sft-all + <HF_ORG>/sft-new, non-translation rows): used by sources with vs_old=True (muri_it)"""
    jobs = []
    for repo in ((os.environ['HF_ORG_LEGACY'] + "/sft-all"), (os.environ['HF_ORG'] + "/sft-new")):
        for p, _ in list_files(repo):
            if p.endswith(".parquet") and "translation" not in p:
                jobs.append((repo, p))
    allk, errs = [], []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for path, k, err in pool.imap_unordered(_oldkeys_file, jobs):
            allk.append(k)
            if err:
                errs.append((path, err))
    keys = np.unique(np.concatenate(allk)) if allk else np.empty(0, np.uint64)
    np.save(os.path.join(STATE, "old_keys.npy"), keys)
    print("old keys:", len(keys), "files", len(jobs), "errors", errs[:3])
    progress(f"old_keys built: {len(keys)} keys from {len(jobs)} files, errors={len(errs)}")


def read_shards(name, cols=None):
    for fp in sorted(glob.glob(os.path.join(OUT, name, "part-*.parquet"))):
        yield fp, pq.read_table(fp, columns=cols)


def read_shards_dir(d, cols=None):
    for fp in sorted(glob.glob(os.path.join(d, "part-*.parquet"))):
        yield fp, pq.read_table(fp, columns=cols)


def cmd_tokens(a):
    """Gemma-3 token estimate: for every language a random 2,000-row sample over all sources (main + held-back synthetic rows; chat template of google/gemma-3-270m-it), mean tokens per row per (source, language),
    scaled by the row counts. Output rows are keyed (source, iso, held) so that the main path and the held-back synthetic rows are reported separately."""
    import random
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it", token=hf_token())
    dirs = []
    for n in SOURCES:
        if os.path.exists(os.path.join(STATE, n + ".done")):
            dirs.append((n, os.path.join(OUT, n), False))
            dirs.append((n, os.path.join(OUT, "_held_synthetic", n), True))
    rows = collections.Counter()                       # (src, iso, held) -> rows
    syn = collections.Counter()
    pools = collections.defaultdict(list)               # iso -> reservoir of (src, messages)
    seen = collections.Counter()
    rng = random.Random(0)
    K = a.sample
    for n, d, held in dirs:
        for fp, t in read_shards_dir(d, ["iso", "messages", "synthetic"]):
            isos, msgs, syns = t.column("iso").to_pylist(), t.column("messages").to_pylist(), t.column("synthetic").to_pylist()
            for i, m, sy in zip(isos, msgs, syns):
                rows[(n, i, held)] += 1
                if sy:
                    syn[(n, i, held)] += 1
                seen[i] += 1
                if len(pools[i]) < K:
                    pools[i].append((n, m))
                else:
                    j = rng.randrange(seen[i])
                    if j < K:
                        pools[i][j] = (n, m)
    stat = {}
    for iso, pool in pools.items():
        per = collections.defaultdict(list)
        for n, m in pool:
            txt = tok.apply_chat_template(json.loads(m), tokenize=False)
            per[n].append(len(tok(txt, add_special_tokens=False)["input_ids"]))
        allm = sum(sum(v) for v in per.values()) / max(1, sum(len(v) for v in per.values()))
        stat[iso] = (allm, {n: (sum(v) / len(v), len(v)) for n, v in per.items()})
    out = []
    for (n, iso, held), r in sorted(rows.items()):
        allm, per = stat[iso]
        m, k = per.get(n, (allm, 0))
        if k < 20:
            m = allm
        out.append(dict(source=n, iso=iso, held=int(held), rows=r, synthetic_rows=syn[(n, iso, held)], mean_tokens=round(m, 1), sampled=k, est_tokens=int(r * m)))
    json.dump(out, open(os.path.join(STATE, "tokens.json"), "w"), indent=1)
    with open(os.path.join(STATE, "tokens.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    tot = collections.Counter(); rr = collections.Counter()
    for o in out:
        if not o["held"]:
            tot[o["iso"]] += o["est_tokens"]; rr[o["iso"]] += o["rows"]
    print("iso rows est_tokens (main path)")
    for iso, t in tot.most_common():
        print(iso, rr[iso], t)
    progress(f"tokens estimated for {len(tot)} languages, main path {sum(tot.values())/1e6:.1f}M tokens")


def cmd_hold(a):
    """policy: synthetic rows of a language are held back (moved to _held_synthetic/<source>/) when that language has >= --threshold non-synthetic rows over all sources; reversible (files are only moved)"""
    nons = collections.Counter()
    srcs = [n for n in SOURCES if glob.glob(os.path.join(OUT, n, "part-*.parquet"))]
    for n in srcs:
        for fp, t in read_shards(n, ["iso", "synthetic"]):
            for i, sy in zip(t.column("iso").to_pylist(), t.column("synthetic").to_pylist()):
                if not sy:
                    nons[i] += 1
    covered = {i for i, c in nons.items() if c >= a.threshold}
    moved = collections.Counter()
    for n in srcs:
        hd = os.path.join(OUT, "_held_synthetic", n)
        for fp in sorted(glob.glob(os.path.join(OUT, n, "part-*.parquet"))):
            t = pq.read_table(fp)
            isos, syns = t.column("iso").to_pylist(), t.column("synthetic").to_pylist()
            hold = np.array([bool(s_) and i in covered for i, s_ in zip(isos, syns)])
            if not hold.any():
                continue
            os.makedirs(hd, exist_ok=True)
            held = t.filter(pa.array(hold))
            keep = t.filter(pa.array(~hold))
            hp = os.path.join(hd, os.path.basename(fp))
            pq.write_table(held, hp, compression="zstd")
            if keep.num_rows:
                pq.write_table(keep, fp, compression="zstd")
            else:
                os.remove(fp)
            for i in held.column("iso").to_pylist():
                moved[(n, i)] += 1
    json.dump({"threshold": a.threshold, "languages_with_nonsynthetic": {i: nons[i] for i in sorted(covered)}, "non_synthetic_rows": dict(nons),
               "held": {f"{n}:{i}": c for (n, i), c in sorted(moved.items())}}, open(os.path.join(STATE, "hold.json"), "w"), indent=1)
    print("held rows:", sum(moved.values()), "covered languages:", len(covered))
    progress(f"hold applied (threshold {a.threshold}): {sum(moved.values())} synthetic rows moved to _held_synthetic")


def cmd_merge(a):
    """stage 2 only: merge the candidate directories left in _cand (priority order); used after an interrupted run"""
    names = sorted(select(a.sources), key=lambda n: (SOURCES[n].order, n))
    for name in names:
        if os.path.exists(os.path.join(STATE, name + ".done")) and not a.force:
            continue
        if not os.path.isdir(os.path.join(CAND, name)):
            continue
        src = SOURCES[name]
        t0 = time.time()
        mst, kept, capped = merge_source(name)
        shutil.rmtree(os.path.join(CAND, name), ignore_errors=True)
        rep = dict(source=name, repo=src.repo, seconds=round(time.time() - t0), files=None, file_errors=[], n_file_errors=0, read={}, merge=mst, kept_by_lang=kept, capped_off=capped, licence=src.licence,
                   licence_raw=src.licence_raw, synthetic=src.synthetic, note=src.note, merge_only=True)
        json.dump(rep, open(os.path.join(STATE, name + ".report.json"), "w"), ensure_ascii=False, indent=1)
        open(os.path.join(STATE, name + ".done"), "w").write(time.strftime("%F %T"))
        progress(f"{name}: merged kept={mst.get('kept', 0)} langs={len(kept)} dup_within={mst.get('dup_within_source', 0)} dup_earlier={mst.get('dup_vs_earlier_sources', 0)} capped={mst.get('capped_off', 0)} (merge-only)")


def cmd_list(a):
    for n, s in SOURCES.items():
        print(f"{s.order:4d} {n:40s} {s.repo:60s} {s.licence:14s} syn={int(bool(s.synthetic))}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    p = sp.add_parser("list"); p.set_defaults(fn=cmd_list)
    p = sp.add_parser("test"); p.add_argument("--sources"); p.add_argument("--n", type=int, default=1000); p.add_argument("--file"); p.set_defaults(fn=cmd_test)
    p = sp.add_parser("oldkeys"); p.add_argument("--workers", type=int, default=8); p.set_defaults(fn=cmd_oldkeys)
    p = sp.add_parser("tokens"); p.add_argument("--sample", type=int, default=2000); p.set_defaults(fn=cmd_tokens)
    p = sp.add_parser("hold"); p.add_argument("--threshold", type=int, default=20000); p.set_defaults(fn=cmd_hold)
    p = sp.add_parser("merge"); p.add_argument("--sources"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_merge)
    p = sp.add_parser("run"); p.add_argument("--sources"); p.add_argument("--workers", type=int, default=20); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    if not getattr(a, "fn", None):
        ap.print_help(); return
    os.makedirs(TMP, exist_ok=True); os.makedirs(STATE, exist_ok=True)
    a.fn(a)


if __name__ == "__main__":
    main()
