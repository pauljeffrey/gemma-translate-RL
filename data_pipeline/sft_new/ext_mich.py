"""michsethowusu (owner 2026-10-11): the NLLB-v1 mined `<A>-<B>_sentence-pairs` sets (759) are NOT converted (NLLB-related, skipped for now). What is converted:

  mich_mt560      english-<lang>_sentence-pairs_mt560 (OPUS MT560) for the languages we carry: kept when the QE sample (audit_mich.py) says the set is good; both sides
                  GlotLID-audited; both directions; capped per language
  mich_emotions   <lang>-emotions-corpus     (label = j-hartmann emotion model on the English meaning; neutral down-sampled)
  mich_sentiments <lang>-sentiments-corpus   (binary sentiment from a classifier on the English meaning)
The labels of the last two are model-generated, not human; every row says so in its metadata.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from sft_new import hubio
from sft_new.compose import A, Rec, Skip
from sft_new.converters import register
from sft_new.ext_common import label_stream, pairs
from sft_new.ext_common import lang
from sft_new.textutil import clean

ACC = os.environ.get("SFT_ACCOUNTS", "/workspace/sft_build/accounts.json")
AUDIT = os.environ.get("SFT_AUDIT_MICH", "/workspace/sft_build/out/audit_mich.json")
MT560_CAP = 120_000
CLS_CAP = 40_000


def _repos(accounts) -> list[str]:
    return [d["id"] for d in json.loads(Path(accounts).read_text())["michsethowusu"]]


def mt560_repos(accounts=ACC):
    """(repo, our code, name of the non-English column) for the MT560 sets of languages we carry."""
    out = []
    for rid in _repos(accounts):
        n = rid.split("/", 1)[1]
        m = re.fullmatch(r"english-(.+)_sentence-pairs_mt560", n)
        if not m:
            continue
        code = lang(m.group(1).replace("-", " ")) or lang(m.group(1))
        if code and code != "eng":
            out.append((rid, code, None))
    # the non-English column of these sets is the ISO code of the language (eng, fon / orm / tsn ...): read from the first row at conversion time
    return out


def _col(row: dict) -> str | None:
    return next((k for k in row if k != "eng" and isinstance(row[k], str)), None)


@register("mich_mt560", "michsethowusu/english-<lang>_sentence-pairs_mt560 (OPUS MT560) for our languages, gated by the QE sample audit and GlotLID; capped per language")
def mich_mt560(rng, limit=None):
    audit = json.loads(Path(AUDIT).read_text()) if Path(AUDIT).exists() else {}
    for repo, code, _ in mt560_repos():
        if code not in A.VALID_LANGS:
            yield Skip(f"language_has_no_tag:{code}")
            continue
        a = audit.get(repo)
        if a is None:
            yield Skip(f"not_audited:{repo.split('/')[1][:30]}")
            continue
        if "mean" in a and not (a["mean"] >= 0.50 and a.get("share<0.46", 1.0) <= 0.30):
            yield Skip(f"qe_sample_below_bar:{code}")
            continue
        try:
            it = hubio.iter_repo(repo, limit=None)
            first = next(it)
        except Exception as e:
            yield Skip(f"repo_error:{code}:{type(e).__name__}")
            continue
        col = _col(first)
        import itertools
        yield from pairs(itertools.chain([first], it), "eng", col, "eng", code, repo, rng, limit=limit or MT560_CAP,
                         meta={"opus": "MT560", "qe_sample": a.get("mean") if "mean" in a else a.get("qe")})


def _cls_repos(kind: str):
    out = []
    for rid in _repos(ACC):
        n = rid.split("/", 1)[1]
        if kind in n and "corpus" in n:
            code = lang(n.split("-")[0])
            if code:
                out.append((rid, code))
    return out


def _balanced(rng, neutral_names=("neutral",), keep=0.35):
    state = {"n": 0, "neutral": 0}

    def ok(r):
        return True
    return ok


def _label_rows(repo: str, code: str, label_key: str, rng, limit, mapper, kind: str):
    it = hubio.iter_repo(repo)
    try:
        first = next(it)
    except StopIteration:
        return
    import itertools
    text_key = next((k for k in first if k not in (label_key, "__file", "__index_level_0__") and isinstance(first[k], str)), None)
    if not text_key:
        yield Skip("no_text_column")
        return
    cap, seen_neutral, seen = limit or CLS_CAP, 0, 0
    def gen():
        nonlocal seen_neutral, seen
        for r in itertools.chain([first], it):
            lab = mapper(clean(r.get(label_key)))
            if lab is None:
                continue
            if lab.lower() == "neutral":
                if rng.random() > 0.2:                                   # neutral is 60-80% of these corpora
                    continue
            yield r
    for rec in label_stream(gen(), text_key, lambda r: mapper(clean(r.get(label_key))), repo, rng, task=kind, instr_task=kind, plan=["<classify>"],
                            label_token="<answer>" if kind == "emotion" else "<sentiment>", declared=code, limit=cap, min_chars=8,
                            meta={"labels": "model-generated from the English meaning", "corpus": "michsethowusu"},
                            slots={"labels": "anger, disgust, fear, joy, sadness, surprise, neutral"} if kind == "emotion" else None):
        yield rec


@register("mich_emotions", "michsethowusu/<lang>-emotions-corpus (model-labelled from the English side; neutral down-sampled; capped) -> emotion classification")
def mich_emotions(rng, limit=None):
    for repo, code in _cls_repos("emotion"):
        if code not in A.VALID_LANGS:
            yield Skip(f"language_has_no_tag:{code}")
            continue
        yield from _label_rows(repo, code, "emotion", rng, limit, lambda s: s.lower() if s.lower() in ("joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral") else None, "emotion")


@register("mich_sentiments", "michsethowusu/<lang>-sentiments-corpus (binary, classifier labels on the English side; capped) -> sentiment")
def mich_sentiments(rng, limit=None):
    for repo, code in _cls_repos("sentiment"):
        if code not in A.VALID_LANGS:
            yield Skip(f"language_has_no_tag:{code}")
            continue
        yield from _label_rows(repo, code, "sentiment", rng, limit, lambda s: s.capitalize() if s.lower() in ("positive", "negative", "neutral") else None, "sentiment")


# ----------------------------------------------------------------------- CCMatrix (owner 2026-10-11: use it, African languages only)
CCM_REPO = "sentence-transformers/parallel-sentences-ccmatrix"
CCM_LANGS = {"af": "afr", "ha": "hau", "ig": "ibo", "mg": "mlg", "so": "som", "sw": "swh", "xh": "xho", "zu": "zul", "yo": "yor", "am": "amh", "rw": "kin", "sn": "sna",
             "st": "sot", "tn": "tsn", "wo": "wol", "ny": "nya", "om": "orm", "ti": "ti", "ln": "lin", "lg": "lug"}
CCM_AUDIT = os.environ.get("SFT_AUDIT_CCM", "/workspace/sft_build/out/audit_ccm.json")
CCM_CAP = 150_000


def ccm_folders() -> list:
    from sft_new.ext_common import files_of
    out = {}
    for p, _ in files_of(CCM_REPO, patterns=(".parquet",)):
        folder = p.split("/")[0]
        m = re.fullmatch(r"en-([a-z]{2,3})", folder)
        code = CCM_LANGS.get(m.group(1)) if m else None
        if code:
            out.setdefault((folder, code), []).append(p)
    return [(f, c, sorted(ps)) for (f, c), ps in sorted(out.items())]


@register("ccmatrix_african", "sentence-transformers/parallel-sentences-ccmatrix, en-<african language> folders only (the first rows of each, capped); QE sample gate; GlotLID on both sides")
def ccmatrix(rng, limit=None):
    audit = json.loads(Path(CCM_AUDIT).read_text()) if Path(CCM_AUDIT).exists() else {}
    for folder, code, paths in ccm_folders():
        if code not in A.VALID_LANGS:
            yield Skip(f"language_has_no_tag:{code}")
            continue
        a = audit.get(folder)
        if a is None:
            yield Skip(f"not_audited:{folder}")
            continue
        if "mean" in a and not (a["mean"] >= 0.50 and a.get("share<0.46", 1.0) <= 0.30):
            yield Skip(f"qe_sample_below_bar:{code}")
            continue
        from sft_new.ext_common import iter_files
        yield from pairs(iter_files(CCM_REPO, paths), "english", "non_english", "eng", code, CCM_REPO, rng, limit=limit or CCM_CAP,
                         meta={"mined": "CCMatrix", "qe_sample": a.get("mean", a.get("qe"))})


# ----------------------------------------------------------------------- owner 2026-10-11: Bible parallel (use), Akan POS (use, small cap), Twi 50M synthetic (QE-filtered indices)
BIBLE_REPO = "michsethowusu/african-bible-parallel"
BIBLE_CAP_EN = 30_000                    # verses per language paired with English
BIBLE_CAP_FR = 10_000                    # ... and with French (Louis Segond is the French text; the `en` column reads like a modern copyrighted English version: logged)


@register("mich_bible_parallel", "michsethowusu/african-bible-parallel (verse-aligned, one config per language; local text <-> English / French); capped per language, GlotLID-audited; licence not declared, used on owner's decision")
def mich_bible(rng, limit=None):
    from sft_new.ext_common import files_of, iter_files
    from africomet_translation.instructions import DISPLAY_NAMES
    by: dict = {}
    for p, _ in files_of(BIBLE_REPO, patterns=(".parquet",)):
        by.setdefault(p.split("/")[0], []).append(p)
    for cfg, paths in sorted(by.items()):
        code = lang(cfg)
        if code is None or code == "eng" or code not in A.VALID_LANGS or code not in DISPLAY_NAMES:
            yield Skip(f"language_has_no_tag:{cfg}")
            continue
        for col, cap, lg in (("en", BIBLE_CAP_EN, "eng"), ("fr", BIBLE_CAP_FR, "fra")):
            yield from pairs(iter_files(BIBLE_REPO, sorted(paths)), col, "local", lg, code, BIBLE_REPO, rng, limit=limit or cap, max_chars=2500,
                             meta={"genre": "scripture", "bible_column": col, "licence_note": "no licence declared on the repo; use approved by the owner 2026-10-11"})


POS_REPO = "michsethowusu/twi-fante-sentences-parts-of-speech-pos-10m"
POS_PER_PATTERN = 100
POS_CAP = 20_000


@register("mich_pos_akan", "michsethowusu/twi-fante-sentences-parts-of-speech-pos-10m (rule-generated Akan sentences with POS tags): small stratified sample (<=100 per sentence pattern, 20k total) -> word <tag> TAG pairs")
def mich_pos(rng, limit=None):
    from sft_new import templates as T
    from sft_new.ext_common import files_of
    cap = limit or POS_CAP
    got: dict = {}
    for p, _ in files_of(POS_REPO, patterns=(".parquet",)):
        for r in hubio.iter_file(POS_REPO, p, ["sentence", "pos_sequence", "word_count", "pattern_id"]):
            pid = r.get("pattern_id")
            if got.setdefault(pid, []).__len__() >= POS_PER_PATTERN:
                continue
            got[pid].append(r)
    rows = [r for v in got.values() for r in v]
    rng.shuffle(rows)
    n = 0
    for r in rows:
        words, tags = clean(r.get("sentence")).split(), clean(r.get("pos_sequence")).split()
        if not words or len(words) != len(tags) or len(words) > 20:
            yield Skip("word_tag_mismatch")
            continue
        resp = " ".join(f"{w} <tag> {t}" for w, t in zip(words, tags))
        ins = T.make("pos", rng, target="twi", source="twi", prefer="twi" if "twi" in T.languages("pos") else None)
        yield Rec("pos", POS_REPO, "twi", "twi", resp, input=" ".join(words), instruction=ins.text, position=ins.position, plan=["<POS>"], sequence_labels=True,
                  unsure_src=True, unsure_tgt=True,
                  meta={"synthetic": "rule-generated sentences", "tagset": "dataset-own (N, V, PN, ...)", "pattern_id": r.get("pattern_id"), "instruction_language": ins.language})
        n += 1
        if n >= cap:
            return


TWI50_DIR = Path(os.environ.get("SFT_TWI50", "/workspace/sft_build/out/twi50m_filter"))
TWI50_SRC = Path(os.environ.get("SFT_TWI50_SRC", "/workspace/sft_build/out/twi50m_src"))
TWI50_REPO = "michsethowusu/twi-english-parallel-synthetic-50m"
TWI50_CAP = 1_000_000


@register("mich_twi50m", "michsethowusu/twi-english-parallel-synthetic-50m: random 1M pairs of the AfriCOMET-QE >= 0.60 indices (twi50m_filter), GlotLID-audited; synthetic Twi + English paraphrase")
def mich_twi50m(rng, limit=None):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    keep_p = TWI50_DIR / "keep_qe_ge_0.60.npy"
    if not keep_p.exists():
        raise RuntimeError(f"{keep_p} is missing: the Twi 50M QE job has not finished")
    keep = np.load(keep_p)
    cap = limit or TWI50_CAP
    pick = np.sort(np.random.default_rng(20261011).choice(keep, size=min(cap, keep.size), replace=False))
    shards = json.loads((TWI50_DIR / "shards.json").read_text())["shards"]
    ends = np.array([s["offset"] + s["rows"] for s in shards])

    def rows():
        lo = 0
        for s in shards:
            hi = int(np.searchsorted(pick, s["offset"] + s["rows"]))
            if hi > lo:
                f = next(TWI50_SRC.rglob(s["file"]))
                t = pq.read_table(f, columns=["twi_sentence", "english_parahrase"]).take(pa.array(pick[lo:hi] - s["offset"]))
                yield from t.to_pylist()
            lo = hi
    yield from pairs(rows(), "twi_sentence", "english_parahrase", "twi", "eng", TWI50_REPO, rng, limit=None,
                     meta={"synthetic": "generated Twi + English paraphrase", "qe_threshold": 0.60, "qe": "masakhane/africomet-qe-stl-1.1"})
