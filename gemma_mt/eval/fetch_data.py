# SPDX-License-Identifier: AGPL-3.0-or-later
"""Download the TEST sets of the three translation benchmarks into a small local directory (idempotent, no git, no `datasets`).

    python -m gemma_mt.eval.fetch_data --root /workspace/sft_build/bench --benchmarks flores_plus ntrex mafand

Layout written (exactly what `gemma_mt.eval.benchmarks.load_benchmark(name, path=...)` reads):

    <root>/flores_plus/devtest/<code>.jsonl     openlanguagedata/flores_plus (GATED: needs HF_TOKEN with access); only the codes of our
                                                languages + eng_Latn. The DEV split is never downloaded here (dev is for calibration only).
    <root>/ntrex/NTREX-128/newstest2019-{src.eng,ref.<code>}.txt     MicrosoftTranslator/NTREX (GitHub raw files)
    <root>/mafand/en-<code>/test.json           masakhane-io/lafand-mt data/json_files (JSON lines)

Opt-in extra benchmarks (D-42; never fetched unless named): `smol_sent`, `smol_doc`, `gatitos` = the three resources of the Hub dataset google/smol
(CC-BY-4.0, NOT gated, no token needed), English -> X files only, from the pinned revision `B.SMOL_REVISION`:

    <root>/smol_sent/en_<code>.jsonl            smolsent/en_<code>.jsonl
    <root>/smol_doc/en_<code>.jsonl             smoldoc/en_<code>.jsonl
    <root>/gatitos/en_<code>.jsonl              gatitos/en_<code>.jsonl
    <root>/<benchmark>/SOURCE.json              repo, revision, licence, citation (written once; keep it with the data: CC-BY needs attribution)

Four more small public sets for languages without any test set (also opt-in; `B.OTHER_BENCHMARKS`), each pinned to a revision:
    <root>/ibom_mt/en_<code>.jsonl              howard-nlp/ibom-mt eng-<code>/test.csv (CSV -> {"id","src","trg"})          ibb, efi
    <root>/flores_moz/devtest/{eng_Latn,seh_Latn_MZ}.jsonl   MOZNLP/FLORES_MOZ devtest (FLORES+ layout)                       seh
    <root>/glot_story/en_<code>.jsonl           cis-lmu/GlotStoryBook.csv, rows aligned to English by (Parallel ID, Text Number)   kqn lun loz kua her dag tsc nbl
    <root>/salt/en_<code>.jsonl                 Sunbird/salt text-all/test parquet (GATED: HF_TOKEN of an account that has access) nyn ach lug ibo swh

A file that exists and is not empty is kept. A language whose file does not exist upstream is reported as missing (never an error):
the evaluation then has no rows for it ("no data"), exactly like a language without a code in `languages.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence

from gemma_mt.eval import benchmarks as B
from gemma_mt.languages import ALL_ISOS

NTREX_RAW = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128/"
MAFAND_RAW = "https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/json_files/"


SMOL_RAW = f"https://huggingface.co/datasets/{B.SMOL_REPO}/resolve/{B.SMOL_REVISION}/"
SMOL_CITATION = ("Caswell et al. 2025, 'SMOL: Professionally translated parallel data for 115 under-represented languages', arXiv:2502.12301; "
                 "GATITOS: Jones et al. 2023, 'Bilex Rx: Lexical data augmentation for massively multilingual machine translation', arXiv:2303.15265")


def _have(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _http_get(url: str, dest: str, retries: int = 3, timeout: float = 120.0) -> bool:
    """Download `url` to `dest` (atomically). True on success, False on 404; other errors are retried then raised."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last: Optional[BaseException] = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            if not data:
                raise OSError("empty response")
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            last = e
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = e
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"download failed: {url}: {last}")


def fetch_ntrex_raw(root: str, isos: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """newstest2019 English source + the reference files of our languages. Returns {"fetched": [...], "missing": [...]} (NTREX codes)."""
    d = os.path.join(root, "NTREX-128")
    out: Dict[str, List[str]] = {"fetched": [], "missing": []}
    if not _have(os.path.join(d, "newstest2019-src.eng.txt")):
        if not _http_get(NTREX_RAW + "newstest2019-src.eng.txt", os.path.join(d, "newstest2019-src.eng.txt")):
            raise RuntimeError("NTREX English source file not found upstream")
    for iso in (isos or B.languages_with_data("ntrex")):
        code = B.code_for("ntrex", iso)
        if not code:
            continue
        name = f"newstest2019-ref.{code}.txt"
        if _have(os.path.join(d, name)) or _http_get(NTREX_RAW + name, os.path.join(d, name)):
            out["fetched"].append(code)
        else:
            out["missing"].append(code)
    return out


def fetch_mafand_raw(root: str, isos: Optional[Sequence[str]] = None, split: str = "test") -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"fetched": [], "missing": []}
    for iso in (isos or B.languages_with_data("mafand")):
        code = B.code_for("mafand", iso)
        if not code:
            continue
        dest = os.path.join(root, f"en-{code}", f"{split}.json")
        if _have(dest) or _http_get(f"{MAFAND_RAW}en-{code}/{split}.json", dest):
            out["fetched"].append(code)
        else:
            out["missing"].append(code)
    return out


def fetch_smol_raw(root: str, benchmark: str, isos: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """The English -> X files of `benchmark` (smol_sent / smol_doc / gatitos) for our languages. Returns {"fetched": [...], "missing": [...]} (SMOL codes)."""
    out: Dict[str, List[str]] = {"fetched": [], "missing": []}
    sub = B.SMOL_SUBDIR[benchmark]
    for iso in (isos or B.languages_with_data(benchmark)):
        code = B.code_for(benchmark, iso)
        if not code:
            continue
        dest = os.path.join(root, f"en_{code}.jsonl")
        if _have(dest) or _http_get(f"{SMOL_RAW}{sub}/en_{code}.jsonl", dest):
            out["fetched"].append(code)
        else:
            out["missing"].append(code)
    src = os.path.join(root, "SOURCE.json")
    if out["fetched"] and not os.path.exists(src):
        with open(src, "w", encoding="utf-8") as f:
            json.dump({"dataset": B.SMOL_REPO, "revision": B.SMOL_REVISION, "subset": sub, "licence": "cc-by-4.0",
                       "url": f"https://huggingface.co/datasets/{B.SMOL_REPO}", "citation": SMOL_CITATION}, f, indent=1)
    return out


def _write_pairs(dest: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, dest)


def _write_source(root: str, benchmark: str, repo: str, revision: str, note: str) -> None:
    src = os.path.join(root, "SOURCE.json")
    if not os.path.exists(src):
        with open(src, "w", encoding="utf-8") as f:
            json.dump({"dataset": repo, "revision": revision, "licence": B.LICENCES[benchmark], "url": f"https://huggingface.co/datasets/{repo}", "note": note}, f, indent=1)


def fetch_ibom_mt(root: str, isos: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """IbomMT `eng-<code>/test.csv` (columns id, eng, <code>) -> `<root>/en_<code>.jsonl`."""
    import csv
    import io

    out: Dict[str, List[str]] = {"fetched": [], "missing": []}
    for iso in (isos or B.languages_with_data("ibom_mt")):
        code = B.code_for("ibom_mt", iso)
        if not code:
            continue
        dest = os.path.join(root, f"en_{code}.jsonl")
        if not _have(dest):
            raw = os.path.join(root, "_raw", f"eng-{code}.test.csv")
            if not _http_get(f"https://huggingface.co/datasets/{B.IBOM_REPO}/resolve/{B.IBOM_REVISION}/eng-{code}/test.csv", raw):
                out["missing"].append(code)
                continue
            with open(raw, encoding="utf-8", newline="") as f:
                rows = [{"id": r["id"], "src": r["eng"], "trg": r[code]} for r in csv.DictReader(io.StringIO(f.read()))]
            _write_pairs(dest, rows)
            os.remove(raw)
        out["fetched"].append(code)
    if out["fetched"]:
        _write_source(root, "ibom_mt", B.IBOM_REPO, B.IBOM_REVISION, "test split = FLORES+ devtest English translated by paid linguists (Ibom NLP, arXiv 2511.06531); non-commercial licence")
    return out


def fetch_flores_moz(root: str, isos: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"fetched": [], "missing": []}
    base = f"https://huggingface.co/datasets/{B.FLORES_MOZ_REPO}/resolve/{B.FLORES_MOZ_REVISION}/devtest/"
    en = os.path.join(root, "devtest", "eng_Latn.jsonl")
    if not _have(en) and not _http_get(base + "eng_Latn.jsonl", en):
        raise RuntimeError("FLORES-MOZ English devtest file not found upstream")
    for iso in (isos or B.languages_with_data("flores_moz")):
        code = B.code_for("flores_moz", iso)
        if not code:
            continue
        dest = os.path.join(root, "devtest", f"{code}.jsonl")
        if _have(dest) or _http_get(base + f"{code}.jsonl", dest):
            out["fetched"].append(code)
        else:
            out["missing"].append(code)
    if out["fetched"]:
        _write_source(root, "flores_moz", B.FLORES_MOZ_REPO, B.FLORES_MOZ_REVISION, "FLORES-MOZ devtest; the card says: evaluation benchmark, do not train on it")
    return out


def fetch_glot_story(root: str, isos: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """GlotStoryBook.csv (26 MB, all languages) -> `<root>/en_<code>.jsonl` for our codes; a row is kept when the English text of the same story and
    sentence number exists. The id is `<Parallel ID>#<Text Number>` (+ `@<Language>` for a second translation of the same sentence, e.g. the dialect variants loz-na / loz-zm)."""
    import csv
    import io

    out: Dict[str, List[str]] = {"fetched": [], "missing": []}
    want = {B.code_for("glot_story", i): i for i in (isos or B.languages_with_data("glot_story")) if B.code_for("glot_story", i)}
    todo = [c for c in want if not _have(os.path.join(root, f"en_{c}.jsonl"))]
    if todo:
        raw = os.path.join(root, "_raw", "GlotStoryBook.csv")
        if not _have(raw) and not _http_get(f"https://huggingface.co/datasets/{B.GLOT_STORY_REPO}/resolve/{B.GLOT_STORY_REVISION}/GlotStoryBook.csv", raw, timeout=300.0):
            raise RuntimeError("GlotStoryBook.csv not found upstream")
        with open(raw, encoding="utf-8", newline="") as f:
            table = list(csv.DictReader(io.StringIO(f.read())))
        eng = {(r["Parallel ID"], r["Text Number"]): r["Text"] for r in table if r["ISO639-3"] == "eng"}
        by: Dict[str, List[Dict[str, str]]] = {c: [] for c in todo}
        used: set = set()
        for r in table:
            c = r["ISO639-3"]
            k = (r["Parallel ID"], r["Text Number"])
            if c in by and k in eng and r["Text"].strip():
                rid = f"{k[0]}#{k[1]}"
                if (c, rid) in used:                       # a second translation of the same sentence (dialect variant, e.g. loz-na / loz-zm): keep it, with its own id
                    rid = f"{rid}@{r['Language']}"
                    while (c, rid) in used:
                        rid += "'"
                used.add((c, rid))
                by[c].append({"id": rid, "src": eng[k], "trg": r["Text"], "license": r["License"]})
        for c, rows in by.items():
            if rows:
                _write_pairs(os.path.join(root, f"en_{c}.jsonl"), rows)
        os.remove(raw)
    for c in want:
        (out["fetched"] if _have(os.path.join(root, f"en_{c}.jsonl")) else out["missing"]).append(c)
    if out["fetched"]:
        _write_source(root, "glot_story", B.GLOT_STORY_REPO, B.GLOT_STORY_REVISION, "African Storybook Project stories; the per-row `license` is kept in the files (CC-BY, CC-BY-NC, CC BY-NC-SA)")
    return out


def fetch_salt(root: str, isos: Optional[Sequence[str]] = None, token: Optional[str] = None) -> Dict[str, List[str]]:
    """Sunbird/salt `text-all/test` parquet (gated) -> `<root>/en_<code>.jsonl` (src = eng_source_text, trg = <code>_text)."""
    out: Dict[str, List[str]] = {"fetched": [], "missing": []}
    want = {B.code_for("salt", i): i for i in (isos or B.languages_with_data("salt")) if B.code_for("salt", i)}
    if [c for c in want if not _have(os.path.join(root, f"en_{c}.jsonl"))]:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        p = hf_hub_download(B.SALT_REPO, "text-all/test-00000-of-00001.parquet", repo_type="dataset", revision=B.SALT_REVISION, token=token,
                            local_dir=os.path.join(root, "_raw"))
        t = pq.read_table(p).to_pydict()
        for c in want:
            if f"{c}_text" not in t:
                continue
            _write_pairs(os.path.join(root, f"en_{c}.jsonl"), [{"id": str(i), "src": e, "trg": x} for i, e, x in zip(t["id"], t["eng_source_text"], t[f"{c}_text"]) if e and x])
        import shutil

        shutil.rmtree(os.path.join(root, "_raw"), ignore_errors=True)
    for c in want:
        (out["fetched"] if _have(os.path.join(root, f"en_{c}.jsonl")) else out["missing"]).append(c)
    if out["fetched"]:
        _write_source(root, "salt", B.SALT_REPO, B.SALT_REVISION, "text-all test split (500 sentences); gated dataset, accepted terms of the account whose token is used")
    return out


def fetch_flores_plus_files(root: str, isos: Optional[Sequence[str]] = None, split: str = "devtest",
                            token: Optional[str] = None) -> Dict[str, List[str]]:
    """FLORES+ (gated) files `<split>/<code>.jsonl` through huggingface_hub; only the codes we need."""
    if split != "devtest":
        raise ValueError("fetch_data downloads FLORES+ devtest only (dev is calibration data, see DECISIONS.md)")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    codes = ["eng_Latn"] + [c for c in (B.code_for("flores_plus", i) for i in (isos or B.languages_with_data("flores_plus"))) if c]
    out: Dict[str, List[str]] = {"fetched": [], "missing": []}
    for code in codes:
        dest = os.path.join(root, split, f"{code}.jsonl")
        if _have(dest):
            out["fetched"].append(code)
            continue
        try:
            hf_hub_download(B.FLORES_REPO, f"{split}/{code}.jsonl", repo_type="dataset", token=token, local_dir=root)
            out["fetched"].append(code)
        except EntryNotFoundError:
            out["missing"].append(code)
    if "eng_Latn" in out["missing"]:
        raise RuntimeError("FLORES+ English devtest file not found (token without access to the gated dataset?)")
    return out


def ensure_benchmarks(root: str, names: Sequence[str] = B.BENCHMARKS, isos: Optional[Sequence[str]] = None,
                      token: Optional[str] = None) -> Dict[str, Dict]:
    """Download what is missing; returns {benchmark: {"path": dir for load_benchmark, "fetched": [...], "missing": [...]}}."""
    info: Dict[str, Dict] = {}
    for name in names:
        sub = os.path.join(root, name)
        if name == "flores_plus":
            r = fetch_flores_plus_files(sub, isos, "devtest", token)
        elif name == "ntrex":
            r = fetch_ntrex_raw(sub, isos)
        elif name == "mafand":
            r = fetch_mafand_raw(sub, isos, "test")
        elif name in B.SMOL_BENCHMARKS:
            r = fetch_smol_raw(sub, name, isos)
        elif name == "ibom_mt":
            r = fetch_ibom_mt(sub, isos)
        elif name == "flores_moz":
            r = fetch_flores_moz(sub, isos)
        elif name == "glot_story":
            r = fetch_glot_story(sub, isos)
        elif name == "salt":
            r = fetch_salt(sub, isos, token)
        else:
            raise ValueError(f"unknown benchmark {name!r}")
        info[name] = {"path": sub, **r}
    return info


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--benchmarks", nargs="*", default=list(B.BENCHMARKS), help="names or comma-separated names; also " + ", ".join(B.EXTRA_BENCHMARKS))
    ap.add_argument("--languages", nargs="*", default=None)
    a = ap.parse_args(argv)
    info = ensure_benchmarks(a.root, B.parse_benchmark_names(a.benchmarks), a.languages)
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
