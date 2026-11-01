#!/usr/bin/env python3
"""tokswap step A (LAPTOP): download everything the box needs, with the READ token from the repo .env.

The token is read inside this script and never printed or written anywhere. Nothing is pushed to the Hub.
Outputs under --out (default: scratch dir), ready to scp to the box (/workspace/tokswap/in/):

  models/sft_final/   <HF_ORG>/gemma3-270m-translate @ step-27638 (SFT-final; private repo)
  models/rl2c/        <HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c, main:latest/ (RL-final)
  tok_orig/           tokenizer files of the SFT-final repo (identical vocab/merges to google/gemma-3-270m-it)
  flores/{dev,devtest}/<iso>_<Script>.jsonl   FLORES+ (all languages that exist)   [dev: choose N + identity test;
                      devtest: identity test + fertility MEASUREMENT only; never used for training merges]
  text/amh.txt tir.txt fon.txt   training-side text (one segment per line; the build uses *.wiki.txt / *.ft.txt / fon.mt560.txt / fon.fineweb2.txt):
        amh, tir : HuggingFaceFW/finetranslations (ODC-By) original-language chunks (the same source that feeds the
                   SFT mixture, commercial_ok class)  +  wikimedia/wikipedia 20231101 am / ti (CC-BY-SA)
        fon      : michsethowusu/english-fon_sentence-pairs_mt560 (cc-by-4.0) target side  +  wikimedia/wikipedia fon
  text/SOURCES.json  row/line/char counts per source

Usage:  .venv-clf/bin/python gemma_mt/scripts/tokswap_fetch_laptop.py --out <dir>
"""
import argparse, json, os, re, sys, unicodedata, hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def read_token():
    env = {}
    for l in open(REPO / ".env", encoding="utf-8"):
        l = l.strip()
        if l and not l.startswith("#") and "=" in l:
            k, v = l.split("=", 1)
            env[k] = v.strip().strip('"').strip("'")
    return env["HF_READ_TOKEN"]


def clean(s, nfc=True, fon_repair=False):
    """amh/tir: NFC (FLORES+ amh/tir dev is 100% NFC-stable).  fon: NOT normalised (FLORES+ fon mixes precomposed and
    decomposed tone marks), but the JW-style artefacts of the mt560 text are repaired: a space before a combining mark
    (o + space + U+0301) and before , . ; : ? ! is removed."""
    if nfc:
        s = unicodedata.normalize("NFC", s)
    s = re.sub("[ \t\u00a0]+", " ", s).strip()
    if fon_repair:
        s = re.sub(" ([\u0300-\u036f])", r"\1", s)
        s = re.sub(" ([,.;:?!])", r"\1", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--geez-chars-per-lang", type=int, default=60_000_000, help="cap on characters taken from FineTranslations per language")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--only", choices=["all", "fon"], default="all", help="fon: only rebuild the Fon text files")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = read_token()
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download, HfFileSystem
    api = HfApi(token=tok)
    log = {}

    if a.only == "fon":
        a.skip_models = True
    if not a.skip_models:
        p = snapshot_download((os.environ['HF_ORG'] + "/gemma3-270m-translate"), revision="step-27638", token=tok, local_dir=out / "models/sft_final")
        log["sft_final_commit"] = api.model_info((os.environ['HF_ORG'] + "/gemma3-270m-translate"), revision="step-27638").sha
        snapshot_download((os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"), revision="main", allow_patterns=["latest/*"], token=tok, local_dir=out / "models/rl2c")
        log["rl2c_commit"] = api.model_info((os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"), revision="main").sha
        (out / "tok_orig").mkdir(exist_ok=True)
        for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "config.json"]:
            src = out / "models/sft_final" / f
            if src.exists():
                (out / "tok_orig" / f).write_bytes(src.read_bytes())

    # ---- FLORES+ (gated; our token is authorised)
    files = [] if a.only == "fon" else [f for f in api.list_repo_files("openlanguagedata/flores_plus", repo_type="dataset") if f.endswith(".jsonl")]
    for f in files:
        dst = out / "flores" / f
        if dst.exists():
            continue
        hf_hub_download("openlanguagedata/flores_plus", f, repo_type="dataset", token=tok, local_dir=out / "flores_raw")
        dst.parent.mkdir(parents=True, exist_ok=True)
        (out / "flores_raw" / f).replace(dst)
    log["flores_files"] = len(files)

    # ---- training-side text
    import pyarrow.parquet as pq
    tdir = out / "text"; tdir.mkdir(exist_ok=True)
    fs = HfFileSystem(token=tok)
    srcs = {}

    def wiki(code, **kw):
        p = hf_hub_download("wikimedia/wikipedia", f"20231101.{code}/train-00000-of-00001.parquet", repo_type="dataset", token=tok, local_dir=out / "wiki_raw")
        t = pq.read_table(p, columns=["text"]).column("text").to_pylist()
        lines = []
        for doc in t:
            for ln in doc.split("\n"):
                ln = clean(ln, **kw)
                if len(ln) >= 20:
                    lines.append(ln)
        return lines

    def finetrans(cfg, files_, cap):
        lines, chars = [], 0
        for fn in files_:
            f = fs.open(f"datasets/HuggingFaceFW/finetranslations/data/{cfg}/{fn}")
            pf = pq.ParquetFile(f)
            # row groups are in file order: take every 3rd group so the sample is spread over the file
            for rg in range(0, pf.metadata.num_row_groups, 3):
                tb = pf.read_row_group(rg, columns=["og_chunks", "og_language", "og_language_score"]).to_pylist()
                for r in tb:
                    for ch in r["og_chunks"] or []:
                        for ln in ch.split("\n"):
                            ln = clean(ln)
                            if len(ln) >= 20:
                                lines.append(ln); chars += len(ln)
                if chars >= cap:
                    return lines
        return lines

    for lang, cfg, files_, wcode in [] if a.only == "fon" else [("amh", "amh_Ethi", ["000_00000.parquet", "001_00000.parquet"], "am"), ("tir", "tir_Ethi", ["000_00000.parquet"], "ti")]:
        w = wiki(wcode)
        ft = finetrans(cfg, files_, a.geez_chars_per_lang)
        srcs[lang] = {"wikipedia_20231101_" + wcode: {"lines": len(w), "chars": sum(map(len, w))}, "finetranslations_" + cfg + "_og_chunks": {"lines": len(ft), "chars": sum(map(len, ft))}}
        (tdir / f"{lang}.txt").write_text("\n".join(w + ft) + "\n", encoding="utf-8")
        (tdir / f"{lang}.wiki.txt").write_text("\n".join(w) + "\n", encoding="utf-8")
        (tdir / f"{lang}.ft.txt").write_text("\n".join(ft) + "\n", encoding="utf-8")

    w = wiki("fon", nfc=False, fon_repair=True)
    p = hf_hub_download("michsethowusu/english-fon_sentence-pairs_mt560", "data/train-00000-of-00001.parquet", repo_type="dataset", token=tok, local_dir=out / "mt560_raw")
    tb = pq.read_table(p)
    cols = tb.column_names
    log["mt560_fon_columns"] = cols
    tcol = [c for c in cols if "fon" in c.lower() or c.lower() in ("target", "translation")]
    log["mt560_fon_target_col"] = tcol
    rows = tb.to_pylist()
    mt = []
    for r in rows:
        v = None
        for c in tcol:
            if isinstance(r[c], str): v = r[c]
        if v is None:
            # fallback: the column that is not English
            for c in cols:
                if isinstance(r[c], str) and c.lower() not in ("english", "eng", "en", "source"):
                    v = r[c]
        if v:
            v = clean(v, nfc=False, fon_repair=True)
            if len(v) >= 10:
                mt.append(v)
    fw = []
    pfw = hf_hub_download("HuggingFaceFW/fineweb-2", "data/fon_Latn/train/000_00000.parquet", repo_type="dataset", token=tok, local_dir=out / "fw2_raw")
    for doc in pq.read_table(pfw, columns=["text"]).column("text").to_pylist():
        for ln in doc.split("\n"):
            ln = clean(ln, nfc=False, fon_repair=True)
            if len(ln) >= 20:
                fw.append(ln)
    (tdir / "fon.fineweb2.txt").write_text("\n".join(fw) + "\n", encoding="utf-8")
    srcs["fon_fineweb2"] = {"fineweb-2_fon_Latn_train_000_00000 (ODC-By 1.0)": {"lines": len(fw), "chars": sum(map(len, fw))}}
    srcs["fon"] = {"wikipedia_20231101_fon": {"lines": len(w), "chars": sum(map(len, w))}, "michsethowusu_english-fon_mt560_target": {"lines": len(mt), "chars": sum(map(len, mt))}}
    (tdir / "fon.txt").write_text("\n".join(w + mt) + "\n", encoding="utf-8")
    (tdir / "fon.wiki.txt").write_text("\n".join(w) + "\n", encoding="utf-8")
    (tdir / "fon.mt560.txt").write_text("\n".join(mt) + "\n", encoding="utf-8")
    log["text_sources"] = srcs
    (tdir / "SOURCES.json").write_text(json.dumps(log, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in log.items() if k != "text_sources"}, indent=1))
    print(json.dumps(srcs, indent=1))


if __name__ == "__main__":
    main()
