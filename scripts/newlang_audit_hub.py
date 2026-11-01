#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hub / web audit of the new-language study (D-47): what text and parallel data exist for each candidate language. Read-only metadata queries, nothing downloaded except
README / small index files, nothing pushed.

    python scripts/newlang_audit_hub.py --out /workspace/newlang_probe/audit [--parts all]

Parts (each writes one json):
    card      AfriCOMET-QE-1.1 / STL-1.1 model-card language lists
    glotlid   GlotLID v3 labels per candidate (from the local model_v3.bin)
    hubsearch HfApi.list_datasets(filter=language:<iso>) and name searches (+ the michsethowusu mined repos) per candidate
    corpora   language lists of the big corpora: NLLB-Seed (oldi_seed), biblenlp, mteb/biblenlp, MADLAD-400, FineWeb-2, GlotCC, HPLT, tatoeba_mt, MT560, flores_plus / smol membership
    opus      language coverage of the OPUS-MT tc-bible models (Helsinki-NLP)
    wiki      Wikimedia: does a <code>.wikipedia.org exist, article count
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request

# language -> (names/search terms, iso tags for the `language:` filter, wikipedia codes)
CANDS = {
    "fuv": (["fulfulde", "fula", "fulah", "pular", "pulaar", "peul", "fulani", "adamawa fulfulde"], ["fuv", "ful", "ff", "fub", "fuh", "fuq", "fuf", "fue", "fui", "ffm", "fuc"], ["ff"]),
    "gaa": (["english-ga", "ga_", "ga-", "gaa", "gã", "ga language"], ["gaa"], []),
    "knc": (["kanuri"], ["knc", "kr", "kby", "krt"], ["kr"]),
    "tzm": (["tamazight", "tashelhit", "tachelhit", "central-atlas", "amazigh", "berber", "ber-", "zgh"], ["tzm", "zgh", "shi", "ber", "thv", "taq", "tmh", "ttq"], ["zgh", "tzm", "shi", "ber"]),
    "dik": (["dinka"], ["dik", "din", "dip", "diw", "dib", "dks"], ["din"]),
    "cjk": (["chokwe"], ["cjk"], []),
    "taq": (["tamasheq", "tamashek", "tuareg", "taq"], ["taq", "tmh", "ttq", "thv"], []),
    "nqo": (["nko", "n'ko", "nqo", "n_ko"], ["nqo"], ["nqo"]),
    "nde": (["ndebele", "north-ndebele", "nde"], ["nde", "nd"], ["nd"]),
    "fat": (["fante", "fanti"], ["fat"], []),
    "men": (["mende"], ["men"], []),
    "luy": (["luhya", "luyia", "luyha", "luyi"], ["luy", "lrg", "bxk", "nyd", "lko", "lkb", "lwg", "lri", "ida", "rag", "lts", "lsm", "ntk", "xog"], []),
    "mfe": (["mauritian", "morisyen", "kreol-morisien", "mfe"], ["mfe"], ["mfe"]),
    "dje": (["zarma", "djerma", "dje"], ["dje"], []),
    "sus": (["susu", "soso", "sus"], ["sus"], []),
    "bci": (["baoule", "baoulé", "bci"], ["bci"], []),
    "tig": (["tigre", "tig"], ["tig"], []),
    "alz": (["alur", "alz"], ["alz"], []),
    "dov": (["dombe", "dov", "tonga"], ["dov", "toi"], []),
    # group C
    "mnk": (["mandinka", "mandingo", "malinke", "mandinka-"], ["mnk", "man", "mwk", "emk", "mku", "msc"], ["bm"]),
    "snk": (["soninke", "sarakole"], ["snk"], []),
    "srr": (["serer", "sereer"], ["srr", "sav", "sine"], []),
    "mas": (["maasai", "masai", "maa"], ["mas", "kpe"], []),
    "kln": (["kalenjin", "kipsigis", "nandi"], ["kln", "sgc", "niq", "enb"], []),
    "mer": (["meru", "kimeru", "kîmîîrû"], ["mer"], []),
}

CARD_REPOS = ["masakhane/africomet-qe-stl-1.1", "masakhane/africomet-stl-1.1"]


def get_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "newlang-audit/1.0 (research; read-only)"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))


def part_card(out):
    from huggingface_hub import HfApi, hf_hub_download

    res = {}
    for r in CARD_REPOS:
        try:
            p = hf_hub_download(r, "README.md")
            txt = open(p, encoding="utf-8").read()
            m = re.search(r"^language:\s*\n((?:- .*\n)+)", txt, re.M)
            meta = [x[2:].strip() for x in m.group(1).strip().split("\n")] if m else []
            res[r] = {"language_metadata": meta, "n": len(meta), "readme_head": txt[:200], "readme": txt[:12000]}
        except Exception as e:                                          # noqa: BLE001
            res[r] = {"error": str(e)[:200]}
    json.dump(res, open(os.path.join(out, "card.json"), "w"), ensure_ascii=False, indent=1)
    return res


def part_glotlid(out):
    labs = json.load(open("/workspace/newlang_probe/data/glotlid_labels.json"))
    res = {}
    for k, (names, isos, _) in CANDS.items():
        found = [l for l in labs if l.split("_")[0] in set(isos) | {k}]
        res[k] = found
    json.dump(res, open(os.path.join(out, "glotlid.json"), "w"), indent=1)
    return res


def part_hubsearch(out):
    from huggingface_hub import HfApi

    api = HfApi()
    res = {}
    for k, (names, isos, _) in CANDS.items():
        seen = {}
        try:
            for iso in isos:
                for d in api.list_datasets(filter=f"language:{iso}", limit=300, full=True):
                    seen.setdefault(d.id, d)
            for n in names:
                for d in api.list_datasets(search=n, limit=200, full=True):
                    seen.setdefault(d.id, d)
        except Exception as e:                                          # noqa: BLE001
            res.setdefault(k, {})["error"] = str(e)[:200]
        rows = []
        for did, d in seen.items():
            tags = list(d.tags or [])
            rows.append({"id": did, "downloads": d.downloads, "likes": d.likes, "license": next((t[8:] for t in tags if t.startswith("license:")), None),
                         "tasks": [t[5:] for t in tags if t.startswith("task_categories:")][:4], "size": next((t[16:] for t in tags if t.startswith("size_categories:")), None),
                         "langs": [t[9:] for t in tags if t.startswith("language:")][:12], "n_langs": sum(1 for t in tags if t.startswith("language:")), "gated": d.gated, "private": d.private,
                         "lastmod": str(d.last_modified)[:10]})
        res.setdefault(k, {})["datasets"] = rows
        # NLLB-mined repos of michsethowusu
        mich = []
        try:
            for n in names:
                for d in api.list_datasets(author="michsethowusu", search=n, limit=100):
                    mich.append(d.id)
        except Exception:                                               # noqa: BLE001
            pass
        res[k]["michsethowusu"] = sorted(set(mich))
        print("hubsearch", k, len(rows), len(res[k]["michsethowusu"]), flush=True)
    json.dump(res, open(os.path.join(out, "hubsearch.json"), "w"), ensure_ascii=False, indent=1)
    return res


def list_files(api, repo, kind="dataset"):
    try:
        info = api.dataset_info(repo, files_metadata=True) if kind == "dataset" else api.model_info(repo, files_metadata=True)
        return [(s.rfilename, s.size) for s in info.siblings]
    except Exception as e:                                              # noqa: BLE001
        return str(e)[:200]


def part_corpora(out):
    from huggingface_hub import HfApi

    api = HfApi()
    res = {}
    allisos = sorted({i for v in CANDS.values() for i in v[1]} | set(CANDS))
    for repo in ["openlanguagedata/oldi_seed", "openlanguagedata/flores_plus", "google/smol", "nordpolemil/biblenlp-corpus", "bible-nlp/biblenlp-corpus", "mteb/biblenlp-corpus-mmteb",
                 "Helsinki-NLP/tatoeba_mt", "Helsinki-NLP/bible_para", "allenai/madlad-400", "HuggingFaceFW/fineweb-2", "cis-lmu/glotcc-v1", "HPLT/HPLT2.0_cleaned", "HPLT/HPLT2.0",
                 "facebook/flores", "allenai/nllb", "cis-lmu/Glot500", "HuggingFaceFW/finetranslations", "cis-lmu/GlotStoryBook", "Helsinki-NLP/opus-100", "Helsinki-NLP/tatoeba",
                 "wikimedia/wikipedia", "Davlan/sib200", "WueNLP/sib-fleurs", "google/fleurs", "mozilla-foundation/common_voice_17_0", "uonlp/CulturaX", "oscar-corpus/OSCAR-2301"]:
        files = list_files(api, repo)
        if isinstance(files, str):
            res[repo] = {"error": files}
            print("corpora", repo, "ERR", files[:80], flush=True)
            continue
        hits = {}
        for iso in allisos:
            m = [(f, s) for f, s in files if re.search(rf"(^|[/_\-.]){re.escape(iso)}([/_\-.]|$)", f)]
            if m:
                hits[iso] = m[:12]
        res[repo] = {"n_files": len(files), "hits": hits}
        print("corpora", repo, len(files), sorted(hits), flush=True)
    json.dump(res, open(os.path.join(out, "corpora.json"), "w"), ensure_ascii=False, indent=1)
    return res


def part_opus(out):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    res = {}
    models = [m.id for m in api.list_models(author="Helsinki-NLP", search="opus-mt-tc-bible", limit=500)]
    for mid in models:
        try:
            p = hf_hub_download(mid, "README.md")
            txt = open(p, encoding="utf-8").read()
            src = re.search(r"source language\(s\)\*?\*?:\s*(.*)", txt, re.I)
            tgt = re.search(r"target language\(s\)\*?\*?:\s*(.*)", txt, re.I)
            res[mid] = {"src": src.group(1).strip()[:3000] if src else None, "tgt": tgt.group(1).strip()[:3000] if tgt else None}
        except Exception as e:                                          # noqa: BLE001
            res[mid] = {"error": str(e)[:100]}
    json.dump(res, open(os.path.join(out, "opus.json"), "w"), ensure_ascii=False, indent=1)
    print("opus models", len(res), flush=True)
    return res


def part_wiki(out):
    res = {}
    try:
        sm = get_json("https://meta.wikimedia.org/w/api.php?action=sitematrix&format=json&smtype=language&smlangprop=code|name|site&smsiteprop=url|dbname|code|closed&smlimit=max", 60)["sitematrix"]
    except Exception as e:                                              # noqa: BLE001
        json.dump({"error": str(e)[:200]}, open(os.path.join(out, "wiki.json"), "w"))
        return {}
    langs = {v["code"]: v for k, v in sm.items() if isinstance(v, dict) and "code" in v}
    codes = sorted({c for v in CANDS.values() for c in v[2]})
    for c in codes:
        v = langs.get(c)
        if not v:
            res[c] = {"exists": False}
            continue
        wp = [s for s in v.get("site", []) if s.get("code") == "wiki"]
        d = {"exists": bool(wp), "name": v.get("name"), "closed": bool(wp and "closed" in wp[0])}
        if wp and not d["closed"]:
            try:
                st = get_json(wp[0]["url"] + "/w/api.php?action=query&meta=siteinfo&siprop=statistics&format=json")["query"]["statistics"]
                d["articles"], d["pages"] = st.get("articles"), st.get("pages")
            except Exception as e:                                      # noqa: BLE001
                d["error"] = str(e)[:100]
        res[c] = d
        print("wiki", c, d, flush=True)
    json.dump(res, open(os.path.join(out, "wiki.json"), "w"), ensure_ascii=False, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", nargs="+", default=["card", "glotlid", "hubsearch", "corpora", "opus", "wiki"])
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    fns = dict(card=part_card, glotlid=part_glotlid, hubsearch=part_hubsearch, corpora=part_corpora, opus=part_opus, wiki=part_wiki)
    for p in a.parts:
        t0 = time.time()
        fns[p](a.out)
        print("part", p, "done", round(time.time() - t0), "s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
