"""Slow, resumable crawler for African-language datasets on the Hugging Face Hub.

Stages (each caches to _raw/, so a re-run fetches nothing twice):
  A  list   : datasets tagged with each language code (API, full=true -> card data, tags, license)
  B  authors: every author seen in A is listed too; only datasets tagged with an African language (or whose repo name names one) are kept
  C  detail : per dataset README (raw), rows/columns/sample (datasets-server), once per dataset

Politeness: one request at a time, MIN_GAP seconds between requests (+ jitter), exponential back-off on 429/5xx,
Retry-After honoured. Authenticated with HF_TOKEN from the repo .env (read access only).

    python data-catalog/crawl_hf.py list|authors|detail [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "african_hf"
RAW = ROOT / "_raw"
LOG = ROOT / "_logs" / "crawl.log"
MIN_GAP = float(os.environ.get("CRAWL_GAP", "1.2"))

# codes the owner listed (+ the 639-1/639-3 twins HF uses for the same language) -> canonical (iso3, name)
LANG = {
    "amh": ("amh", "Amharic"), "am": ("amh", "Amharic"),
    "ti": ("tir", "Tigrinya"), "tir": ("tir", "Tigrinya"),
    "oke": ("oke", "Okpe"), "yo": ("yor", "Yoruba"), "yor": ("yor", "Yoruba"),
    "ibo": ("ibo", "Igbo"), "ig": ("ibo", "Igbo"),
    "ha": ("hau", "Hausa"), "hau": ("hau", "Hausa"),
    "pcm": ("pcm", "Nigerian Pidgin"), "gpe": ("gpe", "Ghanaian Pidgin"), "nag": ("nag", "Naga Pidgin (not African)"),
    "wes": ("wes", "Cameroon Pidgin"), "urh": ("urh", "Urhobo"), "efi": ("efi", "Efik"), "tiv": ("tiv", "Tiv"),
    "so": ("som", "Somali"), "som": ("som", "Somali"),
    "ff": ("ful", "Fulah (Fula/Fulfulde)"), "ful": ("ful", "Fulah (Fula/Fulfulde)"),
    "fuv": ("fuv", "Nigerian Fulfulde"), "fuh": ("fuh", "Western Niger Fulfulde"), "fub": ("fub", "Adamawa Fulfulde"),
    "fuq": ("fuq", "Central-Eastern Niger Fulfulde"), "fui": ("fui", "Bagirmi Fulfulde"),
    "ar": ("ara", "Arabic (general; DEFERRED)"),
    "zu": ("zul", "Zulu"), "zul": ("zul", "Zulu"),
    "rw": ("kin", "Kinyarwanda"), "kin": ("kin", "Kinyarwanda"),
    "ny": ("nya", "Chichewa"), "nya": ("nya", "Chichewa"),
    "wo": ("wol", "Wolof"), "wol": ("wol", "Wolof"), "wof": ("wof", "Gambian Wolof"),
    "ibb": ("ibb", "Ibibio"), "ijs": ("ijs", "Southeast Ijo"), "igl": ("igl", "Igala"), "ish": ("ish", "Esan"),
    "om": ("orm", "Oromo"), "orm": ("orm", "Oromo"), "ary": ("ary", "Moroccan Arabic (Darija)"),
    "sna": ("sna", "Shona"), "sn": ("sna", "Shona"), "fon": ("fon", "Fon"), "xho": ("xho", "Xhosa"), "xh": ("xho", "Xhosa"),
    "bem": ("bem", "Bemba"), "mfn": ("mfn", "Mbembe (Cross River)"),
    "st": ("sot", "Sesotho"), "sot": ("sot", "Sesotho"), "umb": ("umb", "Umbundu"),
    "ak": ("twi", "Twi (Akan macro-tag)"), "aka": ("twi", "Twi (Akan macro-tag)"), "tw": ("twi", "Twi"), "twi": ("twi", "Twi"),
    "lg": ("lug", "Luganda"), "lug": ("lug", "Luganda"),
    "nso": ("nso", "Northern Sotho (Sepedi)"), "arz": ("arz", "Egyptian Arabic"),
    "knc": ("knc", "Central Kanuri"), "kr": ("kau", "Kanuri"), "kau": ("kau", "Kanuri"), "kby": ("kby", "Manga Kanuri"),
    "tso": ("tso", "Tsonga"), "ts": ("tso", "Tsonga"), "nbl": ("nbl", "South Ndebele"), "nr": ("nbl", "South Ndebele"),
    "iso": ("iso", "Isoko"), "bm": ("bam", "Bambara"), "bam": ("bam", "Bambara"),
    "swa": ("swa", "Swahili (macro)"), "swh": ("swh", "Swahili"), "swc": ("swc", "Congo Swahili"), "sw": ("swh", "Swahili"),
    "ccl": ("ccl", "Cutchi-Swahili"), "ki": ("kik", "Kikuyu"), "kik": ("kik", "Kikuyu"),
    "kbp": ("kbp", "Kabiyè"), "ewe": ("ewe", "Ewe"), "ee": ("ewe", "Ewe"),
}
# further African languages the owner invited ("feel free to add any other"); 639-1 and 639-3 spelled alike where they differ
EXTRA = {
    "af": ("afr", "Afrikaans"), "afr": ("afr", "Afrikaans"), "tn": ("tsn", "Setswana"), "tsn": ("tsn", "Setswana"),
    "ln": ("lin", "Lingala"), "lin": ("lin", "Lingala"), "kg": ("kon", "Kongo"), "kon": ("kon", "Kongo"),
    "mg": ("mlg", "Malagasy"), "mlg": ("mlg", "Malagasy"), "rn": ("run", "Kirundi"), "run": ("run", "Kirundi"),
    "ss": ("ssw", "Swazi"), "ssw": ("ssw", "Swazi"), "ve": ("ven", "Venda"), "ven": ("ven", "Venda"),
    "sg": ("sag", "Sango"), "sag": ("sag", "Sango"), "lua": ("lua", "Luba-Kasai"), "lu": ("lub", "Luba-Katanga"),
    "kam": ("kam", "Kamba"), "luo": ("luo", "Luo"), "luy": ("luy", "Luyia"), "kln": ("kln", "Kalenjin"), "mer": ("mer", "Meru"),
    "ach": ("ach", "Acholi"), "nyn": ("nyn", "Nyankore"), "teo": ("teo", "Teso"), "myx": ("myx", "Masaaba"),
    "dag": ("dag", "Dagbani"), "dga": ("dga", "Dagaare"), "gaa": ("gaa", "Ga"), "gur": ("gur", "Frafra"), "fat": ("fat", "Fante"),
    "dyu": ("dyu", "Dyula"), "mos": ("mos", "Mossi"), "men": ("men", "Mende"), "tem": ("tem", "Temne"), "kri": ("kri", "Krio"),
    "bci": ("bci", "Baoulé"), "ktu": ("ktu", "Kituba"), "sus": ("sus", "Susu"), "ber": ("ber", "Berber"), "kab": ("kab", "Kabyle"),
    "tzm": ("tzm", "Central Atlas Tamazight"), "zgh": ("zgh", "Standard Moroccan Tamazight"), "shi": ("shi", "Tachelhit"),
    "aeb": ("aeb", "Tunisian Arabic"), "apd": ("apd", "Sudanese Arabic"), "acm": ("acm", "Mesopotamian Arabic (not African)"),
    "wal": ("wal", "Wolaytta"), "sid": ("sid", "Sidama"), "gaz": ("gaz", "West Central Oromo"), "hae": ("hae", "Eastern Oromo"),
    "tig": ("tig", "Tigre"), "ssy": ("ssy", "Saho"), "aar": ("aar", "Afar"), "aa": ("aar", "Afar"),
    "nus": ("nus", "Nuer"), "din": ("din", "Dinka"), "lgg": ("lgg", "Lugbara"), "nd": ("nde", "North Ndebele"), "nde": ("nde", "North Ndebele"),
    "toi": ("toi", "Tonga (Zambia)"), "loz": ("loz", "Lozi"), "lun": ("lun", "Lunda"), "kqn": ("kqn", "Kaonde"),
    "tum": ("tum", "Tumbuka"), "yao": ("yao", "Yao"), "seh": ("seh", "Sena"), "mgh": ("mgh", "Makhuwa-Meetto"), "vmw": ("vmw", "Makhuwa"),
    "ndc": ("ndc", "Ndau"), "hz": ("her", "Herero"), "her": ("her", "Herero"), "ng": ("ndo", "Ndonga"), "naq": ("naq", "Nama"),
    "kj": ("kua", "Kuanyama"), "kua": ("kua", "Kuanyama"), "tsc": ("tsc", "Tswa"), "xsm": ("xsm", "Kasem"), "bba": ("bba", "Baatonum"),
    "ann": ("ann", "Obolo"), "ogo": ("ogo", "Khana"), "ebu": ("ebu", "Embu"), "mas": ("mas", "Maasai"), "sbp": ("sbp", "Sangu"),
    "kcg": ("kcg", "Tyap"), "idu": ("idu", "Idoma"), "bin": ("bin", "Edo (Bini)"),
    "ukw": ("ukw", "Ukwuani"), "gbr": ("gbr", "Gbagyi"), "nup": ("nup", "Nupe"), 
    "ewo": ("ewo", "Ewondo"), "bas": ("bas", "Basaa"), "dua": ("dua", "Duala"),
    "knn": ("knn", "Konkani"),  # knn is NOT African; kept out below by NOT_AFRICAN
}
NOT_AFRICAN = {"nag", "acm", "knn", "ar", "ara"}   # ar: general Arabic is deferred by the owner
ALL = {**EXTRA, **LANG}
AFRICAN_CODES = {c for c, (iso, _) in ALL.items() if iso not in NOT_AFRICAN and c not in NOT_AFRICAN}

_last = 0.0
_names = sorted({n.split(" (")[0].split(" ")[0].lower() for iso, n in ALL.values() if iso not in NOT_AFRICAN and len(n.split(" ")[0]) >= 4}
                | {"naija", "twi", "akan", "kiswahili", "isizulu", "isixhosa", "sesotho", "setswana", "kinyarwanda", "chichewa", "afaan", "tigrigna", "tiv", "ewe", "fon"})
NAME_RE = re.compile(r"(?<![a-z])(" + "|".join(re.escape(n) for n in _names) + r")(?![a-z])", re.I)


def log(msg: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(time.strftime("%F %T ") + msg + "\n")


def token() -> str:
    t = os.environ.get("HF_TOKEN", "")
    if t:
        return t
    env = ROOT.parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def get(url: str, as_json=True, retries=6):
    global _last
    hdr = {"User-Agent": "sabiyarn-catalog/1.0 (slow research crawler)"}
    tk = token()
    if tk and "datasets-server" not in url:
        hdr["Authorization"] = f"Bearer {tk}"
    elif tk:
        hdr["Authorization"] = f"Bearer {tk}"
    wait = 5.0
    for attempt in range(retries):
        gap = MIN_GAP + random.random() * 0.6
        dt = time.time() - _last
        if dt < gap:
            time.sleep(gap - dt)
        _last = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as r:
                body = r.read()
                link = r.headers.get("Link", "")
                return (json.loads(body) if as_json else body.decode("utf-8", "replace")), link
        except urllib.error.HTTPError as e:
            if e.code in (404, 401, 403):
                return None, ""
            ra = e.headers.get("Retry-After")
            if e.code >= 500 and "datasets-server" in url and e.code != 429:
                # the viewer returns a persistent 500/501 for script-based, gated or broken repos: one retry, then give up on that call
                if attempt >= 1:
                    return None, ""
                time.sleep(3)
                continue
            if e.code == 429 or e.code >= 500:
                pause = float(ra) if ra and ra.isdigit() else wait
                log(f"{e.code} on {url[:110]} -> sleeping {pause:.0f}s")
                time.sleep(pause)
                wait = min(wait * 2, 300)
                continue
            return None, ""
        except Exception as e:  # network hiccup
            log(f"ERR {e!r} on {url[:110]}")
            time.sleep(wait)
            wait = min(wait * 2, 300)
    return None, ""


def list_datasets(params: dict):
    """Follow the API's cursor pagination (Link: rel=next)."""
    q = urllib.parse.urlencode({**params, "full": "true", "limit": 1000})
    url = f"https://huggingface.co/api/datasets?{q}"
    out = []
    while url:
        data, link = get(url)
        if data is None:
            break
        out.extend(data)
        m = re.search(r'<([^>]+)>;\s*rel="next"', link or "")
        url = m.group(1) if m else None
    return out


def keep(row: dict) -> dict:
    cd = row.get("cardData") or {}
    return {"id": row["id"], "author": row.get("author") or row["id"].split("/")[0], "downloads": row.get("downloads"), "likes": row.get("likes"),
            "gated": row.get("gated"), "private": row.get("private"), "createdAt": row.get("createdAt"), "lastModified": row.get("lastModified"),
            "tags": row.get("tags") or [], "license": cd.get("license"), "license_name": cd.get("license_name"),
            "language": cd.get("language"), "task_categories": cd.get("task_categories"), "task_ids": cd.get("task_ids"),
            "size_categories": cd.get("size_categories"), "multilinguality": cd.get("multilinguality"),
            "source_datasets": cd.get("source_datasets"), "pretty_name": cd.get("pretty_name"), "description": (row.get("description") or "")[:500]}


def stage_list():
    d = RAW / "A_lists"
    d.mkdir(parents=True, exist_ok=True)
    for code in sorted(ALL):
        f = d / f"{code}.json"
        if f.exists():
            continue
        rows = list_datasets({"filter": f"language:{code}"})
        f.write_text(json.dumps([keep(r) for r in rows]))
        log(f"list {code}: {len(rows)}")
        print(code, len(rows), flush=True)


def lang_tags(row: dict) -> set:
    t = set()
    for x in row.get("tags") or []:
        if x.startswith("language:"):
            t.add(x.split(":", 1)[1])
    for x in row.get("language") or []:
        t.add(str(x))
    return t


def stage_authors(limit=None):
    seen_authors = {}
    for f in (RAW / "A_lists").glob("*.json"):
        if f.stem in ("ar", "nag", "acm", "knn"):       # general Arabic is deferred; the others are not African
            continue
        for r in json.loads(f.read_text()):
            seen_authors[r["author"]] = seen_authors.get(r["author"], 0) + 1
    d = RAW / "B_authors"
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for a in sorted(seen_authors, key=lambda k: -seen_authors[k]):
        f = d / (a.replace("/", "_") + ".json")
        if f.exists():
            continue
        rows = list_datasets({"author": a})
        # keep only repos carrying an African language tag; huge generic accounts are therefore never copied wholesale
        sel = []
        for r in rows:
            k = keep(r)
            if lang_tags(k) & AFRICAN_CODES or lang_tags({"tags": r.get("tags")}) & AFRICAN_CODES:
                sel.append(k)
            elif NAME_RE.search(r["id"].split("/", 1)[-1]):          # untagged repo whose name names an African language
                k["name_match"] = True
                sel.append(k)
        f.write_text(json.dumps({"author": a, "total_in_account": len(rows), "african": sel}))
        log(f"author {a}: {len(sel)}/{len(rows)}")
        print(a, len(sel), "/", len(rows), flush=True)
        n += 1
        if limit and n >= limit:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["list", "authors"])
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    {"list": stage_list, "authors": lambda: stage_authors(a.limit)}[a.stage]()
