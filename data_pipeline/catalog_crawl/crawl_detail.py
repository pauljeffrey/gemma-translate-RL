"""Stage C: README + splits/columns/sample + row counts for every candidate dataset (cached per dataset in _raw/C_detail/)."""
from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import crawl_hf as C

OUT = C.RAW / "C_detail"
DS = "https://datasets-server.huggingface.co"


def candidates() -> dict:
    cand = {}
    for f in sorted((C.RAW / "A_lists").glob("*.json")):
        code = f.stem
        if code in C.NOT_AFRICAN or code not in C.ALL:
            continue
        for r in json.loads(f.read_text()):
            if code == "ar":               # general Arabic is deferred: keep only if another African code is present
                if not (C.lang_tags(r) & (C.AFRICAN_CODES - {"ar"})):
                    continue
            cand.setdefault(r["id"], r)
    for f in sorted((C.RAW / "B_authors").glob("*.json")):
        for r in json.loads(f.read_text())["african"]:
            if r.get("name_match") or C.lang_tags(r) & C.AFRICAN_CODES:      # drops repos that were kept only because of an `ar` tag
                cand.setdefault(r["id"], r)
    return cand


def fetch_one(rid: str, meta: dict) -> dict:
    out = {"id": rid, "meta": meta}
    q = urllib.parse.quote(rid, safe="/")
    readme, _ = C.get(f"https://huggingface.co/datasets/{q}/raw/main/README.md", as_json=False)
    out["readme"] = (readme or "")[:60000]
    splits, _ = C.get(f"{DS}/splits?dataset={urllib.parse.quote(rid, safe='')}")
    out["splits"] = (splits or {}).get("splits") if isinstance(splits, dict) else None
    if out["splits"]:
        first = out["splits"][0]
        # pick the config/split pair whose config looks like a language code asked for, else the first one
        fr, _ = C.get(f"{DS}/first-rows?dataset={urllib.parse.quote(rid, safe='')}&config={urllib.parse.quote(first['config'], safe='')}&split={urllib.parse.quote(first['split'], safe='')}")
        if isinstance(fr, dict):
            out["features"] = fr.get("features")
            rows = fr.get("rows") or []
            out["sample"] = rows[0].get("row") if rows else None
        size, _ = C.get(f"{DS}/size?dataset={urllib.parse.quote(rid, safe='')}")
        if isinstance(size, dict):
            s = size.get("size") or {}
            out["num_rows"] = (s.get("dataset") or {}).get("num_rows")
            out["configs_rows"] = {c.get("config"): c.get("num_rows") for c in (s.get("configs") or [])}
            out["splits_rows"] = {f"{c.get('config')}/{c.get('split')}": c.get("num_rows") for c in (s.get("splits") or [])}
    return out


def main(limit=None, reverse=False):
    OUT.mkdir(parents=True, exist_ok=True)
    cand = candidates()
    print("candidates:", len(cand), flush=True)
    done = n = 0
    items = list(cand.items())
    if reverse:
        items.reverse()
    for rid, meta in items:
        f = OUT / (rid.replace("/", "__") + ".json")
        if f.exists():
            done += 1
            continue
        try:
            f.write_text(json.dumps(fetch_one(rid, meta)))
        except Exception as e:  # keep going; a failed repo is retried on the next run
            C.log(f"detail failed {rid}: {e!r}")
            continue
        n += 1
        if n % 25 == 0:
            print(f"fetched {n} (+{done} cached) of {len(cand)}", flush=True)
        if limit and n >= limit:
            break
    print("done", n, "new;", done, "cached", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(args[0]) if args else None, reverse="--reverse" in sys.argv)
