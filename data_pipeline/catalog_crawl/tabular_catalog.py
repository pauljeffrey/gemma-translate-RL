"""Table-analysis bucket: statistical / spreadsheet datasets (mostly English tables about African countries), kept apart from pretrain/sft/rl.

Output: african_hf/tabular/<ISO3>_<country>.csv (one file per country) + ALL_tables.csv. The language of these tables is English and the African
language tags some of them carried were an artefact of country names, so the grouping is by COUNTRY, not language.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import analyze as A
import crawl_hf as C

ROOT = C.ROOT
OUT = ROOT / "tabular"
GENERIC = {"tabular", "csv", "xlsx", "xls", "africa", "official-statistics", "open-data", "region:us", "json", "parquet", "text", "datasets", "pandas", "polars", "croissant"}
NUMERIC = re.compile(r"^(u?int\d*|float\d*|double|decimal.*|number)$", re.I)
TIME = re.compile(r"^(year|date|month|period|time|quarter|week|.*_year|.*_date|.*_period.*)$", re.I)
GEO = re.compile(r"^(country.*|iso3?|iso_?code|region|admin\d?.*|province|state|district|city|location|area|zone|continent|.*_iso3)$", re.I)
SOURCES = [("World Bank", r"world[- ]?bank|\bwdi\b|worldbank"), ("FEWS NET", r"fews"), ("HDX / OCHA", r"\bhdx\b|ocha|humanitarian"), ("WHO / health", r"\bwho\b|cholera|health|disease"),
           ("UNHCR / displacement", r"unhcr|idp|refugee|displace"), ("FAO / food security", r"fao|food|ipc|nutrition|agricultur"),
           ("climate", r"climate|rainfall|temperature|drought|flood"), ("population / census", r"population|census|demograph"), ("gender", r"gender"),
           ("conflict / security", r"conflict|acled|security|fatalit")]


def kind_of(names: list[str], types: dict) -> str:
    low = [n.lower() for n in names]
    if {"indicator_name", "year", "value"} <= set(low) or {"indicator", "year", "value"} <= set(low):
        return "long indicator time series (indicator, year, value)"
    if sum(bool(re.match(r"^(w\d+|week_?\d+|d_\d+|column_\d+|unnamed.*)$", n)) for n in low) >= 3:
        return "wide spreadsheet sheet (headers partly lost: w1..wN / column_N)"
    if any(TIME.match(n) for n in names) and sum(bool(NUMERIC.match(t)) for t in types.values()) >= 2:
        return "panel / time-indexed table"
    if sum(bool(NUMERIC.match(t)) for t in types.values()) >= 2:
        return "cross-section table"
    return "unclassified table"


def main():
    det_dir = C.RAW / "C_detail"
    recs = []
    for f in det_dir.glob("*.json"):
        d = json.loads(f.read_text())
        rid, meta = d["id"], d["meta"]
        is_es = rid.lower().startswith("electricsheepafrica/")
        tags = [t for t in (meta.get("tags") or [])]
        tabular_tag = "tabular" in tags or any(t in tags for t in ("format:csv", "modality:tabular")) or "tabular" in " ".join(meta.get("task_categories") or [])
        if not (is_es or tabular_tag):
            continue
        feats = A.feat_types(d.get("features"))
        if not feats:
            continue
        types = dict(feats)
        names = [n for n, _ in feats]
        n_num = sum(bool(NUMERIC.match(t)) for t in types.values())
        if not is_es and n_num < 2:
            continue
        sample = d.get("sample") or {}
        iso = (sample.get("country_iso3") or sample.get("iso3") or "")
        iso = iso.upper() if isinstance(iso, str) and re.match(r"^[A-Za-z]{3}$", iso) else ""
        country = sample.get("country_name") or sample.get("country") or ""
        if not iso:
            iso, country = "ZZZ", country or "(multi-country / unknown)"
        topic = [t for t in tags if ":" not in t and t.lower() not in GENERIC][:8]
        hay = (rid + " " + " ".join(tags) + " " + A.clean_readme(d.get("readme") or "")[:1500]).lower()
        src = [n for n, rx in SOURCES if re.search(rx, hay)]
        lic, lic_src, lic_cls = A.lic_norm(meta, d.get("readme") or "")
        time_cols = [n for n in names if TIME.match(n)]
        geo_cols = [n for n in names if GEO.match(n)]
        num_cols = [n for n, t in feats if NUMERIC.match(t)]
        txt_cols = [n for n, t in feats if t == "string" and n not in geo_cols and n not in time_cols and not re.search(r"(^|_)(id|iso|code|record)", n)]
        kind = kind_of(names, types)
        messy = sum(bool(re.match(r"^(d_\d+.*|column_\d+|unnamed.*|w\d+)$", n.lower())) for n in names) / max(len(names), 1)
        ind = next((n for n in names if n.lower() in ("indicator_name", "indicator")), None)
        use = []
        if "indicator" in kind:
            use.append(f"Table QA: look up `{'value'}` for a given `{ind}` and `year`; compare years, compute change/percent change and rank indicators; answer in the user's language after a calculator/pandas tool call.")
            use.append("Trend analysis and short forecast: fit/extrapolate `value` over `year` (regression) and explain the trend in prose (data-to-text).")
        elif "wide spreadsheet" in kind:
            use.append("Spreadsheet comprehension: infer the header from `source_sheet` and neighbouring columns before answering; good for 'ask the user / clean first' behaviour.")
        else:
            use.append(f"Table QA and aggregation over numeric columns {num_cols[:4]} grouped by {(geo_cols or time_cols or txt_cols or ['row'])[:2]}.")
        use.append("Data-to-text: describe the table or a filtered slice in an African language (use translation or generated descriptions); structured output: return selected rows as JSON; function calling: expose the table through a query_table(filter, aggregate) tool.")
        use.append("Needs synthetic question/answer generation (the repo ships tables only; no questions) and numeric verification, so answers are checkable (RL-verifiable).")
        recs.append(dict(dataset_id=rid, url="https://huggingface.co/datasets/" + rid, country_iso3=iso, country=country, topic_tags="; ".join(topic),
                         source_hints="; ".join(src), table_kind=kind, num_rows=d.get("num_rows"),
                         columns="; ".join(f"{n}:{t}" for n, t in feats), numeric_columns="; ".join(num_cols[:12]), time_columns="; ".join(time_cols),
                         geo_columns="; ".join(geo_cols), text_columns="; ".join(txt_cols[:8]),
                         table_language="en (numbers + English labels; not African-language text)", header_quality="messy" if messy > 0.3 else "ok",
                         tasks="table question-answering; numeric reasoning; trend analysis; data-to-text; structured output; function calling (query tool)",
                         usage_explanation="  |  ".join(use), license=lic, commercial_use=lic_cls, downloads=meta.get("downloads"),
                         last_modified=(meta.get("lastModified") or "")[:10], author=meta.get("author") or rid.split("/")[0],
                         sample_row=json.dumps({k: (str(v)[:60] if isinstance(v, str) else v) for k, v in list(sample.items())[:8]}, ensure_ascii=False, default=str)[:500]))
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.csv"):
        old.unlink()
    if not recs:
        print("no tables")
        return
    fields = list(recs[0])
    with (OUT / "ALL_tables.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(sorted(recs, key=lambda r: (r["country_iso3"], r["dataset_id"])))
    by = defaultdict(list)
    for r in recs:
        by[(r["country_iso3"], r["country"])].append(r)
    for (iso, c), rows in by.items():
        nm = re.sub(r"[^a-z0-9]+", "-", str(c).lower()).strip("-") or "unknown"
        with (OUT / f"{iso}_{nm}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(len(recs), "tables;", len(by), "country files;", Counter(r["table_kind"] for r in recs).most_common(), Counter(r["commercial_use"] for r in recs))


if __name__ == "__main__":
    main()
