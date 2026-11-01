#!/usr/bin/env python3
"""Tables of reports/intake2_instr_2026-10-31.md (markdown) from /workspace/work/instr/_state: per-source table, per-language token table (main path + held-back synthetic), rejects. Narrative is hand-written."""
import collections, csv, glob, json, os, sys
STATE = "/workspace/work/instr/_state"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def human(n):
    return f"{n/1e6:.2f}M" if n >= 1e6 else (f"{n/1e3:.1f}k" if n >= 1e3 else str(n))


def main():
    import intake2_instr as M
    reps = [json.load(open(f)) for f in sorted(glob.glob(STATE + "/*.report.json"))]
    order = {n: s.order for n, s in M.SOURCES.items()}
    reps.sort(key=lambda r: (order.get(r["source"], 999), r["source"]))
    held = json.load(open(STATE + "/hold.json")) if os.path.exists(STATE + "/hold.json") else {}
    held_by_src = collections.Counter()
    for k, v in (held.get("held") or {}).items():
        held_by_src[k.split(":")[0]] += v
    import re
    raw_from_log = {}
    for ln in open("/workspace/work/logs/intake2_instr.progress"):
        m = re.search(r"(\S+): stage1 done raw=(\d+)", ln)
        if m:
            raw_from_log[m.group(1)] = int(m.group(2))
    L = ["| source | repo | licence class | synthetic | rows read | kept in total | held back (synthetic) | languages (rows kept) |", "|---|---|---|---|---:|---:|---:|---|"]
    for r in reps:
        langs = ", ".join(f"{k} {human(v)}" for k, v in sorted(r["kept_by_lang"].items(), key=lambda x: -x[1])[:14]) + (" ..." if len(r["kept_by_lang"]) > 14 else "")
        syn = "yes" if r["synthetic"] else ("mixed" if r["source"] == "muri_it" else "no")
        L.append(f"| {r['source']} | `{r['repo']}` | {r['licence']} | {syn} | {human(r['read'].get('raw', 0) if r['read'] else raw_from_log.get(r['source'], 0))} | {human(r['merge'].get('kept', 0))} | {human(held_by_src.get(r['source'], 0))} | {langs} |")
    open(STATE + "/report_sources.md", "w").write("\n".join(L) + "\n")
    rej = collections.Counter(); cand = 0
    for r in reps:
        for k, v in r["read"].items():
            if k.startswith("reject_"):
                rej[k] += v
        cand += r["merge"].get("candidates", 0)
    open(STATE + "/report_rejects.md", "w").write("\n".join(f"| {k[7:]} | {v} |" for k, v in rej.most_common()) + "\n")
    if os.path.exists(STATE + "/tokens.csv"):
        rows = list(csv.DictReader(open(STATE + "/tokens.csv")))
        lang = collections.defaultdict(collections.Counter)
        for r in rows:
            Lg = lang[r["iso"]]
            h = "h_" if r["held"] == "1" else ""
            Lg[h + "rows"] += int(r["rows"]); Lg[h + "tok"] += int(r["est_tokens"])
            if not h:
                Lg["syn_rows"] += int(r["synthetic_rows"]); Lg["syn_tok"] += int(int(r["est_tokens"]) * int(r["synthetic_rows"]) / max(1, int(r["rows"])))
        T = ["| iso | main-path rows | est. Gemma tokens | synthetic rows | synthetic tokens | held-back synthetic rows | held-back tokens | main-path sources (rows) |", "|---|---:|---:|---:|---:|---:|---:|---|"]
        for iso, Lg in sorted(lang.items(), key=lambda x: -(x[1]["tok"] + x[1]["h_tok"])):
            srcs = ", ".join(f"{r['source']} {human(int(r['rows']))}" for r in sorted((r for r in rows if r["iso"] == iso and r["held"] == "0"), key=lambda r: -int(r["rows"]))[:6])
            T.append(f"| {iso}{' (new)' if iso not in M.OUR else ''} | {human(Lg['rows'])} | {human(Lg['tok'])} | {human(Lg['syn_rows'])} | {human(Lg['syn_tok'])} | {human(Lg['h_rows'])} | {human(Lg['h_tok'])} | {srcs} |")
        s = lambda k: sum(Lg[k] for Lg in lang.values())
        T.append(f"| **total** | {human(s('rows'))} | {human(s('tok'))} | {human(s('syn_rows'))} | {human(s('syn_tok'))} | {human(s('h_rows'))} | {human(s('h_tok'))} | |")
        open(STATE + "/report_tokens.md", "w").write("\n".join(T) + "\n")
    print(len(reps), "sources; rejects:", rej.most_common(12))


if __name__ == "__main__":
    main()
