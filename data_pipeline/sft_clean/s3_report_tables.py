#!/usr/bin/env python3
"""Aggregate the per-file statistics of s3_prompts.py (and the AfriScience stats) into CSV tables for the S3 report (2026-10-28).

    python s3_report_tables.py --stats out/stats --out reports/data/clean_rl_science_2026-10-28 [--afs out_afs]
Every table has one row per key and the columns named in the header; nothing is estimated.
"""
import argparse
import collections
import csv
import glob
import json
import os

BINS = ("b64", "b256", "b512", "b1024", "b2048")


def w(path, rows, fields):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def pct(a, b):
    return round(100 * a / b, 3) if b else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--afs", default=None)
    ap.add_argument("--verify", default=None, help="s3_verify result json (full run)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.stats, "*__*.json")))
    S = [json.load(open(f)) for f in files if not os.path.basename(f).startswith("TEST_")]
    # ---- per pool / language / bin
    per_bin = []
    tot_bin = collections.defaultdict(lambda: [0, 0])
    for s in S:
        cb = collections.defaultdict(lambda: [0, 0])
        for k, v in s["cells"].items():
            b = k.split("|")[0]
            cb[b][0] += v[0]
            cb[b][1] += v[1]
        for b in BINS:
            if b not in cb:
                continue
            bef, aft = cb[b]
            if b == "b2048":
                per_bin.append(dict(pool=s["pool"], iso=s["iso"], bin=b, before=bef, after=0, removed_by_filter="", removed_pct="", note="bin dropped by design (not scanned)"))
                continue
            per_bin.append(dict(pool=s["pool"], iso=s["iso"], bin=b, before=bef, after=aft, removed_by_filter=bef - aft, removed_pct=pct(bef - aft, bef), note=""))
            tot_bin[(s["pool"], b)][0] += bef
            tot_bin[(s["pool"], b)][1] += aft
    w(os.path.join(a.out, "prompts_per_bin_language.csv"), per_bin, ["pool", "iso", "bin", "before", "after", "removed_by_filter", "removed_pct", "note"])
    tb = []
    for (pool, b), (bef, aft) in sorted(tot_bin.items()):
        tb.append(dict(pool=pool, bin=b, languages=len({s["iso"] for s in S if s["pool"] == pool}), before=bef, after=aft, removed=bef - aft, removed_pct=pct(bef - aft, bef)))
    for pool in sorted({s["pool"] for s in S}):
        bef = sum(v[0] for (p, b), v in tot_bin.items() if p == pool)
        aft = sum(v[1] for (p, b), v in tot_bin.items() if p == pool)
        b2048 = sum(s["rows_b2048_dropped"] for s in S if s["pool"] == pool)
        tb.append(dict(pool=pool, bin="ALL b64-b1024", languages=len({s["iso"] for s in S if s["pool"] == pool}), before=bef, after=aft, removed=bef - aft, removed_pct=pct(bef - aft, bef)))
        tb.append(dict(pool=pool, bin="b2048 (dropped)", languages=len({s["iso"] for s in S if s["pool"] == pool}), before=b2048, after=0, removed="", removed_pct=""))
    w(os.path.join(a.out, "prompts_per_bin_total.csv"), tb, ["pool", "bin", "languages", "before", "after", "removed", "removed_pct"])
    # ---- per language total
    pl = collections.defaultdict(lambda: collections.Counter())
    for s in S:
        c = pl[(s["pool"], s["iso"])]
        c.update(rows_in=s["rows_in"], b2048=s["rows_b2048_dropped"], scanned=s["rows_scanned"], kept=s["rows_kept"], removed=s["rows_removed"], exempt_only=s["rows_removed_exempt_only"],
                 kept_flag=s["kept_flag_ssa_mte_train"], removed_flag=s["removed_flag_ssa_mte_train"], unique=s["unique_english"])
    rows = [dict(pool=p, iso=i, rows_in=c["rows_in"], b2048_dropped=c["b2048"], scanned=c["scanned"], kept=c["kept"], removed=c["removed"], removed_pct=pct(c["removed"], c["scanned"]),
                 removed_only_by_flores_dev_train_half=c["exempt_only"], kept_flag_ssa_mte_train=c["kept_flag"], removed_flag_ssa_mte_train=c["removed_flag"]) for (p, i), c in sorted(pl.items())]
    w(os.path.join(a.out, "prompts_per_language.csv"), rows, list(rows[0].keys()))
    # ---- removed per protected set (rows; a row can hit several sets)
    ps = collections.defaultdict(collections.Counter)
    for s in S:
        for k, v in s["by_rule_set"].items():
            rule, nm = k.split("|", 1)
            ps[(s["pool"], nm)][rule] += v
        for k, v in s["by_set_any"].items():
            ps[(s["pool"], k)]["any"] += v
        for k, v in s["by_set_sole"].items():
            ps[(s["pool"], k)]["sole"] += v
        for k, v in s["exempt_only_sets"].items():
            ps[(s["pool"], k)]["only_train_half"] += v
    scanned = {p: sum(s["rows_scanned"] for s in S if s["pool"] == p) for p in {s["pool"] for s in S}}
    rows = []
    for (p, nm), c in sorted(ps.items(), key=lambda kv: (kv[0][0], -kv[1]["any"])):
        rows.append(dict(pool=p, protected_set=nm, rows_any_rule=c["any"], pct_of_scanned=pct(c["any"], scanned[p]), R1_exact=c["R1_exact"], R2_contiguous=c["R2_contiguous"], R3_8gram=c["R3_8gram"],
                         rows_with_this_set_only=c["sole"], rows_removed_only_because_of_flores_dev_train_half=c["only_train_half"]))
    w(os.path.join(a.out, "removed_by_protected_set.csv"), rows, ["pool", "protected_set", "rows_any_rule", "pct_of_scanned", "R1_exact", "R2_contiguous", "R3_8gram", "rows_with_this_set_only", "rows_removed_only_because_of_flores_dev_train_half"])
    # ---- by bin x set
    bs = collections.defaultdict(collections.Counter)
    for s in S:
        for k, v in s["by_set_bin"].items():
            b, nm = k.split("|", 1)
            bs[(s["pool"], nm)][b] += v
    rows = [dict(pool=p, protected_set=nm, **{b: c[b] for b in BINS[:4]}) for (p, nm), c in sorted(bs.items())]
    w(os.path.join(a.out, "removed_by_bin_and_set.csv"), rows, ["pool", "protected_set"] + list(BINS[:4]))
    # ---- by source domain
    dm = collections.defaultdict(lambda: [0, 0])
    for s in S:
        for k, v in s["cells"].items():
            b, src, dom, fg = k.split("|")
            if b == "b2048":
                continue
            kk = (s["pool"], src, dom, fg)
            dm[kk][0] += v[0]
            dm[kk][1] += v[1]
    rows = [dict(pool=p, source_dataset=src, domain=dom, field_group=fg, before=v[0], after=v[1], removed=v[0] - v[1], removed_pct=pct(v[0] - v[1], v[0])) for (p, src, dom, fg), v in sorted(dm.items())]
    w(os.path.join(a.out, "prompts_by_source_domain.csv"), rows, ["pool", "source_dataset", "domain", "field_group", "before", "after", "removed", "removed_pct"])
    # ---- source domain x set
    ds = collections.defaultdict(collections.Counter)
    for s in S:
        for k, v in s["by_dom_set"].items():
            src, dom, nm = k.split("|", 2)
            ds[(s["pool"], src, dom)][nm] += v
    rows = []
    for (p, src, dom), c in sorted(ds.items()):
        for nm, v in c.most_common():
            rows.append(dict(pool=p, source_dataset=src, domain=dom, protected_set=nm, rows=v))
    w(os.path.join(a.out, "removed_by_source_domain_and_set.csv"), rows, ["pool", "source_dataset", "domain", "protected_set", "rows"])
    # ---- full cell table (language x bin x source x domain x field group)
    rows = []
    for s in S:
        for k, v in sorted(s["cells"].items()):
            b, src, dom, fg = k.split("|")
            rows.append(dict(pool=s["pool"], iso=s["iso"], bin=b, source_dataset=src, domain=dom, field_group=fg, before=v[0], after=(v[1] if b != "b2048" else 0)))
    w(os.path.join(a.out, "prompts_cells_language_bin_source_domain.csv"), rows, ["pool", "iso", "bin", "source_dataset", "domain", "field_group", "before", "after"])
    # ---- input files (sha verified)
    rows = [dict(pool=s["pool"], iso=s["iso"], file=s["file"], bytes=s["input_bytes"], sha256=s.get("input_sha256") or "", sha256_matches_backup_MANIFEST=s.get("input_sha256_verified"), rows=s["rows_in"],
                 unique_english=s["unique_english"], index_version=s["index_version"], seconds=s["seconds"]) for s in S]
    w(os.path.join(a.out, "prompts_input_files.csv"), rows, ["pool", "iso", "file", "bytes", "sha256", "sha256_matches_backup_MANIFEST", "rows", "unique_english", "index_version", "seconds"])
    # ---- the two sets the lead asked about
    ov = []
    ver = json.load(open(a.verify)) if a.verify else None
    after_txt_ix = ("0 by the full pipeline pass over every kept window; %d of %d by the index re-run (this set is not in the independent python check)" % (ver["index_hits"], ver["index_checked"])) if ver else "NOT VERIFIED"
    after_txt = ("%d of %d unique windows tested by the independent python check, %d of %d by the index re-run" % (ver["py_hits"], ver["py_checked"], ver["index_hits"], ver["index_checked"])) if ver else "NOT VERIFIED"
    for p in sorted({s["pool"] for s in S}):
        for nm in ("afriscience_mt:dev", "afriscience_mt:test", "afrilongdoc:all", "ssa_mte:dev", "ssa_mte:test", "afrimte_wmt2024:all"):
            n_ = sum(s["by_set_any"].get(nm, 0) for s in S if s["pool"] == p)
            ov.append(dict(pool=p, protected_set=nm, prompts_removed_for_overlap=n_, prompts_overlapping_after_filter=(after_txt if nm.startswith(("afriscience", "afrilongdoc")) else after_txt_ix)))
        flagb = sum(s["kept_flag_ssa_mte_train"] for s in S if s["pool"] == p)
        flagr = sum(s["removed_flag_ssa_mte_train"] for s in S if s["pool"] == p)
        ov.append(dict(pool=p, protected_set="ssa_mte:train (FLAG ONLY, not removed)", prompts_removed_for_overlap="(removed by other sets: %d)" % flagr, prompts_overlapping_after_filter=str(flagb)))
    w(os.path.join(a.out, "overlap_afriscience_afrilongdoc_ssa_mte.csv"), ov, ["pool", "protected_set", "prompts_removed_for_overlap", "prompts_overlapping_after_filter"])
    print("tables written to", a.out, "from", len(S), "stat files")
    if a.afs:
        for n in ("afriscience_split_stats.csv", "afriscience_clean_train_stats.csv", "afriscience_protected_papers.csv"):
            src = os.path.join(a.afs, n)
            if os.path.exists(src):
                open(os.path.join(a.out, n), "w").write(open(src).read())


if __name__ == "__main__":
    main()
