#!/usr/bin/env python3
"""Commercial-use rule (Amendment 4, 2026-10-29): rows of licence class `non_commercial` must not reach the chain. Report, guard, and (optionally) write NC-free copies.

    python nc_filter.py report --src data data_devhalf specialist --out nc_report            # rows per language / source / licence_raw that are removed; flags suspicious NC-looking rows that carry another class
    python nc_filter.py guard-windows --windows windows_sft                                    # fails if any window carries licence non_commercial without an nc_exception tag or if windows.py kept NC rows (stats.json)
    python nc_filter.py write-clean --src specialist --out specialist_commercial               # optional: physical NC-free copy of a pair directory

The chain itself never needs the copies: `windows.py`, `qe_score.py --exclude-nc` and `gate.py --exclude-nc` drop NC rows inline (BEFORE the cross-directory dedupe, so an NC copy cannot mask a commercial copy).
Where NC rows can come from: `data/` = howard-nlp/ibom-mt (efi, ibb); `specialist/` = the R10 NC folder (masakhane/AfriDocMT, an OPUS slice (Tanzil/TED), AfriNLP/AfricanFineTranslations, polynews: amh);
`data_devhalf/` = FLORES+ dev (CC-BY-SA, class share_alike: checked below); `quarantine_nc/` is never an input (assert_clean_src).
"""
import argparse, collections, csv, json, os, re, sys
import pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import shard_files, assert_clean_src, is_nc, parse_nc_allow, nc_exception_tag, owner_tag

NC_WORDS = re.compile(r"(^|[^a-z])(nc|non-?commercial|noncommercial)([^a-z]|$)|tanzil|\bted\b|cc-by-nc|nc-sa|nc-nd", re.I)


def report(a):
    assert_clean_src(a.src); allow = parse_nc_allow(a.nc_allow)
    os.makedirs(a.out, exist_ok=True)
    nc = collections.Counter(); tot = collections.Counter(); susp = collections.Counter(); by_dir = collections.defaultdict(collections.Counter)
    ncchars = collections.Counter(); exc = collections.Counter(); own = collections.Counter()
    for f in shard_files(a.src):
        d = os.path.basename(os.path.dirname(f.rstrip("/")))
        pf = pq.ParquetFile(f)
        cols = [c for c in ("iso", "source_dataset", "licence", "licence_raw", "target") if c in pf.schema_arrow.names]
        for rb in pf.iter_batches(batch_size=100000, columns=cols):
            t = rb.to_pydict(); n = len(t["iso"])
            for i in range(n):
                lic = t["licence"][i]; iso = t["iso"][i]; src = t["source_dataset"][i].split(":")[0] if "source_dataset" in t else ""
                tot[(d, iso)] += 1
                ot = owner_tag(t["source_dataset"][i]) if "source_dataset" in t else ""
                if ot:
                    own[(d, iso, ot)] += 1
                if is_nc(lic) and nc_exception_tag(src, allow):
                    exc[(d, iso, src)] += 1; continue
                if is_nc(lic):
                    nc[(d, iso, src, (t.get("licence_raw") or [""] * n)[i][:60])] += 1; by_dir[d][iso] += 1; ncchars[iso] += len(t["target"][i]) if "target" in t else 0
                elif "licence_raw" in t and NC_WORDS.search(t["licence_raw"][i] or ""):
                    susp[(d, iso, src, lic, t["licence_raw"][i][:60])] += 1
    with open(os.path.join(a.out, "nc_rows_by_language_source.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dir", "iso", "source_repo", "licence_raw", "rows_removed"])
        for k, v in sorted(nc.items(), key=lambda kv: -kv[1]): w.writerow([*k, v])
    with open(os.path.join(a.out, "nc_rows_by_language.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "rows_removed_total", "target_chars_removed", "of_which_by_dir", "rows_in_inputs_of_this_language"])
        langs = sorted({k[1] for k in nc})
        for iso in sorted(langs, key=lambda i: -sum(by_dir[d][i] for d in by_dir)):
            w.writerow([iso, sum(by_dir[d][iso] for d in by_dir), ncchars[iso], json.dumps({d: by_dir[d][iso] for d in by_dir if by_dir[d][iso]}), sum(v for (d, i), v in tot.items() if i == iso)])
    with open(os.path.join(a.out, "suspicious_not_labelled_nc.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dir", "iso", "source_repo", "licence_class", "licence_raw", "rows"])
        for k, v in sorted(susp.items(), key=lambda kv: -kv[1]): w.writerow([*k, v])
    with open(os.path.join(a.out, "nc_kept_by_owner_exception.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dir", "iso", "source_repo", "rows_kept_nc_exception"]); [w.writerow([*k, v]) for k, v in sorted(exc.items(), key=lambda kv: -kv[1])]
    with open(os.path.join(a.out, "owner_exception_rows.csv"), "w", newline="") as f:         # disclosure list of Amendment 6 (AfroLingu-MT, NLLB-mined Fulah, Bible-derived, TWB terms, ...)
        w = csv.writer(f); w.writerow(["dir", "iso", "owner_exception_tag", "rows_in_inputs"]); [w.writerow([*k, v]) for k, v in sorted(own.items(), key=lambda kv: (kv[0][2], -kv[1]))]
    print(f"NC rows kept by owner exception {allow}: {sum(exc.values()):,}")
    print(f"NC rows to remove: {sum(nc.values()):,} in {len({k[1] for k in nc})} languages; by input dir: {{ {', '.join(f'{d}: {sum(c.values()):,}' for d, c in by_dir.items())} }}; rows with NC-looking licence text but another class: {sum(susp.values()):,}")
    if susp:
        print("CHECK suspicious_not_labelled_nc.csv: these rows are NOT removed by the class rule")


def guard_windows(a):
    st = json.load(open(os.path.join(a.windows, "stats.json")))
    bad = 0; tagged = collections.Counter(); n = 0
    for f in shard_files([a.windows]):
        t = pq.read_table(f, columns=["licence", "iso", "nc_exception"]).to_pydict(); n += len(t["licence"])
        for l, iso, tag in zip(t["licence"], t["iso"], t["nc_exception"]):
            if is_nc(l):
                if tag: tagged[(tag, iso)] += 1
                else: bad += 1
    print(f"windows with an owner NC exception: { {f'{k[0]}|{k[1]}': v for k, v in tagged.items()} }")
    print(f"{n:,} windows checked; untagged windows with licence non_commercial: {bad}; windows.py include_nc={st.get('include_nc')}; NC pairs dropped at windowing: {st.get('dropped_nc', 'n/a')}")
    if bad or st.get("include_nc"):
        raise SystemExit("NC rows reached the windows")
    print("GUARD OK: no non_commercial window")


def write_clean(a):
    assert_clean_src(a.src); os.makedirs(a.out, exist_ok=True); kept = removed = 0
    for f in shard_files(a.src):
        t = pq.read_table(f); m = [not is_nc(x) for x in t["licence"].to_pylist()]
        removed += m.count(False); kept += m.count(True)
        pq.write_table(t.filter(pa.array(m)), os.path.join(a.out, os.path.basename(f)), compression="zstd")
    print(f"kept {kept:,}, removed {removed:,} -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report"); r.add_argument("--src", nargs="+", required=True); r.add_argument("--out", required=True); r.add_argument("--nc-allow", nargs="*", default=[])
    g = sub.add_parser("guard-windows"); g.add_argument("--windows", required=True)
    w = sub.add_parser("write-clean"); w.add_argument("--src", nargs="+", required=True); w.add_argument("--out", required=True)
    a = ap.parse_args(); {"report": report, "guard-windows": guard_windows, "write-clean": write_clean}[a.cmd](a)
