#!/usr/bin/env python3
"""Hidden non-commercial text (owner 2026-10-29: "feel free to filter out"): rows whose class says commercial/unverified but whose origin is non-commercial.
Nothing is dropped by `report`. `apply` writes the keys of the rows that are identifiable PER ROW; whole sources that cannot be split are only listed.

    python nc_hidden.py build-index --out nc_origin_index.npz [--hub-corpus 'repo|file_regex|col_en|col_tg|iso|label' ...]       # hashes of NC-origin corpora (MAFAND-MT all splits from GitHub + the Hub corpora given)
    python nc_hidden.py report --src data data_devhalf specialist --index nc_origin_index.npz --out nc_hidden_report --cpt chars_per_token.json [--nc-allow howard-nlp/ibom-mt] [--drop-source REPO ...]
    python nc_hidden.py apply  --src ... --index ... --out nc_hidden_keys.parquet [--nc-allow ...] [--allow-below-min]

Rules (default action in brackets):
  R1 licence text   licence_raw looks NC (nc / non-commercial / Tanzil / TED / cc-by-nc ...) but class != non_commercial            [DROP: identifiable per row]
  R2 NC-origin pair the normalised (english, target) pair of the row occurs, for the SAME language, in an NC-origin corpus of the index (MAFAND-MT train/dev/test, pristine-twi ...)  [DROP: identifiable per row]
  R3 whole source   sources whose terms are NC or unstated as a whole and that carry no per-row origin: UBC-NLP/AfroLingu-MT (gated: "non-commercial use ONLY", 43 datasets), michsethowusu *_mt560 (OPUS mixture with JW300,
                    TED, Tanzil: no per-row label), African-Languages-Lab/multi-open (licence other, terms unstated), KwekuAinoo/eng_twi (unspecified; overlap with an NC corpus measured by R2)  [LIST ONLY; --drop-source REPO drops one explicitly]
`report` writes, per language and per source, the pairs and (estimated) target tokens each rule would remove, the pairs that would remain, and flags every language that would fall below 2,000 pairs (min-pairs).
`apply` refuses to write a language's keys if the default rules would leave it below 2,000 pairs unless --allow-below-min (then the list of such languages is printed).
"""
import argparse, collections, csv, io, json, os, re, sys, unicodedata
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import xxhash
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import shard_files, key_of, is_nc, assert_clean_src, parse_nc_allow, nc_exception_tag

NON = re.compile(r"[\W_]+", re.U)
NC_WORDS = re.compile(r"(^|[^a-z])(nc|non-?commercial|noncommercial)([^a-z]|$)|tanzil|\bted\b|cc-by-nc|nc-sa|nc-nd", re.I)
WHOLE = {
    "UBC-NLP/AfroLingu-MT": "KEPT by owner decision (Amendment 6, windows tagged owner_exception=afrolingu-mt), listed only. Gated: users agree to use it for non-commercial use only (card); built from 43 datasets; rows carry only a language pair, no source",
    "African-Languages-Lab/multi-open": "licence 'other', terms not stated; no per-row source",
    "KwekuAinoo/eng_twi": "licence unspecified; may derive from an NC Twi corpus (see R2 overlap)",
}
WHOLE_RE = [(re.compile(r"michsethowusu/english-.*_mt560"), "OPUS MT560 mixture (JW300, TED, Tanzil, Bible ...): licence not established, no per-row source")]
ISO3 = {"swa": "swh"}


def nm(t):
    return NON.sub(" ", unicodedata.normalize("NFKC", t or "").casefold()).strip()


def h(iso, a, b):
    return xxhash.xxh64_intdigest(f"{iso}\x01{a}\x01{b}".encode("utf-8", "surrogatepass"))


def build_index(a):
    import urllib.request
    sets = collections.defaultdict(list)
    def get(u):
        return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "nc-hidden"}), timeout=60).read()
    if not a.no_mafand:
        tree = json.loads(get("https://api.github.com/repos/masakhane-io/lafand-mt/git/trees/main?recursive=1"))
        for f in [t["path"] for t in tree["tree"] if re.match(r"data/json_files/[^/]+/(train|dev|test)\.json$", t["path"])]:
            src, _, tgt = f.split("/")[2].partition("-")
            for line in get("https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/" + f).decode("utf-8", "replace").splitlines():
                if line.startswith("{"):
                    t = json.loads(line)["translation"]; x, y = nm(t.get(src, "")), nm(t.get(tgt, ""))
                    if x and y and src == "en":
                        sets["mafand-mt(CC-BY-NC)"].append(h(ISO3.get(tgt, tgt), x, y))
    for spec in a.hub_corpus or []:
        repo, regex, ce, ct, iso, label = spec.split("|")
        from huggingface_hub import HfApi, hf_hub_download
        tok = os.environ.get("HF_TOKEN"); fs = [s.rfilename for s in HfApi(token=tok).dataset_info(repo).siblings if re.search(regex, s.rfilename)]
        for fn in fs:
            p = hf_hub_download(repo, fn, repo_type="dataset", token=tok, local_dir=a.tmp)
            t = pq.read_table(p, columns=[ce, ct]).to_pydict()
            for e, g in zip(t[ce], t[ct]):
                x, y = nm(e), nm(g)
                if x and y: sets[label].append(h(iso, x, y))
            os.remove(p)
    np.savez(a.out, labels=np.array(list(sets)), **{f"s{i}": np.unique(np.array(v, np.uint64)) for i, v in enumerate(sets.values())})
    print({k: len(set(v)) for k, v in sets.items()}, "->", a.out)


def load_index(path):
    z = np.load(path, allow_pickle=False)
    return [(str(l), z[f"s{i}"]) for i, l in enumerate(z["labels"])]


def scan(a):
    assert_clean_src(a.src); allow = parse_nc_allow(a.nc_allow)
    idx = load_index(a.index) if a.index else []
    cpt = json.load(open(a.cpt)) if a.cpt else {}
    seen = set(); rows = []
    stats = collections.defaultdict(lambda: collections.Counter())     # (iso, source) -> counters
    drop_keys = {"R1": [], "R2": []}
    for f in shard_files(a.src):
        pf = pq.ParquetFile(f); cols = [c for c in ("english", "target", "iso", "key", "licence", "licence_raw", "source_dataset") if c in pf.schema_arrow.names]
        for rb in pf.iter_batches(batch_size=50000, columns=cols):
            t = rb.to_pydict(); n = len(t["iso"]); lic = t.get("licence") or [None] * n; raw = t.get("licence_raw") or [""] * n; sd = t.get("source_dataset") or [""] * n
            for i in range(n):
                if is_nc(lic[i]) and not nc_exception_tag(sd[i], allow):
                    continue                                    # already removed by the class rule
                k = t["key"][i] if "key" in t else key_of(t["english"][i], t["iso"][i], t["target"][i])
                if k in seen: continue
                seen.add(k); iso = t["iso"][i]; src = sd[i].split(":")[0]; c = stats[(iso, src)]
                c["pairs"] += 1; c["chars"] += len(t["target"][i])
                if raw[i] and NC_WORDS.search(raw[i]) and not is_nc(lic[i]):
                    c["R1_pairs"] += 1; c["R1_chars"] += len(t["target"][i]); drop_keys["R1"].append(k)
                elif idx:
                    hv = np.uint64(h(iso, nm(t["english"][i]), nm(t["target"][i])))
                    for label, arr in idx:
                        j = np.searchsorted(arr, hv)
                        if j < len(arr) and arr[j] == hv:
                            c["R2_pairs"] += 1; c["R2_chars"] += len(t["target"][i]); c["R2_label_" + label] += 1; drop_keys["R2"].append(k); break
    return stats, drop_keys, cpt, allow


def whole_reason(src):
    if src in WHOLE: return WHOLE[src]
    for rx, why in WHOLE_RE:
        if rx.fullmatch(src): return why
    return ""


def report(a):
    stats, drop_keys, cpt, allow = scan(a)
    os.makedirs(a.out, exist_ok=True)
    tok = lambda iso, ch: int(ch / cpt.get(iso, 3.4))
    lang_tot = collections.Counter(); lang_def = collections.Counter(); lang_src = collections.defaultdict(collections.Counter)
    for (iso, src), c in stats.items():
        lang_tot[iso] += c["pairs"]; lang_def[iso] += c["R1_pairs"] + c["R2_pairs"]; lang_src[iso][src] = c["pairs"]
    with open(os.path.join(a.out, "nc_hidden_by_language_source.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "source_repo", "pairs", "est_target_tokens", "R1_licence_text_pairs", "R2_nc_origin_pairs", "R2_tokens", "default_removed_pairs", "default_removed_tokens",
                                       "is_whole_source_candidate_R3", "R3_reason", "pairs_remaining_if_source_dropped", "below_min_if_source_dropped", "language_pairs_remaining_after_default_rules", "below_min_after_default"])
        for (iso, src), c in sorted(stats.items(), key=lambda kv: (-(kv[1]["R1_pairs"] + kv[1]["R2_pairs"]), -kv[1]["pairs"])):
            why = whole_reason(src); dr = c["R1_pairs"] + c["R2_pairs"]; rem_src = lang_tot[iso] - c["pairs"]
            if not (dr or why): continue
            w.writerow([iso, src, c["pairs"], tok(iso, c["chars"]), c["R1_pairs"], c["R2_pairs"], tok(iso, c["R2_chars"]), dr, tok(iso, c["R1_chars"] + c["R2_chars"]), int(bool(why)), why, rem_src if why else "", int(rem_src < a.min_pairs) if why else "",
                        lang_tot[iso] - lang_def[iso], int(lang_tot[iso] - lang_def[iso] < a.min_pairs)])
    below = [(i, lang_tot[i] - lang_def[i]) for i in lang_tot if lang_tot[i] - lang_def[i] < a.min_pairs]
    json.dump({"R1_pairs": len(drop_keys["R1"]), "R2_pairs": len(drop_keys["R2"]), "languages_below_min_after_default_rules": below, "dropped_sources_requested": a.drop_source}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"default rules would remove R1 {len(drop_keys['R1']):,} + R2 {len(drop_keys['R2']):,} pairs (identifiable per row); whole-source candidates are listed in nc_hidden_by_language_source.csv (not dropped); languages below {a.min_pairs} after the default rules: {below}")


def apply_(a):
    stats, drop_keys, cpt, allow = scan(a)
    keys = set(drop_keys["R1"]) | set(drop_keys["R2"])
    if a.drop_source:
        for f in shard_files(a.src):
            t = pq.read_table(f, columns=[c for c in ("english", "target", "iso", "key", "source_dataset") if c in pq.ParquetFile(f).schema_arrow.names]).to_pydict()
            for i, s in enumerate(t["source_dataset"]):
                if s.split(":")[0] in a.drop_source:
                    keys.add(t["key"][i] if "key" in t else key_of(t["english"][i], t["iso"][i], t["target"][i]))
    lang_tot = collections.Counter(); lang_def = collections.Counter()
    for (iso, src), c in stats.items():
        lang_tot[iso] += c["pairs"]; lang_def[iso] += c["R1_pairs"] + c["R2_pairs"]
    below = [(i, lang_tot[i] - lang_def[i]) for i in lang_tot if lang_tot[i] - lang_def[i] < a.min_pairs and lang_def[i] > 0]
    if below and not a.allow_below_min:
        raise SystemExit(f"refusing: these languages would end below {a.min_pairs} pairs because of the hidden-NC filter: {below}; tell the owner or pass --allow-below-min")
    pq.write_table(pa.table({"key": pa.array(sorted(keys), pa.uint64())}), a.out, compression="zstd")
    print(f"{len(keys):,} keys to drop -> {a.out}")


def merge_keep(a):
    """keep file of the gate minus the hidden-NC keys -> the keep file that windows.py reads"""
    keep = pq.read_table(a.keep)["key"].to_numpy(); drop = pq.read_table(a.drop)["key"].to_numpy()
    out = keep[~np.isin(keep, drop)]
    pq.write_table(pa.table({"key": pa.array(out, pa.uint64())}), a.out, compression="zstd")
    print(f"keep {len(keep):,} - hidden-NC {len(drop):,} listed ({len(keep) - len(out):,} actually removed) -> {len(out):,} -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge-keep"); m.add_argument("--keep", required=True); m.add_argument("--drop", required=True); m.add_argument("--out", required=True)
    b = sub.add_parser("build-index"); b.add_argument("--out", required=True); b.add_argument("--hub-corpus", nargs="*"); b.add_argument("--no-mafand", action="store_true"); b.add_argument("--tmp", default="/tmp/nc_origin_dl")
    for n in ("report", "apply"):
        s = sub.add_parser(n); s.add_argument("--src", nargs="+", required=True); s.add_argument("--index", default=None); s.add_argument("--cpt", default=None); s.add_argument("--nc-allow", nargs="*", default=[])
        s.add_argument("--drop-source", nargs="*", default=[]); s.add_argument("--min-pairs", type=int, default=2000)
        s.add_argument("--out", required=True)
        if n == "apply": s.add_argument("--allow-below-min", action="store_true")
    a = ap.parse_args(); {"build-index": build_index, "report": report, "apply": apply_, "merge-keep": merge_keep}[a.cmd](a)
