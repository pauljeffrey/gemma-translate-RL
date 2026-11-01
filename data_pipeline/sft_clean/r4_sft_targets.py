#!/usr/bin/env python3
"""r4: deterministic sample of CLEAN SFT target texts per language (T1 text of the rebuilt corpus, agent D2), with a LICENCE FILTER (owner 2026-10-29: calibrations must be commercially usable).
Used for (a) the GlotLID-recall proxy of languages without a FLORES+ dev file, (b) the frequency-filtered widening of the diacritic inventory, (c) function-word / bigram statistics.

    # from local parquet shards (any directory with the D2 columns english, target, iso, [licence], [flipped], [key], [source_dataset])
    python r4_sft_targets.py --kept DIR [DIR ...] --out sft_targets.jsonl [--per-language 40000]
    # from the private Hub repo (streams one shard at a time into --cache, deletes it after use; needs HF_TOKEN)
    python r4_sft_targets.py --hub-repo <HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28 --hub-prefix data/ specialist/ --cache /tmp/sft_dl --out sft_targets.jsonl

Profiles (one pass over the shards fills all of them): --profile NAME:MODE:CAP:OUT with MODE
    all      every row, incl. `non_commercial` (IbomMT efi / ibb: owner exception 2026-10-29) and `unspecified` / `other`  -> function words / bigrams (statistics only) and efi / ibb text
    strict   only cc-by, cc0, apache, mit, mpl, odc-by, share_alike                                                          -> LID proxy and diacritic widening of all other languages
    no-nc    everything except `non_commercial`
Without --profile the old single-output interface applies (--out, --per-language, --strict-licence; `non_commercial` rows are then always dropped).
Keeps targets of 30-500 characters, English -> X rows only (flipped == False); per language the --per-language texts with the smallest sha1(iso|key) (deterministic); per file at most
--per-file evenly spaced rows of every language. Output rows: iso, text, source_dataset, licence. Idempotent: an existing --out is not rewritten. Stays on the machine.
"""
import argparse, glob, hashlib, heapq, json, os, sys
import pyarrow.parquet as pq

STRICT_OK = {"cc-by", "cc0", "apache", "mit", "mpl", "odc-by", "share_alike"}


def keep(mode, lic):
    if mode == "all":
        return True
    if mode == "strict":
        return lic in STRICT_OK
    return lic != "non_commercial"


def process(f, a, profiles, heaps):
    import numpy as np
    pf = pq.ParquetFile(f)
    names = pf.schema_arrow.names
    if "target" not in names or "english" not in names:
        return 0
    cols = [c for c in ("target", "iso", "key", "flipped", "source_dataset", "licence") if c in names]
    tb = pq.read_table(f, columns=cols)
    n = tb.num_rows
    isos = np.array(tb.column("iso").to_pylist()) if "iso" in names else np.zeros(n, dtype=object)
    idx_l = []
    for u in np.unique(isos):
        w = np.nonzero(isos == u)[0]
        idx_l.append(w[np.unique(np.linspace(0, len(w) - 1, min(len(w), a.per_file)).astype(np.int64))])
    d = tb.take(np.concatenate(idx_l)).to_pydict()
    kept = 0
    for k in range(len(d["target"])):
        if d.get("flipped") and str(d["flipped"][k]) in ("True", "true", "1"):
            continue
        lic = d["licence"][k] if "licence" in d else ""
        t = d["target"][k]
        if not t or not (30 <= len(t) <= 500):
            continue
        iso = d["iso"][k]
        h = int(hashlib.sha1(f"{iso}|{d['key'][k] if 'key' in d else t}".encode()).hexdigest()[:12], 16)
        item = (-h, t, d["source_dataset"][k] if "source_dataset" in d else "", lic)
        for name, (mode, cap, out) in profiles.items():
            if not keep(mode, lic):
                continue
            hp = heaps[name].setdefault(iso, [])
            if len(hp) < cap:
                heapq.heappush(hp, item)
            elif item > hp[0]:
                heapq.heapreplace(hp, item)
            kept += 1
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", nargs="*", default=[])
    ap.add_argument("--hub-repo", default=None); ap.add_argument("--hub-prefix", nargs="*", default=["data/", "specialist/"]); ap.add_argument("--cache", default="/tmp/sft_dl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-language", type=int, default=5000)
    ap.add_argument("--per-file", type=int, default=1500)
    ap.add_argument("--strict-licence", action="store_true")
    ap.add_argument("--profile", action="append", default=[], help="NAME:MODE:CAP:OUT (MODE all | strict | no-nc)")
    a = ap.parse_args()
    profiles = {}
    for p in a.profile:
        name, mode, cap, out = p.split(":", 3)
        assert mode in ("all", "strict", "no-nc")
        profiles[name] = (mode, int(cap), out)
    if not profiles:
        assert a.out, "--out or --profile required"
        profiles["default"] = ("strict" if a.strict_licence else "no-nc", a.per_language, a.out)
    todo = {n: v for n, v in profiles.items() if not (os.path.exists(v[2]) and os.path.getsize(v[2]) > 0)}
    for n in profiles:
        if n not in todo:
            print("exists, skipped:", profiles[n][2])
    if not todo:
        return
    heaps = {n: {} for n in todo}
    if a.hub_repo:
        from huggingface_hub import HfApi, hf_hub_download
        tok = os.environ["HF_TOKEN"]
        files = sorted(f for f in HfApi(token=tok).list_repo_files(a.hub_repo, repo_type="dataset") if f.endswith(".parquet") and any(f.startswith(p) for p in a.hub_prefix))
        os.makedirs(a.cache, exist_ok=True)
        for f in files:
            p = hf_hub_download(a.hub_repo, f, repo_type="dataset", token=tok, local_dir=a.cache)
            print(f, process(p, a, todo, heaps), flush=True)
            os.remove(p)
    else:
        for f in sorted(x for d in a.kept for x in glob.glob(os.path.join(d, "*.parquet"))):
            print(os.path.basename(f), process(f, a, todo, heaps), flush=True)
    for name, (mode, cap, out) in todo.items():
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for iso, hp in sorted(heaps[name].items()):
                for nh, t, s_, lic in sorted(hp, reverse=True):
                    fh.write(json.dumps({"iso": iso, "text": t, "source_dataset": s_, "licence": lic}, ensure_ascii=False) + "\n")
        os.replace(tmp, out)
        print(name, mode, {iso: len(hp) for iso, hp in sorted(heaps[name].items())})


if __name__ == "__main__":
    main()
