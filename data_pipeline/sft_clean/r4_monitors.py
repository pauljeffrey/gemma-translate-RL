#!/usr/bin/env python3
"""r4: ids of the dev monitors on H_sel v2 (role S) and the draw-weight file of the clean run. Ids / language lists / weights / hashes only.

general dev : 40 ids = the first 40 of H_sel ordered by sha256("gen|<id>"), the same ids for every language (paired bootstrap), languages = the old general-dev set that has a FLORES+ dev file
              (afr ewe fon hau ibo kin nya orm sna som sot swh tsn wol xho yor zul) + twi; pcm has no clean dev reference (MAFAND dev is not an S source): see the report.
weak dev    : 60 ids = the first 60 of H_sel MINUS the 100 forced ids, ordered by sha256("weak|<id>"). The weak list was DEFINED on the 100 forced ids (reports/data/weak_selection_2026-10-26),
              so a monitor of improvement on those languages uses the other ids of H_sel (no selection circularity); languages = weak list (25) + G + W with a FLORES+ dev file.
draw weights: x2.5 on the dev-defined weak list (25, hash 4472d711...) union G + W (prereg section 7, lean schedule E.3 'do not raise beyond 2.5'); every other language 1.0. The old extra tiers
              (ish 4.0, bin/bbj/oke 3.5, kon/tsc/dag 3.0, hau/lug/nso/zul 2.0 ...) rested on devtest or unknown selection (audit 2.3 c, e) and are not carried over.
"""
import hashlib, json, sys


def hk(tag, i):
    return int(hashlib.sha256(f"{tag}|{i}".encode()).hexdigest(), 16)


def sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def main(split_path, devhalf_path, weak_path, out_path, flores_isos):
    sp = json.load(open(split_path)); dh = json.load(open(devhalf_path)); weak = json.load(open(weak_path))["weak"]
    sel = sorted(sp["h_sel"]); forced = set(dh["forced_ids"])
    gen = sorted(sorted(sel, key=lambda i: hk("gen", i))[:40])
    weak_ids = sorted(sorted([i for i in sel if i not in forced], key=lambda i: hk("weak", i))[:60])
    G = ["amh", "tir"]
    W = "fon dyu kbp nus mos kam bam umb kea kmb vmw kab seh wol ewe luo tum yor".split()          # 17 weak Latin + yor (amendment 1); kea/seh/vmw/kab are in the list of 17
    spec = sorted(set(G + W))
    weights = {i: 2.5 for i in sorted(set(weak) | set(spec))}
    gen_langs = "afr ewe fon hau ibo kin nya orm sna som sot swh tsn wol xho yor zul twi".split()
    out = {"split_file_sha256": hashlib.sha256(open(split_path, "rb").read()).hexdigest(), "h_sel_sha256": sp["h_sel_sha256"],
           "general_dev": {"ids": gen, "ids_sha256": sha(gen), "languages": [i for i in gen_langs if i in flores_isos], "n_sentences": 40 * len([i for i in gen_langs if i in flores_isos]),
                           "old": "760 sentences = 40 x 19 languages (17 FLORES+ dev + pcm + twi, MAFAND dev mixed in)"},
           "weak_dev": {"ids": weak_ids, "ids_sha256": sha(weak_ids), "forced_ids_in": len(set(weak_ids) & forced), "languages": [i for i in sorted(set(weak) | set(spec)) if i in flores_isos],
                        "languages_without_flores_dev": [i for i in sorted(set(weak) | set(spec)) if i not in flores_isos],
                        "n_sentences": 60 * len([i for i in sorted(set(weak) | set(spec)) if i in flores_isos])},
           "draw_weights": {"weights": weights, "default": 1.0, "n_weighted": len(weights), "weak_list": weak, "weak_list_hash": "4472d7115aaf21497dbd32bf7531b3bbd8b67e8b4243ffbccc7850d6c54c6ab7 (rule file)",
                            "specialist_set": spec}}
    json.dump(out, open(out_path, "w"), indent=1)
    print({k: (v if k in ("split_file_sha256",) else {kk: vv for kk, vv in v.items() if kk != "ids"} if isinstance(v, dict) else v) for k, v in out.items()})


if __name__ == "__main__":
    flo = "afr amh bam bem dyu ewe fon hau ibo kab kam kbp kea kik kin kmb lin lua lug luo mlg mos nso nus nya orm run sag seh sna som sot ssw swh tir tsn tso tum twi umb vmw wol xho yor zul".split()
    main(*sys.argv[1:5], set(flo))
