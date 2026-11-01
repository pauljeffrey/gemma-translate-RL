#!/usr/bin/env python3
"""r4: document-level H_cal / H_sel split v2 of the held-out FLORES+ dev half (agent R4, 2026-10-28, written BEFORE any calibration was fitted).

    python r4_split_v2.py --devhalf-split devhalf_split.json --eng-dev <dev/eng_Latn.jsonl> --out heldout_split_cal_sel_v2.json

Why (see reports/clean_calibration_2026-10-28.md section 1): the v1 rule of chain/hcal.py (next sha256 bit of the id) puts sentences of the SAME Wikinews /
Wikivoyage / Wikibooks article into both H_cal and H_sel (111 of 257 held-out articles; 327 of the 557 held-out ids have a sibling sentence on the other side), and
58 of the 100 ids that defined the weak-language list (role S, reports/data/weak_selection_2026-10-26/) sit in H_cal, so a role-S id is also a role-T2 id.

Rule v2 (no tuning knob, no target size, deterministic; the unit is the article = FLORES+ `url`, all sentences of an article have consecutive ids):
    H_sel = every held-out id whose article
              (a) contains at least one of the 100 forced ids (the ids that defined the weak list: role S), OR
              (b) contains no id of the FLORES+ dev TRAIN half (no article neighbour in the T1 specialist data, so selection on it is not helped by neighbour sentences)
    H_cal = the other held-out ids.
Only ids, counts and hashes are written (no FLORES+ text, no URL). Same JSON keys as v1 (`h_cal`, `h_sel`, ...), so `chain/hcal.py refs --split <this file>` works unchanged.
"""
import argparse, collections, hashlib, json


def sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devhalf-split", required=True)
    ap.add_argument("--eng-dev", required=True, help="dev/eng_Latn.jsonl of openlanguagedata/flores_plus (only the id and url fields are read)")
    ap.add_argument("--v1", default=None, help="heldout_split_cal_sel.json (v1) for the comparison numbers")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.devhalf_split))
    held, train, forced = set(d["heldout_ids"]), set(d["train_ids"]), set(d["forced_ids"])
    url = {}
    for line in open(a.eng_dev, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            url[int(r["id"])] = r["url"]
    assert held | train == set(url) and not held & train and forced <= held
    art = collections.defaultdict(list)
    for i, u in url.items():
        art[u].append(i)
    for u, v in art.items():
        assert max(v) - min(v) + 1 == len(v), "article ids not consecutive"
    names = sorted(art, key=lambda u: min(art[u]))
    doc_index = {u: k for k, u in enumerate(names)}                   # article number by first id (no URL is stored)
    forced_art = {url[i] for i in forced}
    train_art = {url[i] for i in train}
    sel_art = {u for u in art if (u in forced_art or u not in train_art) and set(art[u]) & held}
    cal = sorted(i for i in held if url[i] not in sel_art)
    sel = sorted(i for i in held if url[i] in sel_art)
    assert not set(cal) & set(sel) and len(cal) + len(sel) == len(held) and forced <= set(sel)
    assert not ({url[i] for i in cal} & {url[i] for i in sel})                  # no article on both sides
    out = {"version": "v2 (agent R4, document level, fixed before any calibration was fitted)",
           "rule": "unit = FLORES+ dev article (url). H_sel = held-out ids of articles that contain one of the 100 forced ids (role S: they defined the weak list) or that contain no id of the "
                   "dev train half (no T1 neighbour sentence); H_cal = all other held-out ids. Deterministic, no hash and no target size.",
           "devhalf_heldout_ids_sha256": sha(sorted(held)), "heldout_ids_sha256": sha(sorted(held)),
           "h_cal": cal, "h_sel": sel, "n_cal": len(cal), "n_sel": len(sel), "h_cal_sha256": sha(cal), "h_sel_sha256": sha(sel),
           "forced_in_cal": len(forced & set(cal)), "forced_in_sel": len(forced & set(sel)),
           "n_articles_held": len({url[i] for i in held}), "n_articles_cal": len({url[i] for i in cal}), "n_articles_sel": len({url[i] for i in sel}),
           "articles_on_both_sides": 0,
           "cal_ids_with_train_half_article_neighbour": sum(1 for i in cal if url[i] in train_art),
           "sel_ids_with_train_half_article_neighbour": sum(1 for i in sel if url[i] in train_art),
           "article_index_of_id": {"note": "article number (by first id) of every held-out id", "map": {str(i): doc_index[url[i]] for i in sorted(held)}}}
    if a.v1:
        v1 = json.load(open(a.v1))
        c1, s1 = set(v1["h_cal"]), set(v1["h_sel"])
        out["v1_comparison"] = {"v1_n_cal": len(c1), "v1_n_sel": len(s1), "v1_articles_on_both_sides": sum(1 for u in art if set(art[u]) & c1 and set(art[u]) & s1),
                                "v1_forced_in_cal": len(forced & c1), "overlap_cal_v1_v2": len(c1 & set(cal)), "overlap_sel_v1_v2": len(s1 & set(sel))}
    json.dump(out, open(a.out, "w"))
    print({k: v for k, v in out.items() if k not in ("h_cal", "h_sel", "article_index_of_id")})


if __name__ == "__main__":
    main()
