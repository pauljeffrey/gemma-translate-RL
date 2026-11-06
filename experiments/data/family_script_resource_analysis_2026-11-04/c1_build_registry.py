#!/usr/bin/env python
"""E-12 step 1: 125-language registry table with an EXTERNAL family classification (Glottolog 5.3 CLDF), script,
SFT token budgets, NLLB-mined share, NLLB flags, AfriCOMET availability and our own group label.

usage: c1_build_registry.py <glottolog-cldf v5.3 cldf/values.csv>

Stored inputs:
  reports/data/tag_token_map_2026-11-01.csv                                125-language registry
  reports/data/token_table_provisional_2026-11-01.csv                      tokens per language (Mtok)
  reports/data/nllb_regression_link_2026-11-03/nllb_share_by_language.csv  NLLB-mined share of stage-B tokens
  reports/data/post_editor_2026-11-03/e1/bench_mt/... nllb3.3b_greedy/metrics_by_language.csv  NLLB-3.3B drafts exist
  reports/data/rl_vs_sft_evidence_2026-11-04/groups_family.json            our own group labels
  reports/data/eval_weak_group_2026-11-03/results/results_mixedbest.csv    comet column -> AfriCOMET availability
  glottolog_v5.3_languages.csv (this folder; Glottolog 5.3 CLDF, CC BY 4.0)
  research_archive_2026-11-03/eval_385m/out_m385/predictions.jsonl         reference text -> empirical script
"""
import json, sys, unicodedata, collections
import pandas as pd

ROOT = "/path/to/repo"
D = ROOT + "/reports/data"
OUT = D + "/family_script_resource_analysis_2026-11-04"
VALUES = sys.argv[1]

reg = pd.read_csv(D + "/tag_token_map_2026-11-01.csv")[["iso", "language"]]
g = pd.read_csv(OUT + "/glottolog_v5.3_languages.csv")
byid = g.set_index("ID")
names = byid["Name"].to_dict()

override = {  # codes Glottolog does not carry; the family is not affected by the choice
    "orm": ("west2721", "macrolanguage orm; FLORES+ uses gaz_Latn (West Central Oromo)"),
    "mlg": ("plat1254", "macrolanguage mlg; FLORES+ uses plt_Latn (Plateau Malagasy)"),
    "kon": ("koon1244", "macrolanguage kon; Glottolog carries kng (South-Central Koongo)"),
    "luy": ("luyi1234", "macrolanguage luy; Glottolog family node 'Luyia'"),
    "ber": ("berb1260", "collective code ber; Glottolog family node 'Berber'"),
}
iso2gc = {r.ISO639P3code: r.Glottocode for r in g.itertuples() if isinstance(r.ISO639P3code, str)}
cls = {}
for line in open(VALUES, encoding="utf-8"):
    if ",classification," in line:
        p = line.rstrip("\n").split(",")
        if p[2] == "classification":
            cls[p[1]] = p[3]

recs = []
for r in reg.itertuples():
    gc, note = override.get(r.iso, (iso2gc.get(r.iso), ""))
    row = byid.loc[gc]
    ids = [x for x in cls.get(gc, "").split("/") if x]
    top_id = ids[0] if ids else gc
    pn = [names[i] for i in ids]
    ac_branch = names[ids[1]] if ids and ids[0] == "atla1278" and len(ids) > 1 else ""
    # coarse "branch" label from the Glottolog path itself (no hand-made grouping)
    if "Narrow Bantu" in pn:
        branch = "Atlantic-Congo: Narrow Bantu"
    elif ids and ids[0] == "atla1278":
        if len(pn) > 1 and pn[1] == "Volta-Congo" and len(pn) > 2:
            b = pn[2]
            if b == "North Volta-Congo" and len(pn) > 3:
                b = pn[3]
            branch = "Atlantic-Congo: " + b
        else:
            branch = "Atlantic-Congo: " + (pn[1] if len(pn) > 1 else "(root)")
    elif ids and len(pn) > 1 and pn[0] in ("Afro-Asiatic",):
        branch = "Afro-Asiatic: " + pn[1]
    elif ids and len(pn) > 1 and pn[0] == "Indo-European":
        branch = "Indo-European: " + ("Germanic (Afrikaans, English-lexifier creoles)" if "Germanic" in pn else "Romance (Portuguese/French-lexifier creoles)" if "Romance" in pn else "other")
    else:
        branch = pn[0] if pn else names[gc]
    top_name = names[top_id]
    reg_family = {"Atlantic-Congo": ("Atlantic-Congo: Narrow Bantu" if "Narrow Bantu" in pn else "Atlantic-Congo: non-Bantu"),
                  "Afro-Asiatic": "Afro-Asiatic", "Nilotic": "Nilotic", "Mande": "Mande",
                  "Indo-European": "Indo-European (creoles, Afrikaans)"}.get(top_name, "Other (Austronesian, Saharan, Songhay, Ta-Ne-Omotic)")
    recs.append(dict(
        family_branch=branch, family_reg=reg_family,
        iso=r.iso, language=r.language, glottocode=gc, mapping_note=note, glottolog_name=row.Name,
        glottolog_level=row.Level, glottolog_macroarea=row.Macroarea,
        glottolog_top_family=names[top_id], glottolog_path=" > ".join(pn),
        glottolog_atlantic_congo_branch=ac_branch,
        glottolog_narrow_bantu=int("Narrow Bantu" in pn), glottolog_volta_congo=int("Volta-Congo" in pn),
        glottolog_path_first4=" > ".join(pn[:4]), lat=row.Latitude, lon=row.Longitude))
R = pd.DataFrame(recs)

grp = json.load(open(D + "/rl_vs_sft_evidence_2026-11-04/groups_family.json"))
R["our_group_evid_table4"] = R.iso.map(grp)

tt = pd.read_csv(D + "/token_table_provisional_2026-11-01.csv")
tt = tt[tt.iso != "TOTAL"]
R = R.merge(tt, on="iso", how="left")
nl = pd.read_csv(D + "/nllb_regression_link_2026-11-03/nllb_share_by_language.csv")[
    ["iso", "tokens_B", "share_nllb_mined", "share_nllb_strict", "share_nllb_mined_of_translation"]]
R = R.merge(nl, on="iso", how="left")
R["sft_tokens_M"] = R.all_streams_Mtok
R["sft_tokens_source"] = "token_table all_streams_Mtok"
m = R.sft_tokens_M.isna() & R.tokens_B.notna()
R.loc[m, "sft_tokens_M"] = R.loc[m, "tokens_B"] / 1e6
R.loc[m, "sft_tokens_source"] = "stage-B pack tokens (nllb_share_by_language.csv)"
R.loc[R.sft_tokens_M.isna(), "sft_tokens_source"] = "none (language absent from both token tables)"

# NLLB-3.3B drafts available in the post-editor run (FLORES+ languages NLLB-200 supports) and published NLLB number
nd = pd.read_csv(D + "/post_editor_2026-11-03/e1/eval_e1b/nllb3.3b_greedy/metrics_by_language.csv")
nd = nd[(nd.benchmark == "flores_plus") & (nd.system == "draft")]
R["nllb_draft_available_pe_run"] = R.iso.isin(set(nd.iso)).astype(int)
pub = pd.read_csv(D + "/eval_385m_results_2026-11-03/published_nllb_flores_per_language.csv")
R = R.merge(pub[["iso", "nllb_3p3b"]].rename(columns={"nllb_3p3b": "nllb_3p3b_published_chrfpp"}), on="iso", how="left")

# AfriCOMET availability on FLORES+ (held-out reference-based checkpoint)
rm = pd.read_csv(D + "/eval_weak_group_2026-11-03/results/results_mixedbest.csv")
rm = rm[rm.benchmark == "flores_plus"]
R["africomet_available_flores"] = R.iso.map(rm.set_index("iso").comet.notna().astype(int)).fillna(-1).astype(int)
R["flores_plus_evaluated"] = R.iso.isin(set(rm.iso)).astype(int)

# empirical script from reference text (all benchmarks of the SFT predictions file)
cnt = collections.defaultdict(collections.Counter)
bench = collections.defaultdict(set)
with open(ROOT + "/research_archive_2026-11-03/eval_385m/out_m385/predictions.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        iso = d["iso"]
        bench[iso].add(d["benchmark"])
        if len(cnt[iso]) and sum(cnt[iso].values()) > 400000:
            continue
        for ch in d["reference"]:
            if ch.isalpha():
                try:
                    cnt[iso][unicodedata.name(ch).split()[0]] += 1
                except ValueError:
                    pass
emp, empshare = {}, {}
for iso, c in cnt.items():
    tot = sum(c.values())
    k, v = c.most_common(1)[0]
    emp[iso], empshare[iso] = k, v / tot
R["script_empirical"] = R.iso.map(emp)
R["script_empirical_share"] = R.iso.map(empshare)
R["benchmarks_present"] = R.iso.map(lambda i: ",".join(sorted(bench.get(i, []))))
R["n_benchmarks"] = R.iso.map(lambda i: len(bench.get(i, [])))

# assigned script for languages without a benchmark (prior knowledge, flagged)
assigned = {"amh": "ETHIOPIC", "tir": "ETHIOPIC", "cop": "COPTIC", "arz": "ARABIC", "ary": "ARABIC", "aeb": "ARABIC",
            "zgh": "TIFINAGH", "tzm": "TIFINAGH"}  # everything else Latin
uncertain = {"ber", "taq", "shi", "tzm", "knc", "zgh"}
scr, src = [], []
for r in R.itertuples():
    if isinstance(r.script_empirical, str):
        scr.append(r.script_empirical); src.append("empirical (reference text)")
    else:
        scr.append(assigned.get(r.iso, "LATIN")); src.append("assigned (prior knowledge)" + ("; UNCERTAIN" if r.iso in uncertain else ""))
R["script"] = scr
R["script_source"] = src
R.to_csv(OUT + "/registry_125.csv", index=False)
print(R.shape)
print(R.script.value_counts())
print(R[R.script != "LATIN"][["iso", "language", "script", "script_empirical_share", "script_source"]])
print(R[(R.script_empirical_share < 0.99)][["iso", "script_empirical", "script_empirical_share"]])
print(R.glottolog_top_family.value_counts())
print(R.isna().sum()[lambda s: s > 0])
