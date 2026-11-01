#!/usr/bin/env python3
"""R14: markdown tables for reports/nllb600m_*.md from reports/data/nllb600m_*.csv + nllb_reward_calibration_v1.json (stdlib only). Writes to the directory given as argv[1]."""
import csv, json, os, statistics as st, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
D = ROOT + "/reports/data/"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
rd = lambda p: list(csv.DictReader(open(D + p)))
fl = lambda x: float(x) if x not in (None, "") else None
mac, dec = rd("nllb600m_macro.csv"), rd("nllb600m_decision_table.csv")
def M(b, sub, s, sysn, metric="chrf"):
    for r in mac:
        if r["benchmark"] == b and r["subset"] == sub and r["set"] == s and r["system"] == sysn and r["metric"] == metric: return r
# T1
L = ["| benchmark (English -> X) | langs | ours chrF++ | NLLB-3.3B | delta [95% CI] | ours spBLEU | NLLB-3.3B | delta [95% CI] |", "|---|---|---|---|---|---|---|---|"]
names = {"flores_plus": "FLORES+ devtest", "ntrex": "NTREX-128", "mafand": "MAFAND-MT test", "smol_sent": "SmolSent", "salt": "SALT"}
for b, n in names.items():
    for sub in ("no_pair", "clean"):
        a, c = M(b, sub, "all", "nllb_n33_beam4"), M(b, sub, "all", "nllb_n33_beam4", "spbleu")
        if a and c: L.append(f"| {n} `{sub}` | {a['n_langs']} | {a['ours_macro']} | {a['nllb_macro']} | {fl(a['delta']):+.2f} [{fl(a['lo']):+.2f}, {fl(a['hi']):+.2f}] | {c['ours_macro']} | {c['nllb_macro']} | {fl(c['delta']):+.2f} [{fl(c['lo']):+.2f}, {fl(c['hi']):+.2f}] |")
open(OUT + "/t1_macro_benchmarks.md", "w").write("\n".join(L))
# T2
nm = {"all": "all covered FLORES+ languages", "qe_validated": "QE-validated", "blend": "QE-blend", "consensus_only": "consensus-only", "sft_only": "SFT-only (no RL)", "fam:Bantu": "Bantu",
      "fam:Atlantic-Congo (non-Bantu)": "Atlantic-Congo (non-Bantu)", "fam:Afroasiatic": "Afroasiatic (Ge'ez, Cushitic, Berber)", "fam:Nilo-Saharan": "Nilo-Saharan", "fam:Mande": "Mande",
      "fam:Other (Indo-European, Austronesian)": "Afrikaans, Malagasy", "fam:Creole / Pidgin": "Creole (Kabuverdianu)"}
L = ["| set | langs | ours chrF++ | NLLB-3.3B | delta [95% CI] | NLLB-600M | delta [95% CI] | NLLB-600M greedy | delta |", "|---|---|---|---|---|---|---|---|---|"]
for k, n in nm.items():
    a, b, c = M("flores_plus", "no_pair", k, "nllb_n33_beam4"), M("flores_plus", "no_pair", k, "nllb_n600_beam4"), M("flores_plus", "no_pair", k, "nllb_n600_greedy")
    if a: L.append(f"| {n} | {a['n_langs']} | {a['ours_macro']} | {a['nllb_macro']} | {fl(a['delta']):+.2f} [{fl(a['lo']):+.2f}, {fl(a['hi']):+.2f}] | {b['nllb_macro']} | {fl(b['delta']):+.2f} [{fl(b['lo']):+.2f}, {fl(b['hi']):+.2f}] | {c['nllb_macro']} | {fl(c['delta']):+.2f} |")
open(OUT + "/t2_macro_groups.md", "w").write("\n".join(L))
# T3
cov = sorted([r for r in dec if r["benchmark"]], key=lambda r: -fl(r["d33"]))
L = ["| iso | language | group | bench (n) | ours | NLLB-3.3B | d3.3B [CI] | NLLB-600M | d600M [CI] | category (3.3B) | useful as reward signal? (3.3B) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in cov:
    d6 = f"{fl(r['d600']):+.1f} [{fl(r['d600_lo']):+.1f}, {fl(r['d600_hi']):+.1f}]" if r["d600"] else "-"
    L.append(f"| `{r['iso']}` | {r['language']} | {r['group']} | {r['benchmark']} ({r['n']}) | {r['ours_chrf']} | {r['nllb33_chrf']} | {fl(r['d33']):+.1f} [{fl(r['d33_lo']):+.1f}, {fl(r['d33_hi']):+.1f}] | {r['nllb600_chrf'] or '-'} | {d6} | {r['category_3.3B'].split(' [')[0]} | {r['useful_as_reward_signal_3.3B']} |")
open(OUT + "/t3_decision.md", "w").write("\n".join(L))
# T4 teacher manifest
mf = rd("nllb600m_teacher_manifest.csv")
L = ["| iso | language | rows | tokens (k) | budget (k) | teacher chrF++ devtest / dev | student chrF++ | delta | sentences translated | dropped (LID / ratio / copy / loop / QE) | tokens a teacher pass yields per 100k English sentences (M) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in mf:
    ys = f"{int(r['tokens_per_100k_english_sentences'])/1e6:.1f}"
    L.append(f"| `{r['iso']}` | {r['language']} | {r['rows']} | {int(r['tokens'])/1000:.0f} | {int(r['budget_tokens'])/1000:.0f} | {r['teacher_devtest_chrf']} / {r['teacher_dev_chrf'] or '-'} | {r['student_final1433_chrf']} | {fl(r['delta']):+.1f} | {r['sentences_translated']} | {r['sent_lid']} / {r['sent_ratio']} / {r['sent_copy']} / {r['sent_repetition']} / {r['sent_qe_low']} |  {ys} |")
L.append(f"| | **total** | **{sum(int(r['rows']) for r in mf)}** | **{sum(int(r['tokens']) for r in mf)/1000:.0f}** | {sum(int(r['budget_tokens']) for r in mf)/1000:.0f} | | | | | | |")
open(OUT + "/t4_teacher.md", "w").write("\n".join(L))
# T5 hack aggregate
ag = rd("nllb600m_reward_hack_aggregate.csv")
lab = {"africomet_qe": "AfriCOMET-QE (current)", "ssa_mtl_qe": "SSA-COMET-MTL (current ens. member)", "nllb3b3_pseudoref": "NLLB-3.3B pseudo-reference chrF++ (A)", "nllb3b3_forced": "NLLB-3.3B forced-decoding (B)",
       "nllb3b3_combo_min": "min(A, B) shaped", "nllb3b3_combo_mean": "mean(A, B) shaped", "nllb600m_pseudoref": "NLLB-600M pseudo-reference (A)", "nllb600m_forced": "NLLB-600M forced-decoding (B)"}
L = []
for s in ("all covered", "teacher better (3.3B delta>=2)", "mode=blend"):
    L.append(f"\n**{s}** ({[r for r in ag if r['set'] == s][0]['n_langs']} languages for the 3.3B rows)\n")
    L.append("| signal | AUC human vs mismatched | hard-class failure | copy_source | copy_noisy | partial copy | append_unrelated | drop_half | shuffle |\n|---|---|---|---|---|---|---|---|---|")
    for r in ag:
        if r["set"] == s and r["metric"] in lab: L.append(f"| {lab[r['metric']]} | {r['auc_mismatch']} | {r['fail_hard']} | {r['fail_copy_source']} | {r['fail_copy_noisy']} | {r['fail_partial']} | {r['fail_append_unrelated']} | {r['fail_drop_half']} | {r['fail_shuffle']} |")
open(OUT + "/t5_hack.md", "w").write("\n".join(L))
# T6 policy
pol = rd("nllb600m_reward_policy_per_language.csv")
def mean(rs, c):
    v = [fl(r.get(c)) for r in rs]; v = [x for x in v if x is not None and x == x]
    return (f"{st.mean(v):.2f}", len(v)) if v else ("-", 0)
L = ["| metric | Spearman with human chrF++ (pooled) | within-group Spearman (8 samples of a prompt) | partial Spearman given consensus | partial given AfriCOMET-QE | langs |", "|---|---|---|---|---|---|"]
for key, n in (("consensus", "group consensus (current)"), ("africomet_qe", "AfriCOMET-QE (current)"), ("ssa_mtl_qe", "SSA-COMET-MTL"), ("nllb3b3_pseudoref", "NLLB-3.3B pseudo-reference (A)"), ("nllb3b3_forced", "NLLB-3.3B forced (B)"),
               ("nllb3b3_combo_min", "min(A, B)"), ("nllb3b3_combo_mean", "mean(A, B)"), ("nllb600m_pseudoref", "NLLB-600M pseudo-reference (A)*"), ("nllb600m_forced", "NLLB-600M forced (B)*")):
    sp, k = mean(pol, "spearman_" + key); wg, _ = mean(pol, "within_group_" + key)
    pg = mean(pol, f"partial_{key}_given_consensus")[0] if key.startswith("nllb") else "-"
    pq = mean(pol, f"partial_{key}_given_qe")[0] if key.startswith("nllb") else "-"
    L.append(f"| {n} | {sp} | {wg} | {pg} | {pq} | {k} |")
open(OUT + "/t6_policy.md", "w").write("\n".join(L))
# T7 per language reliability
cal = json.load(open(D + "nllb_reward_calibration_v1.json"))
dd = {r["iso"]: r for r in dec}
L = ["| iso | group | teacher - student (chrF++) | AUC pseudo-ref (dev) | hack: AUC / hard-failure | policy: within-group Spearman / partial given consensus | forced AUC | use | weight (default 0.35 x factor) |", "|---|---|---|---|---|---|---|---|---|"]
for k, v in sorted(cal.items(), key=lambda kv: -(kv[1]["nllb"]["teacher_vs_student"]["delta"] or -99)):
    b = v["nllb"]; p = b["pseudoref"]; f = b["forced"]; w = round(0.35 * p["reliability"] * p["weight_factor"], 3) if b["use"] else 0
    g = lambda x, n=2: "-" if x is None else f"{x:.{n}f}"
    L.append(f"| `{k}` | {dd[k]['group']} | {b['teacher_vs_student']['delta']:+.1f} | {p['auc_vs_mismatched']:.3f} | {g(p['hack_auc_mismatch'], 3)} / {g(p['hack_fail_hard'], 3)} | {g(p['policy_within_group_spearman'])} / {g(p['policy_partial_given_consensus'])} | {f['auc_vs_mismatched']:.3f} | {'yes' if b['use'] else 'no'} | {w} |")
open(OUT + "/t7_reliability.md", "w").write("\n".join(L))
print("ok")
