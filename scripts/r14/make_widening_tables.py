#!/usr/bin/env python3
"""R14: markdown tables for reports/nllb_signal_widening_2026-10-23.md from reports/data/nllb_widening_*.csv and nllb_reward_calibration_v2.json (stdlib only). argv[1] = output dir."""
import csv, json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
D = ROOT + "/reports/data/"; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
rd = lambda p: list(csv.DictReader(open(D + p)))
f = lambda x: float(x) if x not in (None, "", "None") else None
pol, hk = {r["iso"]: r for r in rd("nllb_widening_policy_per_language.csv")}, {r["iso"]: r for r in rd("nllb_widening_hack_auc_per_language.csv")}
cal = json.load(open(D + "nllb_reward_calibration_v2.json")); dec = {r["iso"]: r for r in rd("nllb600m_decision_table.csv")}
def ci(r, k, d=2): return f"{f(r[k]):+.{d}f} [{f(r[k+'_lo']):+.{d}f}, {f(r[k+'_hi']):+.{d}f}]"
order = sorted([k for k, v in cal.items() if k in pol and (dec.get(k) or {}).get("useful_as_reward_signal_3.3B") == "yes" or (k in pol and v["nllb"]["use"])], key=lambda k: -(f(dec[k]["d33"]) if dec.get(k) and dec[k].get("d33") else -99))
L = ["| iso | group / reward mode | teacher - student | Spearman A [95% CI] | Spearman current reward [95% CI] | A - current [95% CI] | within-group A / current | partial A given current [95% CI] | partial A given QE-ens + consensus | use | weight |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for k in order:
    p = pol[k]; n = cal[k]["nllb"]; w = n["weight"]
    L.append(f"| `{k}` | {dec[k]['group'] if k in dec else '-'} / {p['mode']} | {f(dec[k]['d33']):+.1f} | {f(p['sp_A']):.2f} [{f(p['sp_A_lo']):.2f}, {f(p['sp_A_hi']):.2f}] | {f(p['sp_cur']):.2f} [{f(p['sp_cur_lo']):.2f}, {f(p['sp_cur_hi']):.2f}] | {ci(p, 'd_A_cur')} | {f(p['wg_A']):.2f} / {f(p['wg_cur']):.2f} | {ci(p, 'partial_A_given_cur')} | {f(p['partial_A_given_Q_cons']):.2f} | {'yes' if n['use'] else 'no'} | {w} |")
open(OUT + "/tw1_policy.md", "w").write("\n".join(L))
L = ["| iso | AUC mismatch A / QE-ens | AUC copy A / QE-ens | AUC padded (human + unrelated sentence) A / QE-ens | AUC drop-half A / QE-ens | AUC partial copy A / QE-ens |", "|---|---|---|---|---|---|"]
for k in order:
    if k not in hk: continue
    h = hk[k]; g = lambda n: f"{f(h['auc_%s_A' % n]):.3f} [{f(h['auc_%s_A_lo' % n]):.2f}, {f(h['auc_%s_A_hi' % n]):.2f}] / {f(h['auc_%s_Q' % n]):.3f}"
    L.append(f"| `{k}` | {g('mismatch')} | {g('copy')} | {g('padded')} | {g('drop_half')} | {g('partial_copy')} |")
open(OUT + "/tw2_hack.md", "w").write("\n".join(L))
if os.path.exists(D + "nllb_widening_ssamte_human_da.csv"):
    s = rd("nllb_widening_ssamte_human_da.csv")
    L = ["| language | judgements / sources | Spearman A [95% CI] | AfriCOMET-QE | SSA-COMET-MTL | QE ensemble | A - SSA-COMET [95% CI] | within-source A / AfriCOMET / SSA |", "|---|---|---|---|---|---|---|---|"]
    for r in s:
        L.append(f"| `{r['iso']}` | {r['n_judgements']} / {r['n_sources']} | {f(r['sp_A']):.2f} [{f(r['sp_A_lo']):.2f}, {f(r['sp_A_hi']):.2f}] | {f(r['sp_africomet']):.2f} | {f(r['sp_ssa_mtl']):.2f} | {('%.2f' % f(r['sp_QEens'])) if r.get('sp_QEens') else '-'} | [{f(r['d_A_minus_ssa_lo']):+.2f}, {f(r['d_A_minus_ssa_hi']):+.2f}] | {f(r['wsrc_A']):.2f} / {f(r['wsrc_africomet']):.2f} / {f(r['wsrc_ssa_mtl']):.2f} |")
    open(OUT + "/tw3_ssamte.md", "w").write("\n".join(L))
L = ["| iso | weight (SFT mixture without NLLB teacher data) | weight with the double-pull rule (teacher data in the mixture) | teacher data in teacher_v1 | alignment factor | factor |", "|---|---|---|---|---|---|"]
for k in sorted([k for k, v in cal.items() if v["nllb"]["use"]], key=lambda k: -cal[k]["nllb"]["weight"]):
    n = cal[k]["nllb"]; p = n["pseudoref"]
    L.append(f"| `{k}` | {n['weight']} | {n['weight_if_teacher_sft_in_mixture']} | {'yes' if n['teacher_data_in_teacher_v1'] else 'no'} | {p['alignment_factor']} | {p['weight_factor']:.2f} |")
open(OUT + "/tw4_weights.md", "w").write("\n".join(L))
print("ok", len(order))
