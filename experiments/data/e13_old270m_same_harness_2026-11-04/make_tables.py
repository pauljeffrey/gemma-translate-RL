#!/usr/bin/env python
"""Tables of reports/e13_old270m_same_harness_2026-11-04.md from evidence_e13.json.   python make_tables.py > tables.md"""
import json, os
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence_e13.json")))
B = [("flores_plus", "FLORES+ devtest"), ("ntrex", "NTREX"), ("mafand", "MAFAND")]
print("## Table 1. Language-macro chrF++ on rows that are clean for both generations (paired bootstrap, 1000 resamples, 95% CI)\n")
print("| benchmark (languages) | old 270M SFT | 385M SFT | 385M SFT - old SFT [CI] | old 270M RL (rl2c) | 385M RL (mixed best) | 385M RL - old RL [CI] |\n|---|---:|---:|---|---:|---:|---|")
for b, lab in B:
    a = d["stages"]["old_sft->sft"][b]["pp"]; r = d["stages"]["old_rl2c->mixed"][b]["pp"]
    print(f"| {lab} ({a[5]}) | {a[0]:.2f} | {a[1]:.2f} | {a[2]:+.2f} [{a[3]:+.2f}, {a[4]:+.2f}] | {r[0]:.2f} | {r[1]:.2f} | {r[2]:+.2f} [{r[3]:+.2f}, {r[4]:+.2f}] |")
print("\n## Table 2. Held-out AfriCOMET-STL (language macro; languages with a score in parentheses)\n")
print("| benchmark | old SFT | 385M SFT | diff [CI] | old RL | 385M RL | diff [CI] |\n|---|---:|---:|---|---:|---:|---|")
for b, lab in B:
    a = d["stages"]["old_sft->sft"][b]["com"]; r = d["stages"]["old_rl2c->mixed"][b]["com"]
    print(f"| {lab} ({a[5]}) | {a[0]:.4f} | {a[1]:.4f} | {a[2]:+.4f} [{a[3]:+.4f}, {a[4]:+.4f}] | {r[0]:.4f} | {r[1]:.4f} | {r[2]:+.4f} [{r[3]:+.4f}, {r[4]:+.4f}] |")
print("\n## Table 3. 385M RL (mixed best) minus 385M SFT on the same rows, chrF++\n")
print("| benchmark | SFT | RL | diff [CI] |\n|---|---:|---:|---|")
for b, lab in B:
    a = d["stages"]["sft->mixed"][b]["pp"]
    print(f"| {lab} ({a[5]}) | {a[0]:.2f} | {a[1]:.2f} | {a[2]:+.2f} [{a[3]:+.2f}, {a[4]:+.2f}] |")
print("\n## Table 4. Old SFT -> old RL (rl2c), the first research's own RL gain on these rows, chrF++ (macro entry 'all')\n")
print("| benchmark | old SFT | old RL | diff [CI] |\n|---|---:|---:|---|")
for b, lab in B + [("all_cells", "all cells")]:
    a = d["macro"]["all"][b]["pp"]
    print(f"| {lab} ({a[5]}) | {a[0]:.2f} | {a[1]:.2f} | {a[2]:+.2f} [{a[3]:+.2f}, {a[4]:+.2f}] |")
print("\n## Table 5. Finished rate / loop rate (reward's flag) / hypothesis-to-reference length, rows clean for both generations\n")
print("| system | FLORES+ (n rows) | NTREX | MAFAND |\n|---|---|---|---|")
for s, lab in [("old_sft", "old 270M SFT"), ("sft", "385M SFT"), ("old_rl2c", "old 270M RL (rl2c)"), ("mixed", "385M RL (mixed best)")]:
    print(f"| {lab} | " + " | ".join(f"{d['flags'][s][b]['finished']:.3f} / {d['flags'][s][b]['loop']:.3f} / {d['flags'][s][b]['len_ratio']:.2f} (n = {d['flags'][s][b]['n']})" for b, _ in B) + " |")
