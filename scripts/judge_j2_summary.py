"""J2 housekeeping: cost summary from the ledger and one file with the prompt templates and one example per prompt type.  python judge_j2_summary.py <dir>"""
import csv
import json
import sys
from pathlib import Path

from judge_j2_common import MTE_CLOSING, MTE_HEADER

D = Path(sys.argv[1])
led = json.load(open(D / "ledger.json"))
with open(D / "cost_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "n_requests", "max_output_tokens_cap", "input_tokens", "output_tokens_incl_thinking", "worst_case_usd_at_submit", "actual_usd", "job"])
    for e in led:
        w.writerow([e["group"], e["n"], e["cap"], e["actual_input_tokens"], e["actual_output_tokens"], round(e["worst_usd"], 4), round(e["actual_usd"], 4), e["job"]])
    w.writerow(["TOTAL", sum(e["n"] for e in led), "", sum(e["actual_input_tokens"] for e in led), sum(e["actual_output_tokens"] for e in led), "", round(sum(e["actual_usd"] for e in led), 4), ""])
print("total actual USD", sum(e["actual_usd"] for e in led))
parts = ["PROMPT TEMPLATES AND ONE EXAMPLE PER PROMPT TYPE (J2, 2026-10-25)\n"
         "Source: Li et al. (EMNLP 2025), arXiv 2506.04557, Figure 8 (reference-based, score only) and Figure 9 (QE, score only).\n"
         "Figure 9 = Figure 8 with the 'Reference: ...' field removed from the five demonstrations and from the test line; everything else identical.\n"
         "Exact request text follows; the test line ends with 'Score: ' (trailing space) and the judge returns a bare number.\n\n"
         "=== HEADER (both figures) ===\n" + MTE_HEADER + "=== CLOSING LINE (both figures; the test line follows it) ===\n" + MTE_CLOSING + "\n"]
for title, fn in (("RUN 1: Figure 9 (QE), SSA-MTE eng-yor, 5 demonstrations from SSA-MTE train, test = pilot segment", "prompt_example_qe_run1_eng-yor.txt"),
                  ("RUN 1: Figure 9 (QE), AfriMTE eng-swh", "prompt_example_qe_run1_eng-swh.txt"),
                  ("RUN 2: Figure 8 (reference-based), FLORES+ devtest Hausa, RL-final output, human FLORES+ reference", "prompt_example_run2_flores_hau_rl.txt"),
                  ("RUN 3: Figure 9 (QE), final2 whole-document output, aligned unit, Hausa", "prompt_example_run3_final2_whole_hau.txt"),
                  ("RUN 3: Figure 9 (QE), final2 sentence-by-sentence wrapper (sentfix), Hausa", "prompt_example_run3_final2_sentfix_hau.txt"),
                  ("RUN 3: Figure 9 (QE), NLLB-3.3B sentence-by-sentence, Yoruba", "prompt_example_run3_nllb3b3_sent_yor.txt")):
    parts.append(f"\n\n######## {title}\n######## (file: {fn})\n\n" + (D / fn).read_text(encoding="utf-8"))
(D / "prompt_templates_and_examples.txt").write_text("".join(parts), encoding="utf-8")
