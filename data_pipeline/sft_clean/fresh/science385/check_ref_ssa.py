#!/usr/bin/env python3
"""GPU preflight of the reference-based SSA-COMET-MTL worker used by the science RL reward (scorer.ref_ssa). Exit 0 when the worker starts and returns finite scores for src + mt + ref."""
import math, sys
sys.path.insert(0, "/workspace/clean_rerun/tree/gemma_mt")
from gemma_mt.rl.comet import WorkerScorer, SSA_MTL_MODEL_ID
w = WorkerScorer(python="/workspace/venv_gemma/bin/python", model_id=SSA_MTL_MODEL_ID, batch_size=8, reference_based=True,
                 pythonpath="/workspace/sft_build/pylibs_tf4:/workspace/sft_build/pylibs", startup_timeout=600, request_timeout=300)
try:
    s = w.view(0, 8)(["The dose was 5 mg.", "Malaria is a disease."], ["Doze ya 5 mg.", "Malaria ni ugonjwa."], ["Doze ya 5 mg.", "Malaria ni ugonjwa."])
    print("scores", s); sys.exit(0 if len(s) == 2 and all(math.isfinite(float(x)) for x in s) else 1)
finally:
    w.close()
