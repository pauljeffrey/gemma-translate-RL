#!/bin/bash
# usage: push_run.sh WEIGHTS_DIR BRANCH RUN_ID RECIPE [best]   - pushes WEIGHTS_DIR/final (and best when it differs from final) to private ${HF_ORG}/gemma-0.4B-translate-rl-experiments, verifies, deletes the box copy on success
D=$1; BR=$2; RUN=$3; REC=$4; BBR=${5:-$BR-best}
source /workspace/clean_sft/env.sh; export HF_HOME=/workspace/clean_sft/hf_home
PY=/workspace/clean_sft/venv_train/bin/python; P=/workspace/e2405/hf_push.py
DATA="start: ${HF_ORG}/gemma-0.4B-translate-sft (or the mixed-best RL model for E-14); RL prompts: FineWeb-Edu + Wikipedia English windows, BBC News / Wikinews / Global Voices / Wikivoyage English news, AfriScience-MT train + IbomMT train human pairs (reference domain), science text (E-14); reward models africomet-qe-stl-1.1 + ssa-comet-mtl (+ SSA-COMET-MTL reference term and science terms in E-14), GlotLID, NLLB-3.3B pseudo-reference signal (reward only, kea kbp nus bam kab); held-out monitor africomet-stl-1.1. No TranslatePsy."
say() { echo "$(date -u +%FT%TZ) [push] $*"; }
if [ -f $D/best/model.safetensors ] && [ -f $D/final/model.safetensors ] && { [ -n "${FORCE_BEST:-}" ] || ! cmp -s $D/best/model.safetensors $D/final/model.safetensors; }; then
  say "pushing best of $RUN to $BBR"; $PY $P $D/best "$BBR" "$RUN (best by dev)" "$REC" "$DATA" --delete || say "PUSH FAILED best $RUN (box copy kept)"
else
  rm -rf $D/best; say "best of $RUN equals final or does not exist: not pushed separately"
fi
say "pushing final of $RUN to $BR"; $PY $P $D/final "$BR" "$RUN (pre-declared step, final)" "$REC" "$DATA" --delete && say "PUSHED $RUN" || say "PUSH FAILED final $RUN (box copy kept)"
