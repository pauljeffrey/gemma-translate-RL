#!/bin/bash
# E-02 end to end on the A100 (GPU must be free). usage: e02_run.sh [smoke]   markers in /workspace/e2405/e02/
O=/workspace/e2405/e02; E=/workspace/eval_385m; mkdir -p $O /dev/shm/e02 /dev/shm/tmp
say() { echo "$(date -u +%FT%TZ) [e02] $*" | tee -a $O/log.txt; }
[ "$(df --output=avail -k / | tail -1)" -lt 1048576 ] && { say "LOW DISK: stop"; exit 3; }
export GEMMA_MT_TAG_MODE=single_token HF_HOME=/workspace/clean_sft/hf_home GEMMA_MT_TREE=$E/gemma_mt_tree PYTHONPATH=$E/gemma_mt_tree TMPDIR=/dev/shm/tmp
V=$E/vllm_cache_guard                               # compile caches on the root disk (/dev/shm is noexec)
export VLLM_CACHE_ROOT=$V/vllm TRITON_CACHE_DIR=$V/triton TORCHINDUCTOR_CACHE_DIR=$V/inductor XDG_CACHE_HOME=$V/xdg VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
SFT=/workspace/clean_rerun/models/g15_385m_sft_final; RL=/dev/shm/rl_mixed_ckpt/best; PYV=/venv/main/bin/python
dec() {  # name model out extra...    retries on vLLM start collisions
  local name=$1 model=$2 out=$3; shift 3
  for a in 1 2 3 4; do
    $PYV $O/../e02_decode.py --model $model --name $name --out $out "$@" > $O/dec_$(basename $out .json).log 2>&1 && [ -f $out ] && return 0
    say "decode $name attempt $a failed (see dec_$(basename $out .json).log); retry in 180 s"; sleep 180
  done; return 1
}
if [ "${1:-}" = smoke ]; then
  say "SMOKE: SFT, 120 items, greedy + mbr16 + beam4 (width rule)"
  dec sft_smoke $SFT $O/smoke_sft.json --limit 120 --systems greedy,mbr16,beam4 --gpu-util 0.5 || exit 1
  touch $O/SMOKE_DONE; say "SMOKE done"; exit 0
fi
for m in sft rl; do
  M=$SFT; [ $m = rl ] && M=$RL
  [ -f $O/$m.json ] || { say "decode $m: greedy, mbr16, beam4 at 768"; dec $m $M $O/$m.json --systems greedy,mbr16,beam4,beam4hf --gpu-util 0.6 || { say "decode $m FAILED"; exit 1; }; }
done
touch $O/GEN_MAIN_DONE; say "main generation done"
[ -n "${DIAG:-}" ] && for m in sft rl; do
  M=$SFT; [ $m = rl ] && M=$RL
  [ -f $O/${m}_oldeos.json ] || dec $m $M $O/${m}_oldeos.json --systems beam4 --eos-mode eos --gpu-util 0.6 || say "diagnostic oldeos $m failed (not blocking)"
done
touch $O/GEN_DONE; say "generation done (GPU free for the next job after the AfriCOMET pass)"
PYTHONPATH=$E/pylibs_none:$E/gemma_mt_tree $E/venv_comet/bin/python $O/../e02_comet.py $(ls $O/sft.json $O/rl.json $O/sft_oldeos.json $O/rl_oldeos.json 2>/dev/null) > $O/comet.log 2>&1
say "AfriCOMET pass done: $(tail -1 $O/comet.log)"; touch $O/COMET_DONE
for m in sft rl sft_oldeos rl_oldeos; do
  [ -f $O/$m.json ] || continue
  cf=$O/$m.comet.json; [ -f $cf ] || cf=none
  PYTHONPATH=$E/pylibs_none:$E/gemma_mt_tree nice -n 10 $E/venv_comet/bin/python $O/../e02_mknpz.py $O/$m.json $cf $O/npz $m >> $O/mknpz.log 2>&1
done
touch $O/NPZ_DONE; say "npz done"
cd $O; S=""; for f in npz/*.npz; do n=$(basename $f .npz); S="$S $n=$f"; done
nice $E/venv_comet/bin/python $O/../ecompare.py --sys $S --out $O/cmp_all --comet-langs-from /dev/shm/evid_e20/mixed.npz --per-language \
  --pair sft_beam4:sft_greedy sft_mbr16:sft_greedy sft_beam4:sft_mbr16 rl_beam4:rl_greedy rl_mbr16:rl_greedy rl_beam4:rl_mbr16 sft_beam4hf:sft_greedy rl_beam4hf:rl_greedy sft_beam4hf:sft_beam4 rl_beam4hf:rl_beam4 rl_greedy:sft_greedy rl_beam4:sft_beam4 rl_mbr16:sft_mbr16 \
         rl_greedy:sft_beam4 rl_greedy:sft_mbr16 rl_beam4:sft_greedy rl_mbr16:sft_greedy > $O/ecompare.log 2>&1
touch $O/E02_ALL_DONE; say "E02_ALL_DONE"
