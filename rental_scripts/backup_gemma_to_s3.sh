#!/bin/bash
# Owner 2026-10-20: maintenance in ~3 h, push all work (models + datasets) to the object-storage bucket, compressed (zstd -10, lossless) with sha256 + byte-count checks.
# Layout: remote:${S3_BUCKET}/gemma_mt/<group>/<name>.tar.zst  (+ .sha256).  Streams, no local temp files.  Order = value / size.
cd /workspace/sft_build
source .s3_env
export RCLONE_CONFIG_REMOTE_S3_UPLOAD_CONCURRENCY=8 RCLONE_CONFIG_REMOTE_S3_CHUNK_SIZE=64M
DEST=remote:${S3_BUCKET}/gemma_mt; LOG=logs/backup_s3.log; STAT=/tmp/backup_stat; mkdir -p $STAT
say() { echo "$(date -u +%FT%TZ) [backup] $*" | tee -a $LOG; }
# push NAME GROUP LEVEL -- tar args...   (stdin of zstd = tar stream)
push() {
  local name=$1 group=$2 level=$3; shift 3
  local key=$DEST/$group/$name.tar.zst
  say "start $group/$name"
  local t0=$(date +%s)
  nice -n 10 tar -C /workspace/sft_build --exclude='._*' -cf - "$@" 2>>$LOG | nice -n 10 zstd -$level -T24 -q -c \
     | tee >(sha256sum | cut -d' ' -f1 > $STAT/$name.sha) >(wc -c > $STAT/$name.bytes) | rclone rcat $key 2>>$LOG
  local rc=${PIPESTATUS[0]}; sleep 2
  local rb=$(rclone size --json $key 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['bytes'])" 2>/dev/null)
  local lb=$(cat $STAT/$name.bytes 2>/dev/null | tr -d ' ')
  if [ -n "$rb" ] && [ "$rb" = "$lb" ]; then
    echo "$(cat $STAT/$name.sha)  $group/$name.tar.zst  $rb bytes" | rclone rcat $key.sha256 2>>$LOG
    say "OK $group/$name: $rb bytes in $(( $(date +%s) - t0 )) s"
  else say "FAILED $group/$name: remote=$rb local=$lb rc=$rc"; fi
}
snapshot_sqlite() {   # consistent copies of the live SQLite archives
  mkdir -p /tmp/arch_snap; for r in rl2 rl2b rl2c; do /workspace/venv_gemma/bin/python - <<PY
import sqlite3,os
src="runs/$r/archive/archive.sqlite"
if os.path.exists(src):
    s=sqlite3.connect("file:"+src+"?mode=ro",uri=True); d=sqlite3.connect("/tmp/arch_snap/archive_$r.sqlite"); s.backup(d); d.close(); s.close()
PY
  done
}
say "begin"
rclone mkdir $DEST 2>/dev/null
# 1 small things: code overlays, scripts, logs, calibration, benchmark sets, eval results
push code_scripts_logs_bench small 10 gemma_mt_2c gemma_mt_2b *.sh mk_*.py logs calibration gemma_calibration bench eval agent_20261017 runs/*/config.json runs/*/summary.json runs/*/metrics.jsonl
# 2 archives (snapshots) + run metadata
snapshot_sqlite
nice tar -C /tmp -cf - arch_snap | zstd -10 -q -c | rclone rcat $DEST/small/rl_archives_snapshot_$(date -u +%Y%m%dT%H%M).tar.zst 2>>$LOG && say "OK archives snapshot"
# 3 live training state of 2c (resume point) and its latest model
push rl2c_resume_state ckpt 3 runs/rl2c/last runs/rl2c/config.json
push rl2c_hub_stage_latest ckpt 6 runs/rl2c/hub_stage_latest
# 4 reference models of the earlier phases (also on the Hub)
push rl_phase1_final ckpt 6 runs/rl_phase1_final
push rl_phase2_final ckpt 6 runs/rl_phase2_final
push rl_2b_stage_b64 ckpt 6 runs/rl_2b_stage_b64
# 5 datasets for the Gemma RL / SFT
push gemma_rl_prompts_v2 data 10 gemma/rl_prompts_v2 gemma/rl_prompts_v2_meta
push gemma_packed_fwd_sft_windows data 10 gemma/packed_fwd
push gemma_rl_prompts_v2_tok data 10 gemma/rl_prompts_v2_tok
# 6 SFT final training state
push sft_unimax_last ckpt 6 runs/sft_unimax/last runs/sft_unimax/config.json runs/sft_unimax/summary.json
rclone lsf -R --format "sp" $DEST > /tmp/gemma_mt_bucket_listing.txt 2>>$LOG; rclone copyto /tmp/gemma_mt_bucket_listing.txt $DEST/MANIFEST_listing.txt
say "ALL DONE (re-run only item 3 before maintenance for the freshest resume state)"
