cd /workspace/jobs/r15/probe
export ISOS="kea kri" K=8 BATCH=240 MINUTES=${1:-150} KV_GIB=8.0 UTIL=0.29 SEQS=10
setsid nohup ./waitrun.sh gen_synth.py gen_synthB > /dev/null 2>&1 < /dev/null &
disown
echo launched
