cd /workspace/jobs/r15/probe
export ISOS="kea kri" K=8 BATCH=240 MINUTES=${1:-60}
setsid nohup ./waitrun.sh gen_synth.py gen_synth > /dev/null 2>&1 < /dev/null &
disown
echo launched
