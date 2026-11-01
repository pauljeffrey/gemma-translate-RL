cd /workspace/jobs/r15/probe
export PLAN_JSON='[["kea kri",["fs8t07","zs","fs3","fs8note"]]]'
setsid nohup ./waitrun.sh gen_gemma2.py gen_calib > /dev/null 2>&1 < /dev/null &
disown
echo launched
