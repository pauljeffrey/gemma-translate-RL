cd /workspace/jobs/r15/probe
export PLAN_JSON='[["ibb urh nbl shi ach vmw seh nyn her yao sid ndc dua bas wes dag tsc ish bin bbj oke",["zs"]],["ish bin bbj oke tsc dag",["fs8note","fs3"]],["nbl shi her ndc",["fs8note"]]]'
setsid nohup ./waitrun.sh gen_gemma2.py gen_abl > /dev/null 2>&1 < /dev/null &
disown
echo launched
