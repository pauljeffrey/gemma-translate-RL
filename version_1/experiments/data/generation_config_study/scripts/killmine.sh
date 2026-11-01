#!/bin/bash
# usage: killmine.sh pattern -- kills only processes whose cwd is under /workspace/jobs/r12 and whose command line matches
for p in $(ps -eo pid,args | awk -v pat="$1" '$0 ~ pat && $0 !~ /awk/ && $0 !~ /killmine/ && $0 !~ /bash -c/ {print $1}'); do
  c=$(readlink /proc/$p/cwd 2>/dev/null); case "$c" in /workspace/jobs/r12*) kill $p;; esac
done
