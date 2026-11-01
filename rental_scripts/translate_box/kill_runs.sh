#!/bin/bash
for p in $(ps -eo pid,args | grep -a '[l]ongdoc_translate.py' | awk '{print $1}'); do kill $p; done
for i in $(seq 1 60); do n=$(ps -eo args | grep -a -c '[l]ongdoc_translate.py'); [ $n = 0 ] && break; sleep 2; done
echo remaining runner processes: $(ps -eo args | grep -a -c '[l]ongdoc_translate.py')
