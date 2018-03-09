#!/bin/bash

PYTHON35_CONDA=/home/micou/.conda/envs/python35/bin/python3.5
file_to_watch=dataIO.py
core_num=1
mem_threshold=40

while :
do
	if ps ax | grep -v grep | grep "$PYTHON35_CONDA $file_to_watch" > /dev/null
	then
		ps v|grep $file_to_watch | grep -v grep | while read line; do echo $line | awk '{print $9}' | grep -v "%MEM" | while read mem; do if [ ${mem%.*} -ge $mem_threshold ]; then echo $line; fi; done; done | awk '{print $1}' | while read pid; do kill $pid; echo "************ killed $pid *****************";  done
		sleep 3
	else
		echo "Restart program"
		swapoff -a
		$PYTHON35_CONDA $file_to_watch -n $core_num &
		sleep 30
	fi
done
