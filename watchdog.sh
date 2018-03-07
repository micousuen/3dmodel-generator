#!/bin/bash

PYTHON35_CONDA=/home/micou/.conda/envs/python35/bin/python3.5
core_num=4

while :
do
	if ps ax | grep -v grep | grep "$PYTHON35_CONDA dataIO.py" > /dev/null
	then
		echo "*************************** Program running *******************"
		sleep 180
	else
		echo "Restart program"
		swapoff -a
		core_num=$(($core_num-1))
		$PYTHON35_CONDA dataIO.py -n $core_num &
		sleep 30
	fi
done
