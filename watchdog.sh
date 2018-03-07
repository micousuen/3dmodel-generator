#!/bin/bash

PYTHON35_CONDA=/home/micou/.conda/envs/python35/bin/python3.5

while :
do
	if ps ax | grep -v grep | grep "$PYTHON35_CONDA dataIO.py" > /dev/null
	then
		echo "*************************** Program running *******************"
		sleep 180
	else
		echo "Restart program"
		swapoff -a
		$PYTHON35_CONDA dataIO.py -n 2 &
		sleep 30
	fi
done
