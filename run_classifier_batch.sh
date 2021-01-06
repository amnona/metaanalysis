#!/bin/bash

# run a batch of $1 classifier processes in background
# all other parameters are passed to classifier.py
echo "starting $1 processes"
num_proc=$1
shift
for ((i=1;i<=num_proc;i++))
do
    ./classifier.py $@ &
done
