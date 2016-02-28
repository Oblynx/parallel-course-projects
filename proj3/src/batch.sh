#!/bin/bash
# run experiments to measure execution time

logfile='../log/log2'
for psize in $(seq 7 13); do
  for p in 0.01 0.33 0.45 0.66 0.99; do
    echo -e psize=$psize' \t 'p=$p
    ./makeGraph $psize $p in1
    ./fw -i in1 -l $logfile 
    echo -e '\n'
  done
done
