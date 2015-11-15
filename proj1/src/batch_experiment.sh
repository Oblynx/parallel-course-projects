#!/bin/bash
mkdir ../logs
touch ../logs/sample-bitonic.log
touch ../logs/omp.log
touch ../logs/pthread.log
touch ../logs/qsort.log

for experiment in $(seq 1 3); do
  for size in $(seq 16 27); do
    for threads in $(seq 1 8); do
      # Collect data
      # out: <imperative time> <recursive time>
      sb=../build/sample-bitonic size
      if [ "$?" -ne "0" ]; then
        echo "Error executing program"
        exit $?
      fi
      # out: <constuct time> <sort time>
      omp=../build/pbitonic-omp size threads
      if [ "$?" -ne "0" ]; then
        echo "Error executing program"
        exit $?
      fi
      # out: <constuct time> <sort time>
      pthr=../build/pbitonic-stdthread size threads
      if [ "$?" -ne "0" ]; then
        echo "Error executing program"
        exit $?
      fi
      # out: <constuct time> <sort time>
      qsort=../build/pbitonic-stdthread size threads -seq
      if [ "$?" -ne "0" ]; then
        echo "Error executing program"
        exit $?
      fi
      # Save to files
      echo $size';'$sb >> ../logs/sample-bitonic.log
      echo $size';'$threads';'$omp >> ../logs/omp.log
      echo $size';'$threads';'$pthr >> ../logs/pthread.log
      echo $size';'$threads';'$qsort >> ../logs/qsort.log
    done
  done
  
  echo '------------' >> ../logs/sample-bitonic.log
  echo '------------' >> ../logs/omp.log
  echo '------------' >> ../logs/pthread.log
  echo '------------' >> ../logs/qsort.log
done
