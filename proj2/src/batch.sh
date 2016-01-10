#!/bin/bash

for psize in $(seq 21 25); do
  mpirun.mpich -n 4 ./kNN 1 $psize $(($psize-4)) 8
done

