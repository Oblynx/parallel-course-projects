#!/bin/bash

for psize in $(seq 21 25); do
  for mesh in $(seq 16 -1 12); do
    #mpirun.mpich -n 4 ./kNN 3 $psize $mesh 3
    ./kNN 3 $psize $mesh 3
  done
done

