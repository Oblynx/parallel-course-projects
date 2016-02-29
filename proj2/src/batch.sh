#!/bin/bash

mpirun.mpich -n 1 ./kNN 1 21 17 9
for psize in $(seq 22 25); do
  mpirun.mpich -n 1 ./kNN 1 $psize 18 9
done

