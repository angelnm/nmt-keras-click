#!/bin/sh
alphas="0.25 0.50 0.75"
for a in $alphas; do
	./cm_experiments.sh 1.0 0.0 100 0 $a
done
#./cm_experiments.sh 0.01 0.0 20 1
#./cm_experiments.sh 0.0001 0.0 20 2
#./cm_experiments.sh 0.000001 0.0 20 3
#./cm_experiments.sh 0.00000001 0.0 20 4
#./cm_experiments.sh 0.0000000001 0.0 20 5
