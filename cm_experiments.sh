#!/bin/sh

TO=$1
FROM=$2
DIV=$3
DIFF=$4

ESTIMATOR=3
./confidence_measure.sh ${DIV} ${FROM} ${TO} ${ESTIMATOR} ${DIFF}
ESTIMATOR=2
./confidence_measure.sh ${DIV} ${FROM} ${TO} ${ESTIMATOR} ${DIFF}
ESTIMATOR=1
./confidence_measure.sh ${DIV} ${FROM} ${TO} ${ESTIMATOR} ${DIFF}
ESTIMATOR=0
./confidence_measure.sh ${DIV} ${FROM} ${TO} ${ESTIMATOR} ${DIFF}