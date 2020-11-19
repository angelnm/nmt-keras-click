#!/bin/sh

TO=$1
FROM=$2
DIV=$3
DIFF=$4
ALPHA=$5

echo ${DIV} ${FROM} ${TO} ${ESTIMATOR} ${DIFF} ${ALPHA}
ESTIMATOR=0
./confidence_measure.sh ${DIV} ${FROM} ${TO} ${ESTIMATOR} ${DIFF} ${ALPHA}
ESTIMATOR=1
./confidence_measure.sh ${DIV} ${FROM} ${TO} ${ESTIMATOR} ${DIFF} ${ALPHA}
ESTIMATOR=2
./confidence_measure.sh ${DIV} ${FROM} ${TO} ${ESTIMATOR} ${DIFF} ${ALPHA}
ESTIMATOR=3
./confidence_measure.sh ${DIV} ${FROM} ${TO} ${ESTIMATOR} ${DIFF} ${ALPHA}
