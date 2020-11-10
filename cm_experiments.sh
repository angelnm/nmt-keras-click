#!/bin/sh

DIV=20
FROM=0.0
TO=0.00000000001

ESTIMATOR=0
./confidence_measure.sh ${DIV} ${FROM} ${TO} ${ESTIMATOR}
ESTIMATOR=1
./confidence_measure.sh ${DIV} ${FROM} ${TO} ${ESTIMATOR}
ESTIMATOR=2
./confidence_measure.sh ${DIV} ${FROM} ${TO} ${ESTIMATOR}
ESTIMATOR=3
./confidence_measure.sh ${DIV} ${FROM} ${TO} ${ESTIMATOR}