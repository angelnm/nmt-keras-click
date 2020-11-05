#!/bin/sh

DIV=100
FROM=0.0
TO=1.0

ESTIMATOR=0
./confidence_measure.sh ${DIV} ${FROM} ${TO} ${ESTIMATOR}
ESTIMATOR=1
./confidence_measure.sh ${DIV} ${FROM} ${TO} ${ESTIMATOR}