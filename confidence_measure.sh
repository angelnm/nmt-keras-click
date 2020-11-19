#!/bin/sh -v
SOURCE=es
TARGET=en
PAIR=es-en
MODEL=update_78750
CORPUS=EU
FOLDER=./../${CORPUS}/
DIV=$1
FROM=$2
TO=$3
ESTIMATOR=$4
DIFF=$5
ALPHA=$6
OUTPUT=./results_cm/${SOURCE}_${TARGET}_${ESTIMATOR}/data_${DIFF}_${ALPHA}.csv

case ${ESTIMATOR} in
   [0])
        LEXICON="${FOLDER}/model2/${SOURCE}-${TARGET}2/IBM1/m"
        ALIGNMT="_"
   ;;
   [1])
        LEXICON="${FOLDER}/model2/${SOURCE}-${TARGET}2/IBM2/m"
        ALIGNMT="${FOLDER}/model2/${SOURCE}-${TARGET}2/IBM2/prefix.a2to3"
   ;;
   [2])
        LEXICON="${FOLDER}/model2/${SOURCE}-${TARGET}2/FAST/m"
        ALIGNMT="_"
   ;;
   [3])
        LEXICON="${FOLDER}/model2/${SOURCE}-${TARGET}2/HMM/m"
	ALIGNMT="${FOLDER}/model2/${SOURCE}-${TARGET}2/HMM/prefix.hhmm.5"
   ;;
esac
mkdir -p results_cm
mkdir -p results_cm/${SOURCE}_${TARGET}_${ESTIMATOR}
python3 nmt_cm2.py       -ds "${FOLDER}/model/${SOURCE}-${TARGET}/Dataset_${CORPUS}_${SOURCE}${TARGET}.pkl" \
                                        -src "${FOLDER}/preraw/tst/${PAIR}/test.${SOURCE}" -trg "${FOLDER}/preraw/tst/${PAIR}/test.${TARGET}" \
                                        -d "./../out.txt" \
                                        --models "${FOLDER}/model/${SOURCE}-${TARGET}/${MODEL}" \
                                        -est ${ESTIMATOR} \
                                        --lexicon_model ${LEXICON} \
                                        --alignment_model ${ALIGNMT} \
                                        --prefix \
                                        -wt ${DIV} \
                                        -cm_from ${FROM} \
                                        -cm_to ${TO} \
                                        -cm_output ${OUTPUT} \
                                        -cm_alpha ${ALPHA}
