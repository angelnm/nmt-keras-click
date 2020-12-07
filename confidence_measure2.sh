#!/bin/sh
ESTIMATOR=$1
ALPHA=$2
WT=$3

SOURCE=es
TARGET=en
PAIR=es-en
MODEL=update_78750
CORPUS=EU
FOLDER=./../${CORPUS}/
ST=1.1
OUTPUT=./results_cm/${SOURCE}_${TARGET}_${ESTIMATOR}/data_${WT}.csv

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
python3 interactive_nmt_cm.py \
		-ds "${FOLDER}/model/${SOURCE}-${TARGET}/Dataset_${CORPUS}_${SOURCE}${TARGET}.pkl" \
		-src "${FOLDER}/preraw/tst/${PAIR}/test.${SOURCE}" \
		-trg "${FOLDER}/preraw/tst/${PAIR}/test.${TARGET}" \
		-d "./../out.txt" \
		--models "${FOLDER}/model/${SOURCE}-${TARGET}/${MODEL}" \
		-est ${ESTIMATOR} \
		--lexicon_model ${LEXICON} \
		--alignment_model ${ALIGNMT} \
		-cm_output ${OUTPUT} \
		-cm_alpha ${ALPHA} \
		--prefix \
		-st ${ST} \
		-wt ${WT} 
