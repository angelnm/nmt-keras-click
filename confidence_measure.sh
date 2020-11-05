#!/bin/sh
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
OUTPUT=data.csv

case ${ESTIMATOR} in
   [0]) 
	LEXICON="${FOLDER}/model2/${SOURCE}-${TARGET}/m"
	ALIGNMT="${FOLDER}/model2/${SOURCE}-${TARGET}/prefix.a2to3"
   ;;
   [1]) 
	LEXICON="${FOLDER}/model2/${SOURCE}-${TARGET}/m"
	ALIGNMT="${FOLDER}/model2/${SOURCE}-${TARGET}/prefix.a2to3"
   ;;
   [2]) 
	LEXICON="${FOLDER}/model2/${SOURCE}-${TARGET}/m"
	ALIGNMT="_"
   ;;
esac

python3 nmt_cm.py 	-ds "${FOLDER}/model/${SOURCE}-${TARGET}/Dataset_${CORPUS}_${SOURCE}${TARGET}.pkl" \
					-src "${FOLDER}/preraw/tst/${PAIR}/test.${SOURCE}" -trg "${FOLDER}/preraw/tst/${PAIR}/test.${TARGET}" \
					-d "./../out.txt" \
					--models "${FOLDER}/model/${SOURCE}-${TARGET}/${MODEL}" \
					--confidence_estimator 2 \
					--lexicon_model ${LEXICON} \
					--alignment_model ${ALIGNMT} \
					--prefix \
					-wt ${DIV} \
					-cm_from ${FROM} \
					-cm_to ${TO} \
					-cm_output ${OUTPUT} \
					-est ${ESTIMATOR} 
