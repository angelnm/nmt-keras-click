#!/bin/sh
SOURCE=es
TARGET=en
PAIR=es-en
MODEL=update_78750
CORPUS=EU
FOLDER=./../${CORPUS}/
DIV=100
FROM=0.0
TO=1.0
OUTPUT=data.csv
ESTIMATOR=0
python3 nmt_cm.py -ds "${FOLDER}/model/${SOURCE}-${TARGET}/Dataset_${CORPUS}_${SOURCE}${TARGET}.pkl" -src "${FOLDER}/preraw/tst/${PAIR}/test.${SOURCE}" -trg "${FOLDER}/preraw/tst/${PAIR}/test.${TARGET}" -d "./../out.txt" --models "${FOLDER}/model/${SOURCE}-${TARGET}/${MODEL}" --confidence_estimator 2 --lexicon_model "${FOLDER}/model2/${SOURCE}-${TARGET}/m" --alignment_model "./prefix.a2to3" --prefix -wt ${DIV} -cm_from ${FROM} -cm_to ${TO} -cm_output ${OUTPUT} -est ${ESTIMATOR}
