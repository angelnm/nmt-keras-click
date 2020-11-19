#!/bin/sh
SOURCE=es
TARGET=en
PAIR=es-en
MODEL=update_78750
CORPUS=EU
FOLDER=./../${CORPUS}/
WT=1.0
ST=0.0000001
python3 interactive_nmt_cm.py -ds "${FOLDER}/model/${SOURCE}-${TARGET}/Dataset_${CORPUS}_${SOURCE}${TARGET}.pkl" -src "${FOLDER}/preraw/tst/${PAIR}/test.${SOURCE}" -trg "${FOLDER}/preraw/tst/${PAIR}/test.${TARGET}" -d "./../out.txt" --models "${FOLDER}/model/${SOURCE}-${TARGET}/${MODEL}" --confidence_model "${FOLDER}/model2/${SOURCE}-${TARGET}/m" --alignment_model "./prefix.a2to3" --prefix -st ${ST} -wt ${WT} 
