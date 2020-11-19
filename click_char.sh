#!/bin/sh
SOURCE=de
TARGET=en
PAIR=de-en
MODEL=update_86250
FOLDER=./../xerox/
CLICK=2
python3 click_char.py -ds "${FOLDER}/model/${PAIR}/Dataset_xerox_${SOURCE}${TARGET}.pkl" -src "${FOLDER}/preraw/tst/${PAIR}/test.${SOURCE}" -trg "${FOLDER}/preraw/tst/${PAIR}/test.${TARGET}" -d "./../out.txt" --models "${FOLDER}/model/${SOURCE}-${TARGET}/${MODEL}" --prefix -ma ${CLICK}