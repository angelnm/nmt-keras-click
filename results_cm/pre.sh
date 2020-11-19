#!/bin/sh
FOLDER=$1
python3 util_csv.py ./es_en_${FOLDER}/data_0.csv ./es_en_${FOLDER}/data_1.csv ./es_en_${FOLDER}/data_2.csv ./es_en_${FOLDER}/data_3.csv ./es_en_${FOLDER}/data_4.csv ./es_en_${FOLDER}/data_5.csv
mv data.csv data_${FOLDER}.csv
