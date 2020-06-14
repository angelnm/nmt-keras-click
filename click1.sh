#!/bin/sh
folder=$1
source=$2
target=$3
models=$4
number=$5
python3 click.py -ds ./../xerox/preraw/trn/$folder/mapping.$source_$target.pkl -src ./../xerox/preraw/tst/$folder/test.$source -trg ./../xerox/preraw/tst/$folder/test.$target -d ./../xerox/results/$source_$target.txt --models ./../xerox/model/$source-$target/$models --prefix -ma $number
