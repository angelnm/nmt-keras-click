#!/bin/sh
python3 click.py -ds ./datasets/Dataset_xerox_enfr.pkl -src ./../xerox/preraw/test.raw.en -trg ./../xerox/preraw/test.raw.fr -d ./../out.txt --models ./trained_models/best_xerox_512_50/update_37500 --prefix -ma $1
