#!/bin/sh
python3 click_char.py -ds ./datasets/Dataset_xerox_enfr.pkl -src ./../xerox/preraw/test.en -trg ./../xerox/preraw/test.fr -d ./../out.txt --models ./trained_models/best_xerox_512_50/update_37500 --prefix -ma $1
