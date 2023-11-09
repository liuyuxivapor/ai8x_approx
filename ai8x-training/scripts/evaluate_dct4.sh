#!/bin/sh
python ./train.py --model ai85net_dct4 --dataset dct4 --regression --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/nn-dct4-qat8-q.pth.tar -8 --device MAX78000 "$@"