#!/bin/sh
python ./train.py --model ai85net_fir --dataset fir --regression --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-fir-qat8-q.pth.tar -8 --device MAX78000 "$@"