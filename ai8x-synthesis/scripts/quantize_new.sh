#!/bin/sh
python quantize.py trained/nn-dct4-unquantized.pth.tar trained/nn-dct4-qat8-q.pth.tar --device MAX78000 -v "$@"
