#!/bin/sh
python quantize.py trained/ai87-kws20_v2-qat8.pth.tar trained/ai87-kws20_v2-qat8-q.pth.tar --device MAX78000 -v "$@"
