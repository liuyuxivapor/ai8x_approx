#!/bin/sh
DEVICE="MAX78000"
TARGET="demos"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix ai85net_dct4 --checkpoint-file trained/nn-dct4-qat8-q.pth.tar --config-file networks/ai85-bayer2rgb.yaml $COMMON_ARGS "$@"
