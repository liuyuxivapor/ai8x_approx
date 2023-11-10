#!/bin/sh
DEVICE="MAX78000"
TARGET="demos/dct4"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix ai85net_dct4 --checkpoint-file trained/ai85-dct4-qat8-q.pth.tar --config-file networks/ai85-dct4.yaml $COMMON_ARGS "$@"
