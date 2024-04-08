#!/bin/sh
DEVICE="MAX78000"
TARGET="demos"
COMMON_ARGS="--device $DEVICE --timer 0 --compact-data --verbose"
OVER_WRITE="--overwrite"
# SOFTMAX="--softmax"

python ai8xize.py --test-dir $TARGET --prefix ai85net_dct4 --checkpoint-file trained/ai85-dct4-qat8-q.pth.tar --config-file networks/ai85-dct4.yaml --mlator $COMMON_ARGS $OVER_WRITE $SOFTMAX "$@"
