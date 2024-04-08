#!/bin/sh
DEVICE="MAX78000"
TARGET="demos"
COMMON_ARGS="--device $DEVICE --timer 0 --compact-data --verbose"
OVER_WRITE="--overwrite"
# SOFTMAX="--softmax"

python ai8xize.py --test-dir $TARGET --prefix ai85net_biquad --checkpoint-file trained/ai85-biquad-qat8-q.pth.tar --config-file networks/ai85-biquad.yaml $COMMON_ARGS $OVER_WRITE $SOFTMAX "$@"
