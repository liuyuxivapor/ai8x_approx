#!/bin/sh
python ./train.py \
--epochs 300 \
--batch-size 256 \
--gpus 0 \
--deterministic \
--optimizer Adam \
--lr 0.0001 \
--compress policies/schedule-biquad.yaml \
--model ai85net_biquad \
--dataset biquad \
--regression \
--param-hist \
--device MAX78000 "$@"