#!/bin/sh
python ./train.py \
--epochs 300 \
--batch-size 256 \
--gpus 0 \
--deterministic \
--optimizer Adam \
--lr 0.001 \
--compress policies/schedule-fft.yaml \
--model ai85net_fft \
--dataset fft \
--regression \
--param-hist \
--device MAX78000 "$@"