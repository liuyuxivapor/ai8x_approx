#!/bin/sh
python ./train.py \
--epochs 300 \
--batch-size 32 \
--gpus 0 \
--deterministic \
--optimizer Adam \
--lr 0.001 \
--compress policies/schedule-fir.yaml \
--model ai85net_fir \
--dataset fir \
--regression \
--param-hist \
--device MAX78000 "$@"