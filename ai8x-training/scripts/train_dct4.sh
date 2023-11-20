#!/bin/sh
QAT_POLICY="policies/qat_policy_dct4.yaml"

# python ./train.py --epochs 200 --batch-size 16 --gpus 0 --deterministic --optimizer Adam --lr 0.01 --compress policies/schedule-dct4.yaml --model ai85net_dct4 --dataset dct4 --regression --param-hist --qat-policy $QAT_POLICY --device MAX78000 "$@"
python ./train.py --epochs 200 --batch-size 16 --gpus 0 --deterministic --optimizer Adam --lr 0.01 --compress policies/schedule-dct4.yaml --model ai85net_dct4 --dataset dct4 --regression --param-hist --device MAX78000 "$@"
# python ./train.py --epochs 0 --batch-size 8 --gpus 0 --deterministic --optimizer Adam --lr 0.01 --model ai85net_dct4 --dataset dct4 --regression --param-hist --device MAX78000 "$@"