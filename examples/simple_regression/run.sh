#! /bin/bash
#
# run.sh
# Copyright (C) 2015 truong-d <truong-d@ahcsegpu01>
#
# Distributed under terms of the MIT license.
#

train_data="data/train.txt:4(float)-4(float)"
valid_data="data/valid.txt:4(float)-4(float)"
test_data="data/test.txt:4(float)-4(float)"

python ../../train_dnn.py --nnet-struct="linear(4-4)" \
    --train=$train_data --dev=$valid_data --test=$test_data \
    --lr="0.25" --lr-decay="0.5" --lr-stop="0.000001" --start-decay="0.0001" \
    --grad-clip="5" \
    --random-seed="0" --n-epoch="10" --batch-size="8" --bprop-len="10" \
    --forget-on-new-utt \
    --loss-func='mean_squared_error' \
    --save-model="$dir"
