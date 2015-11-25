#! /bin/bash
#
# run.sh
# Copyright (C) 2015 truong-d <truong-d@ahcsegpu01>
#
# Distributed under terms of the MIT license.
#

train_data="data/train.txt:4(float)-1(int)"
valid_data="data/valid.txt:4(float)-1(int)"
test_data="data/test.txt:4(float)-1(int)"

python ../../train_dnn.py --nnet-struct="linear(4-3)" \
    --train=$train_data --dev=$valid_data --test=$test_data \
    --lr="0.25" --lr-decay="0.5" --lr-stop="0.000001" --start-decay="0.0001" \
    --grad-clip="5" \
    --random-seed="0" --n-epoch="10" --batch-size="2" --bprop-len="1" \
    --forget-on-new-utt \
    --loss-func='cross_entropy' \
    --eos-pad='0.580785529155,0.264448166671,0.455260213493,0.243075299513;1' \
    --save-model="$dir"
