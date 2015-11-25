#! /bin/bash
#
# run.sh
# Copyright (C) 2015 truong-d <truong-d@ahcsegpu01>
#
# Distributed under terms of the MIT license.
#

train_data=data/k-folds/train1.iob2
valid_data=data/k-folds/valid1.iob2
test_data=data/k-folds/test1.iob2

stage=1
if [[ $stage -le 0 ]]; then
  echo "Preparing data"
  lang=exp/lang
  dir=exp/lang
  mkdir -p $dir
  echo "...create dictionary"
  python utils/build_vocab.py $train_data $dir || exit 1
  echo "...convert data"
  dir=exp/data
  mkdir -p $dir
  python utils/convert.py $lang $train_data $dir/train.txt || exit 1
  python utils/convert.py $lang $valid_data $dir/valid.txt || exit 1
  python utils/convert.py $lang $test_data $dir/test.txt || exit 1
  #sed -i '/^\s*$/d' exp/data/train.txt
fi

lang=exp/lang
if [[ $stage -le 1 ]]; then
  echo "Start training"
  vocab_size=`wc -l $lang/words.int | cut -f 1 -d" "`
  data=exp/data
  dir=exp/model
  mkdir -p $dir
  set -x
  python ../../train_dnn.py --nnet-struct="embed($vocab_size-124):lstm(124-124):linear(124-3)" \
    --train=$data/train.txt --dev=$data/valid.txt --test=$data/test.txt \
    --lr="0.25" --lr-decay="0.5" --lr-stop="0.000001" --start-decay="0.0001" \
    --grad-clip="5" \
    --random-seed="0" --n-epoch="3" --batch-size="8" --bprop-len="10" \
    --forget-on-new-utt \
    --eos-pad='7057;2'
    --save-model="$dir" || exit 1

  #python segmenter.py --model-in="$dir" $lang $decode_data
fi
