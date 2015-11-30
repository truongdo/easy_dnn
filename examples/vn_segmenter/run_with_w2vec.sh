#! /bin/bash
#
# run.sh
# Copyright (C) 2015 truong-d
#
# Distributed under terms of the MIT license.
#

w2vec_model=/project/nakamura-lab01/Work/truong-dq/chainer/vidnn/exp/word2vec_truecase_200/vectors.bin

stage=0
lang=exp/lang
if [[ $stage -le 0 ]]; then
  mkdir -p $lang
  echo "...create dictionary"
  python utils/build_vocab_from_w2vec.py data/k-folds/train1.iob2 $w2vec_model $lang || exit 1
  dir=exp/data
  mkdir -p $dir
  for i in 1 2 3 4 5;
  do
    echo "Running on fold $i"
    train_data=data/k-folds/train${i}.iob2 
    valid_data=data/k-folds/valid${i}.iob2 
    test_data=data/k-folds/test${i}.iob2 
  
      echo "...convert data"
      python utils/convert.py $lang $train_data $dir/train${i}.txt || exit 1
      python utils/convert.py $lang $valid_data $dir/valid${i}.txt || exit 1
      python utils/convert.py $lang $test_data $dir/test${i}.txt || exit 1
      sed '/^\s*$/d' $dir/train${i}.txt > $dir/train${i}.txt.nospace
      sed '/^\s*$/d' $dir/valid${i}.txt > $dir/valid${i}.txt.nospace
      sed '/^\s*$/d' $dir/test${i}.txt > $dir/test${i}.txt.nospace
  done
fi

if [[ $stage -le 1 ]]; then
  for i in 1 2 3 4 5;
  do
      echo "Start training"
      data=exp/data
      dir=exp/model_fold-${i}
      vocab_len=`wc -l $lang/words.int | cut -f 1 -d" "`
      mkdir -p $dir
      set -x
      python ../../train_dnn.py --nnet-struct="w2vec($w2vec_model):linear(200-512):lstm(512-256):linear(256-3)" \
        --gpu=0 --train=$data/train${i}.txt.nospace \
        --dev=$data/valid${i}.txt \
        --test=$data/test${i}.txt \
        --lr="0.25" --lr-decay="0.5" --lr-stop="0.01" --start-decay="0.002" \
        --grad-clip="5" \
        --random-seed="0" --n-epoch="200" --batch-size="8" --bprop-len="10" \
        --save-model="$dir" || exit 1
  done
fi
