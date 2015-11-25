#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 truong-d <truong-d@ahclab08>
#
# Distributed under terms of the MIT license.

"""

"""
from __future__ import print_function
from nnet_model import NNET_Model
import data_util
from chainer import cuda
import chainer
from misc import error, info, get_opts
import chainer.links as L
from chainer import Variable
import numpy as np
import codecs
import sys
import os
import StringIO

opts, args = get_opts()
xp = np

lang_dir = args[0]
decode_file = args[1]


def load_dict(fname):
    w_dic = {}
    for line in codecs.open(fname, 'r', 'utf-8'):
        s, idx =  line.strip().split()
        w_dic[s] = idx
    return w_dic

w_dic = load_dict(os.path.join(lang_dir, 'words.int'))


def load_data(fname):
    data = []
    text = []
    for line in codecs.open(fname, 'r', 'utf-8'):
        line = line.strip()
        row_input = []
        for w in line.split():
            if not w in w_dic:
                w = "<unk>"
            row_input.append(int(w_dic[w]))
        data.append(row_input)
        text.append(line.split())
    return data, text


data, text_data = load_data(decode_file)


def nice_print(output, words):
    text = []
    for r, w in zip(output, words):
        if r in [0, 2]:
            text.append(w)
        elif r == 1:
            text.append('_%s' % w)
    print(' '.join(text).replace(' _', '_'))


def decode_mode():
    nnet_model =  NNET_Model.load(opts.fname_in_model)
    for row, words in zip(data, text_data):
        row_result = []
        for w in row:
            x = Variable(xp.asarray([w], dtype=np.int32))
            row_result.append(nnet_model(x).data.argmax(axis=1)[0])
        if opts.forget_on_new_utt:
            nnet_model.forget_history()
        
        nice_print(row_result, words)


def main():
    decode_mode()

if __name__ == '__main__':
    main()
