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
import codecs
import six
import numpy as np
import traceback
import sys


def get_data_info(data_info):
    inp_size, tgt_size = 1, 1
    inp_type, tgt_type = np.int32, np.int32
    try:
        in_size_info, tgt_size_info = data_info.split('-')
        if 'int' in in_size_info:
            inp_size = int(in_size_info.replace('(int)', ''))
        elif 'float' in in_size_info:
            inp_size = int(in_size_info.replace('(float)', ''))
            inp_type = np.float32
        else:
            inp_size = int(in_size_info)

        if 'int' in tgt_size_info:
            tgt_size = int(tgt_size_info.replace('(int)', ''))
        elif 'float' in tgt_size_info:
            tgt_size = int(tgt_size_info.replace('(float)', ''))
            tgt_type = np.float32
        else:
            tgt_size = int(tgt_size_info)
    except:
        traceback.print_exc(file=sys.stderr)
        print('Make sure you have correct fname format. Correct format is fname:in_size(type)-target_size(int)')
        exit(1)

    return inp_size, tgt_size, inp_type, tgt_type


def load_data(fname_input):
    inp_size, tgt_size = 1, 1
    inp_type, tgt_type = np.int32, np.int32
    if ':' in fname_input:
        fname_input, data_info = fname_input.split(':')
        inp_size, tgt_size, inp_type, tgt_type = get_data_info(data_info)

    data = np.loadtxt(fname_input, dtype=np.float32)
    if inp_size == 1 and inp_type == np.int32:
        tmp_inp = np.asarray(data[:, 0], dtype=inp_type)
    else:
        tmp_inp = np.asarray(data[:, :inp_size], dtype=inp_type)

    if tgt_size == 1 and tgt_type == np.int32:
        tmp_tgt = np.asarray(data[:, 1], dtype=tgt_type)
    else:
        tmp_tgt = np.asarray(data[:, inp_size:inp_size + tgt_size], dtype=tgt_type)

    inp, tgt = [], []

    start, end = 0, 0
    for line in codecs.open(fname_input, 'r', 'utf-8'):
        line = line.strip()
        if not line:
            inp_row = tmp_inp[start:end]
            tgt_row = tmp_tgt[start:end]
            inp.append(inp_row)
            tgt.append(tgt_row)
            start = end
        else:
            end += 1

    if not inp:  # In case there is no empty line in the data file, treat it as one_row data
        inp = [tmp_inp]
        tgt = [tmp_tgt]

    return {'input': inp, 'target': tgt}


def data_one_row_spliter(dataset, batchsize, n_epoch):
    whole_len = len(dataset['input'][0])
    jump = whole_len // batchsize

    epoch = 0
    iter_idx = 0
    for i in six.moves.range(jump * n_epoch):
        iter_idx += 1
        x_batch = [dataset['input'][0][(jump * j + i) % whole_len]
                   for j in six.moves.range(batchsize)]
        y_batch = [dataset['target'][0][(jump * j + i) % whole_len]
                   for j in six.moves.range(batchsize)]
        yield iter_idx, x_batch, y_batch, epoch, 100 * iter_idx / jump, False
        if (i + 1) % jump == 0:
            # Notify the trainer that we done one epoch
            yield None, None, None, None, None, None
            epoch += 1
            iter_idx = 0


def is_all_elem_equal_size(x, y):
    for a, b in zip(x, y):
        if len(a) != len(b):
            return False
    return True


def pad_eos(x_batch, y_batch, EOS):
    """Pad end-of-sentence character to make all elements have the same length"""
    len_range = [len(x) for x in x_batch]
    len_range.sort(reverse=True)
    max_len = len_range[0]
    for i in xrange(len(x_batch)):
        num = max_len - len(x_batch[i])
        if num:
            if isinstance(EOS[0], int):
                inp_pad_array = np.ndarray((num, ), dtype=np.int32)
                inp_pad_array.fill(EOS[0])
                x_batch[i] = np.append(x_batch[i], inp_pad_array)
            else:
                inp_pad_array = np.tile(EOS[0], num)
                inp_pad_array.shape = (num, EOS[0].size)
                x_batch[i] = np.append(x_batch[i], inp_pad_array, axis=0)

            if isinstance(EOS[1], int):
                tgt_pad_array = np.ndarray((num, ), dtype=np.int32)
                tgt_pad_array.fill(EOS[1])
                y_batch[i] = np.append(y_batch[i], tgt_pad_array)
            else:
                tgt_pad_array = np.tile(EOS[1], num)
                tgt_pad_array.shape = (num, EOS[1].size)
                y_batch[i] = np.append(y_batch[i], tgt_pad_array, axis=0)


def data_multi_row_spliter(dataset, batchsize, n_epoch, EOS=None):
    import copy

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in xrange(0, len(l), n):
            yield l[i:i + n]

    sortedRes = sorted(zip(dataset['input'], dataset['target']), key=lambda x: x[0].size)
    x_data = [x[0] for x in sortedRes]
    y_data = [x[1] for x in sortedRes]

    for epoch in xrange(n_epoch):
        iter_per_epoch = len(x_data) / batchsize
        iter_idx = 0
        subiter_idx = 0
        for n_row_of_x, n_row_of_y in \
                    zip(chunks(x_data, batchsize), chunks(y_data, batchsize)):
            iter_idx += 1
            done_percentage = int(100 * float(iter_idx) / iter_per_epoch)
            # padding some data to make the x_batch_row has the same size with batchsize
            # this often happens at the last chunk
            num = batchsize - len(n_row_of_x)
            for x in xrange(num):
                n_row_of_x.append(copy.copy(n_row_of_x[0]))
                n_row_of_y.append(copy.copy(n_row_of_y[0]))

            # Make sure all utterance has the same dataset
            pad_eos(n_row_of_x, n_row_of_y, EOS=EOS)
            n_row_of_x = np.asarray(n_row_of_x)
            n_row_of_y = np.asarray(n_row_of_y)

            utterance_len = n_row_of_x.shape[1]
            for i in xrange(utterance_len):
                subiter_idx += 1
                x_batch = n_row_of_x[:, i]
                y_batch = n_row_of_y[:, i]
                if i == utterance_len - 1:
                    yield subiter_idx, x_batch, y_batch, epoch, done_percentage, True
                else:
                    yield subiter_idx, x_batch, y_batch, epoch, done_percentage, False
        yield None, None, None, None, None, None


def data_spliter(dataset, batchsize=1, n_epoch=1, EOS=None):
    if len(dataset['input']) == 1:
        return data_one_row_spliter(dataset, batchsize, n_epoch)
    else:
        return data_multi_row_spliter(dataset, batchsize, n_epoch, EOS=EOS)
