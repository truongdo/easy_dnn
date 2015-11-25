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
from easy_dnn.nnet_model import NNET_Model
from chainer import computational_graph as c
import easy_dnn.data_util as data_util
from chainer import cuda
import chainer
import easy_dnn.misc as misc
import chainer.links as L
import chainer.functions as F
from chainer import Variable
import numpy as np
import sys

opts, args = misc.get_opts()
xp = cuda.cupy if opts.gpu >= 0 else np

if opts.random_seed is not None:
    np.random.seed(opts.random_seed)


def setup_training(nnet_model, opts):
    from chainer import optimizers
    if opts.loss_function == 'cross_entropy':
        model = L.Classifier(nnet_model)
        model.compute_accuracy = False
    elif opts.loss_function == 'mean_squared_error':
        model = L.Classifier(nnet_model, lossfun=F.mean_squared_error)
        model.compute_accuracy = False
    else:
        misc.error('Loss function %s is not supported!' % opts.loss_function)

    optimizer = optimizers.SGD(lr=opts.lr)
    optimizer.setup(nnet_model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(opts.grad_clip))
    return optimizer, model


def print_stats(percentage, epoch, lr, loss):
    print('\r#Epoch {0}, lr: {1} [{2}] {3}% loss: {4}'.format(
             epoch, "%.6f" % lr, '#' * (percentage / 5),
             percentage, "%.3f" % loss), end='')


def evaluation(nnet_model, fname, show_progress=False):
    dev_model = nnet_model.copy()
    dev_model.to_cpu()
    devset = data_util.load_data(fname)
    dev_data_resource = data_util.data_spliter(devset, batchsize=1, n_epoch=1)
    if opts.loss_function == 'cross_entropy':
        model = L.Classifier(dev_model)
    if opts.loss_function == 'mean_squared_error':
        model = L.Classifier(dev_model, lossfun=F.mean_squared_error)
        model.compute_accuracy = False
    dev_loss = 0
    pred = []
    target = []
    for dev_idx, x_batch, y_batch, epoch, percentage, eos in dev_data_resource:
        if not dev_idx:
            break
        if show_progress:
            print_stats(percentage, 0, 0, dev_loss)
        x = Variable(np.asarray(x_batch))
        t = Variable(np.asarray(y_batch))
        loss_i = model(x, t)
        target.append(y_batch[0])
        pred.append(model.y.data[0].argmax())
        dev_loss = (dev_loss * (dev_idx - 1) + loss_i.data) / dev_idx
        if eos and opts.forget_on_new_utt:
            dev_model.forget_history()
    return dev_loss, pred, target


def train_nnet(nnet_model, train_data_resource, opts):
    optimizer, model = setup_training(nnet_model, opts)
    if opts.gpu >= 0:
        cuda.check_cuda_available()
        model.to_gpu(opts.gpu)
    accum_loss = 0

    i = 0
    train_loss = 0
    prev_dev_loss = 100000
    prev_percentage = 0
    for train_idx, x_batch, y_batch, epoch, percentage, eos in train_data_resource:
        if train_idx is None:  # Done one epoch
            if opts.fname_dev:
                dev_loss, _, _ = evaluation(nnet_model, opts.fname_dev)
                print(' dev loss: %.3f' % dev_loss, end='')
                if optimizer.lr < opts.lr_stop:
                    break
                if prev_dev_loss - dev_loss < opts.start_decay:
                    optimizer.lr *= opts.lr_decay
                    print('...reducing lr to %.6f' % optimizer.lr)
            print('')
            continue
        x = Variable(xp.asarray(x_batch))
        t = Variable(xp.asarray(y_batch))
        loss_i = model(x, t)
        accum_loss += loss_i
        if train_idx == 0:
            print('Dump graph')
            with open('graph.dot', 'w') as o:
                o.write(c.build_computational_graph((loss_i, )).dump())
        if train_idx >= 1:
            train_loss = (train_loss * (train_idx - 1) + loss_i.data) / train_idx

        if eos and opts.forget_on_new_utt:
            nnet_model.forget_history()

        if eos or (i + 1) % opts.bprop_len == 0:
            model.zerograds()
            accum_loss.backward()
            accum_loss.unchain_backward()
            accum_loss = 0
            optimizer.update()
            i = 0

        if percentage != prev_percentage:
            prev_percentage = percentage
            print_stats(percentage, epoch, optimizer.lr, train_loss)
            sys.stdout.flush()
        i += 1


def train_mode():
    nnet_model = None
    if opts.nnet_struct:
        nnet_model = NNET_Model.parse_structure(opts.nnet_struct)
    elif opts.fname_in_model:
        nnet_model = NNET_Model.load(opts.fname_in_model)

    trainset = data_util.load_data(opts.fname_train)

    eos_pad = misc.get_pading(opts.eos_pad)

    train_data_resource = data_util.data_spliter(trainset, batchsize=opts.batchsize,
                            n_epoch=opts.n_epoch, EOS=eos_pad)
    train_nnet(nnet_model, train_data_resource, opts)
    if opts.fname_test:
        print('====================TESTING=========================')
        test_loss, pred, target = evaluation(nnet_model, opts.fname_test, show_progress=True)
        if 'cross_entropy' in opts.loss_function:
            misc.f_measure(pred, target)
        print(' test loss: %.3f' % test_loss)

    if opts.fname_out_model:
        nnet_model.save(opts.fname_out_model)


def test_mode():
    nnet_model = NNET_Model.load(opts.fname_in_model)
    print('====================TESTING=========================')
    test_loss = evaluation(nnet_model, opts.fname_test, show_progress=True)
    print(' test loss: %.3f' % test_loss)


def main():
    if opts.fname_train:
        train_mode()
    elif opts.fname_test:
        test_mode()


if __name__ == '__main__':
    main()
