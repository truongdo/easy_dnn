#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 truong-d <truong-d@ahclab08>
#
# Distributed under terms of the MIT license.

"""

"""

def train_nnet(nnet_model, train_data_resource, opts):
    nnet_model = L.Classifier(nnet_model)
    optimizer = optimizers.SGD(lr=0.25)
    optimizer.setup(nnet_model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(opts.grad_clip))
    accum_loss = 0
    
    i = 0
    train_loss = 0
    train_idx = 1
    prev_percentage = 0
    for x_batch, y_batch, epoch, percentage, eos in train_data_resource:
        x = Variable(xp.asarray(x_batch))
        t = Variable(xp.asarray(y_batch))
        loss_i = nnet_model(x, t)
        accum_loss += loss_i
        if train_idx > 1:
            train_loss = (train_loss * (train_idx - 1) + loss_i.data) / train_idx

        if eos or (i + 1) % opts.bprop_len == 0:
            nnet_model.zerograds()
            accum_loss.backward()
            accum_loss.unchain_backward()
            accum_loss = 0
            optimizer.update()
            i = 0

        percentage = int(percentage)
        if percentage != prev_percentage:
            print('\r#Epoch {0}, lr: {1} [{2}] {3}% Training loss: {4}'.format(
                epoch, "%.6f" % optimizer.lr, '#'*(percentage/5),
                percentage, "%.2f" % train_loss), end='')
            prev_percentage = percentage

            if percentage == 100:
                train_idx = 0
                print('')
            sys.stdout.flush()
        i += 1
        train_idx += 1
    if opts.fname_out_model:
        nnet_model.predictor.save(opts.fname_out_model)
