#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 truong-d <truong-d@ahcclp01>
#
# Distributed under terms of the MIT license.


from __future__ import print_function
import chainer
import chainer.functions as F
import chainer.links as L

import pickle
import numpy as np
import six
import copy


class NNET_Component(object):
    def __init__(self):
        super(NNET_Component, self).__init__()
        self.name = None
    
    def save(self, fname):
        return False

    def initialize(self, batchsize=1, xp=np, train=True):
        pass

    def __deepcopy__(self, nnet_comp):
        pass

    def get_params(self):
        return []


class NNET_Linear(NNET_Component):
    def __init__(self, in_size, out_size, **kwargs):
        self.in_size = in_size
        self.out_size = out_size
        self.layer = F.Linear(in_size, out_size, **kwargs)
    
    def __call__(self, x):
        return self.layer(x)

    def save(self, fname):
        pickle.dump([self.layer.W, self.layer.b], open(fname, 'wb'))
        return True
    
    def get_params(self):
        return [self.layer]

    def __deepcopy__(self, memo):
        new_layer = NNET_Linear(self.in_size, self.out_size,
                    initialW=self.layer.W, initial_bias=self.layer.b)
        return new_layer

    @classmethod
    def load(cls, fname):
        W, b = pickle.load(open(fname, 'rb'))
        (out_size, in_size) = W.shape
        assert out_size == b.shape[0]
        return cls(in_size, out_size, initialW=W, initial_bias=b)


class NNET_EmbedID(NNET_Component):
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.vocab = {}
        self.layer = F.EmbedID(in_size, out_size)
    
    def __call__(self, x):
        return self.layer(x)
    
    def set_input_size(self, in_size):
        self.layer.W = numpy.random.randn(in_size, self.out_size).astype(numpy.float32)
        self.layer.gW = numpy.full_like(self.layer.W, numpy.nan)

    def get_params(self):
        return [self.layer]

    def save(self, fname):
        pickle.dump([self.vocab, self.layer.W], open(fname, 'wb'))
        return True

    def __deepcopy__(self, memo):
        new_layer = NNET_EmbedID(self.in_size, self.out_size)
        new_layer.layer.W = self.layer.W
        new_layer.vocab = self.vocab
        return new_layer

    @classmethod
    def load(cls, fname):
        try:
            vocab, W = pickle.load(open(fname, 'rb'))
        except:
            return NNET_EmbedID.load_from_w2vec(fname)
        model = cls(W.shape[0], W.shape[1])
        model.layer.W = W
        model.vocab = vocab
        return model
    
    @classmethod
    def load_from_w2vec(cls, fname):
        from gensim.models import Word2Vec
        import string
        model = Word2Vec.load_word2vec_format(fname, binary=True)
        
        in_size, out_size = model.syn0.shape
        embed_layer = cls(in_size, out_size)
        embed_layer.layer.W = model.syn0
        for w, v in model.vocab.items():
            embed_layer.vocab[w] = v.index
        embed_layer.vocab["<unk>"] = -1
        for p in list(string.punctuation):
            embed_layer.vocab[p] = embed_layer.vocab["<unk>"]
        return embed_layer


class NNET_Lstm(NNET_Component):
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.x = NNET_Linear(in_size, 4 * out_size)
        self.h = NNET_Linear(out_size, 4 * out_size)
        self.state = self.reset_state(batch_size=1)

    def get_params(self):
        return [self.x.layer, self.h.layer]

    def reset_state(self, batch_size, gpu=None, train=True):
        if gpu is None:
            self.state = {
               'h': chainer.Variable(np.zeros((batch_size, self.out_size), dtype=np.float32),
                    volatile=not train),
               'c': chainer.Variable(np.zeros((batch_size, self.out_size,), dtype=np.float32),
                    volatile=not train)
                  }
        else:
            self.state = {
               'h': chainer.Variable(cuda.cupy.zeros((batch_size, self.out_size), dtype=np.float32),
                    volatile=not train),
               'c': chainer.Variable(cuda.cupy.zeros((batch_size, self.out_size,), dtype=np.float32),
                    volatile=not train)
                  }

    def initialize(self, batchsize=1, gpu=None, train=True):
        self.reset_state(batchsize, gpu=gpu, train=train)
        pass

    def __deepcopy__(self, memo):
        new_layer = NNET_Lstm(self.in_size, self.out_size)
        new_layer.x = copy.deepcopy(self.x)
        new_layer.h = copy.deepcopy(self.h)
        new_layer.state = copy.deepcopy(self.state)
        return new_layer

    def __call__(self, inp):
        h_in = self.x(inp) + self.h(self.state['h'])
        c, h =  F.lstm(self.state['c'], h_in)
        self.state['c'] = c
        self.state['h'] = h
        return h

    def save(self, fname):
        self.x.save(fname + '.x')
        self.h.save(fname + '.h')
        return True

    @classmethod
    def load(cls, fname):
        x = NNET_Linear.load(fname + '.x')
        h = NNET_Linear.load(fname + '.h')
        layer = cls(x.in_size, h.in_size) 
        layer.x = x
        layer.h = h
        return layer


class NNET_Tanh(NNET_Component):
    def __init__(self, **kwargs):
        pass
    
    def __call__(self, x):
        return F.tanh(x)
    
    def __deepcopy__(self, memo):
        return NNET_Tanh()


class NNET_Sigmoid(NNET_Component):
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, x):
        return F.sigmoid(x)

    def __deepcopy__(self, memo):
        return NNET_Sigmoid()


class NNET_Softmax(NNET_Component):
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, x, use_cudnn=True):
        return F.softmax(x, use_cudnn)

    def __deepcopy__(self, memo):
        return NNET_Softmax()


class NNET_Dropout(NNET_Component):
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, x, ratio=.5, train=True):
        return F.dropout(x, ratio=ratio, train=train)

    def __deepcopy__(self, memo):
        return NNET_Dropout()
