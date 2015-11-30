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
import chainer
import six
import os
import numpy as np
import math
import chainer.links as L
import chainer.functions as F
from misc import *
import component_holder


def get_size_info(struct):
    '''Parse linear component
    Args:
        struct (string): format component_name(124-125)'''
    try:
        in_size, out_size = struct.split('(')[1].replace(')', '').split('-')
        return int(in_size), int(out_size)
    except:
        raise Exception('''Wrong component format <''' + struct  + '''>. The correct format is linear(ninput-noutput). For example, linear(100-200).''')
    return in_size, out_size


def parse_component(struct):
    if 'linear' in struct:
        in_size, out_size = get_size_info(struct)
        comp = L.Linear(in_size, out_size)
    elif 'embed' in struct:
        in_size, out_size = get_size_info(struct)
        comp = L.EmbedID(in_size, out_size)
    elif 'lstm' in struct:
        in_size, out_size = get_size_info(struct)
        # init_W = np.random.normal(
                # 0, 1 * math.sqrt(1. / out_size),
                # (4 * out_size, in_size)).astype(np.float32)
        # init_B = np.repeat(np.float32(0), out_size * 4)

        # upward=linear.Linear(in_size, 4 * out_size)
        # lateral=linear.Linear(out_size, 4 * out_size, nobias=True)
        comp = L.LSTM(in_size, out_size)
    elif 'sigmoid' in struct:
        comp = F.sigmoid
    elif 'softmax' in struct:
        comp = F.softmax
    elif 'tanh' in struct:
        comp = F.tanh
    elif 'dropout' in struct:
        comp = F.dropout
    elif 'w2vec' in struct:
        fname = struct.split('(')[1].replace(')', '')
        from gensim.models import Word2Vec
        import string
        model = Word2Vec.load_word2vec_format(fname, binary=True)
        
        in_size, out_size = model.syn0.shape
        comp = L.EmbedID(in_size, out_size)
        comp.W.data = model.syn0
        comp.volatile = True
        return comp
    else:
        error('Struct: ' + struct + ' is not supported')
    return comp 


class NNET_Model(chainer.Chain):
    def __init__(self):
        """Neural network model.
        Args:
            None

        Attributes:
            _comp_holder (nnet_model.ComponentHolder): An object that hold all components.

        Examples:
            >>> nnet_model = NNET_Model()
            >>> nnet_model.add_component('l1', chainer.links.Linear(2, 2))
            >>> a = chainer.Variable(np.random.rand(0, 1, (2, 2), dtype=np.float32))
            >>> output = nnet_model(a)
        """
        super(NNET_Model, self).__init__()
        self._comp_holder = component_holder.ComponentHolder()
        self.structure = None
        self.debug = False
    
    def add_component(self, name, component):
        """ Add new component to the model.
        Args:
            name (string): the component name
            component (chainer.links.*): the component. For example, chainer.links.Linear
        
        Examples:
            >>> nnet_model = NNET_Model()
            >>> nnet_model.add_component('l1', chainer.links.Linear(2, 2))
            """
        self._comp_holder[name] = component
        super(NNET_Model, self).add_link(name, component)
    
    def copy(self):
        ret = NNET_Model.parse_structure(self.structure)
        for name, comp in self._comp_holder.items():
            ret._comp_holder[name].copyparams(comp)
        return ret
    
    def forget_history(self):
        # Find any lstm or rnn layer, then forget history
        for name, comp in self._comp_holder.items():
            if 'lstm' in name:
                comp.reset_state()

    def __getitem__(self, comp_ident):
        return self._comp_holder[comp_ident]

    def items(self):
        return self._comp_holder.items()

    def __call__(self, x):
        layer_input = x
        for name, comp in self._comp_holder.items():
            if hasattr(comp, 'volatile'):
                layer_input.volatile = comp.volatile
            layer_input = comp(layer_input)
            layer_input.volatile = False
            if hasattr(comp, 'activation'):
                layer_input = comp.activation(layer_input)
            if self.debug:
                print(name, layer_input.data)
                
        return layer_input 

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.to_cpu()
        fout_cfg = open(os.path.join(folder, 'nnet_struct.cfg'), 'w')
        fout_cfg.write(self.structure + '\n')
        idx = 0
        for name, comp in self._comp_holder.items():
            fname = os.path.join(folder, name)
            if 'w2vec' not in name:
                comp.save(fname)
                fout_cfg.write(name + '|||' + fname + '\n')
            idx += 1

    @classmethod
    def parse_structure(cls, nnet_struct):
        import inspect
        nnet_model = cls()
        nnet_model.structure = nnet_struct
        prev_comp = None
        for sid, struct in enumerate(nnet_struct.split(':')):
            comp = parse_component(struct)
            if inspect.isfunction(comp):
                if prev_comp is None:
                    error('An activation function cannot appear at the begining of the nnetstructure')
                prev_comp.activation = comp
                continue
            struct_name = struct + "-c%d" % sid
            nnet_model.add_component(struct_name, comp)
            prev_comp = comp
        return nnet_model

    @classmethod
    def load(cls, folder):
        nnet_cfg = open(os.path.join(folder, 'nnet_struct.cfg'), 'r').readlines()
        nnet_struct = nnet_cfg[0].strip()
        nnet_model = NNET_Model.parse_structure(nnet_struct)
        
        for name_comp in nnet_cfg[1:]:
            # print('Loading', name_comp)
            name, fin_comp = name_comp.strip().split('|||')
            comp = nnet_model[name].load(fin_comp)
            nnet_model[name].copyparams(comp)
        return nnet_model

