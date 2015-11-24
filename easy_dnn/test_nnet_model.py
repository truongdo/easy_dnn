#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 truong-d <truong-d@ahclab08>
#
# Distributed under terms of the MIT license.

"""

"""
import unittest
import chainer
import chainer.links as L
import chainer.functions as F
import os
import shutil
import numpy as np
from nnet_model import NNET_Model
from component_holder import ComponentHolder


class Test_ComponentHolder(unittest.TestCase):
    def setUp(self):
        self.comp_holder = ComponentHolder()
        self.layer1 = F.Linear(2, 2)
        self.layer2 = F.Linear(2, 4)
        self.comp_holder.append_comp('layer1', self.layer1)
        self.comp_holder.append_comp('layer2', self.layer2)
        pass

    def test_append_comp(self):
        self.assertIs(self.comp_holder['layer1'], self.layer1)
        a, b = self.comp_holder['layer1', 'layer2']
        self.assertIs(a, self.layer1)
        self.assertIs(b, self.layer2)
        
        name, comp = self.comp_holder[0]
        self.assertEqual(name, 'layer1')
        self.assertIs(comp, self.layer1)

    def tearDown(self):
        pass


class Test_NNET_Model(unittest.TestCase):
    def setUp(self):
        if not os.path.exists('test_save_model'):
            os.makedirs('test_save_model')
        self.nnet_model = NNET_Model()
        pass
    
    def test_NNET_add_comp(self):
        self.nnet_model.add_component('linear(2-2)', L.Linear(2, 2))
    
    def test_NNET_parse_struct(self):
        struct = 'linear(2-4):sigmoid:tanh:dropout:lstm(10-10)'
        nnet = NNET_Model.parse_structure(struct)
        self.assertIsInstance(nnet['linear(2-4)-c0'], L.Linear)
        self.assertIsInstance(nnet['lstm(10-10)-c4'], L.LSTM)

    def test_NNET_load(self):
        struct = 'linear(2-4):sigmoid:tanh:dropout:lstm(10-10)'
        nnet = NNET_Model.parse_structure(struct)
        nnet.save('test_save_model')
        another = NNET_Model.load('test_save_model')

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
