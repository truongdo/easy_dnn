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
import chainer.cuda as cuda
import chainer.optimizers as optimizers
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
        np.random.seed(0)
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
    
    def test_NNET_save(self):
        struct = 'embed(2-2):lstm(2-2)'
        nnet = NNET_Model.parse_structure(struct)
        nnet.save('test_output')
        
        nnet2 = NNET_Model.load('test_output')
        inp = chainer.Variable(np.asarray([1], dtype=np.int32))

        np.testing.assert_equal(nnet[0][1].W.data, nnet2[0][1].W.data)
        np.testing.assert_equal(nnet[0][1](inp).data, nnet2[0][1](inp).data)

        np.testing.assert_equal(nnet(inp).data, nnet2(inp).data)
        
        struct = 'embed(2-2):lstm(2-3):linear(3-2)'
        nnet = NNET_Model.parse_structure(struct)
        nnet.save('test_output')
        
        nnet2 = NNET_Model.load('test_output')
        inp = chainer.Variable(np.asarray([1], dtype=np.int32))
        np.testing.assert_equal(nnet(inp).data, nnet2(inp).data)

    def test_NNET_train(self):
        xp = cuda.cupy

        # struct = 'w2vec(/project/nakamura-lab01/Work/truong-dq/chainer/vidnn/exp/word2vec_truecase_200/vectors.bin):lstm(200-2):linear(2-2)'
        struct = 'embed(2-200):lstm(200-2):linear(2-2)'
        nnet = NNET_Model.parse_structure(struct)
        nnet.to_gpu()

        # Testing variable
        test_var = chainer.Variable(xp.asarray([1], dtype=np.int32))
        output_before_train = cuda.to_cpu(nnet(test_var).data)


        inp = chainer.Variable(xp.asarray([1], dtype=np.int32))
        target = chainer.Variable(xp.asarray([0], dtype=np.int32))
        output = nnet(inp)
        loss = F.softmax_cross_entropy(output, target)
        optimizer = optimizers.SGD(lr=0.1)
        optimizer.setup(nnet)
        optimizer.zero_grads()
        loss.backward()
        optimizer.update()
        
        inp = chainer.Variable(xp.asarray([0], dtype=np.int32))
        target = chainer.Variable(xp.asarray([0], dtype=np.int32))
        output = nnet(inp)
        loss = F.softmax_cross_entropy(output, target)
        optimizer = optimizers.SGD(lr=0.1)
        optimizer.setup(nnet)
        optimizer.zero_grads()
        loss.backward()
        optimizer.update()

        
        nnet.save('test_output')
        nnet.to_gpu()
        nnet_2 = NNET_Model.load('test_output')
        nnet_2.to_gpu()
        nnet.forget_history()
        nnet_2.forget_history()

        np.testing.assert_equal(cuda.to_cpu(nnet[0][1].W.data), cuda.to_cpu(nnet_2[0][1].W.data))
        np.testing.assert_equal(cuda.to_cpu(nnet[1][1].upward.W.data), cuda.to_cpu(nnet_2[1][1].upward.W.data))
        np.testing.assert_equal(cuda.to_cpu(nnet[1][1].lateral.W.data), cuda.to_cpu(nnet_2[1][1].lateral.W.data))
        np.testing.assert_equal(cuda.to_cpu(nnet[1][1].upward.b.data), cuda.to_cpu(nnet_2[1][1].upward.b.data))

        output_after_train = cuda.to_cpu(nnet(test_var).data)
        output_after_load = cuda.to_cpu(nnet_2(test_var).data)
        
       
        after_first_layer_nnet = nnet[0][1](test_var)
        after_first_layer_nnet_2 = nnet_2[0][1](test_var)
        np.testing.assert_equal(cuda.to_cpu(after_first_layer_nnet.data), cuda.to_cpu(after_first_layer_nnet_2.data))
        
        after_first_layer_nnet.volatile = False
        after_first_layer_nnet_2.volatile = False
        after_second_layer_nnet = nnet[1][1](after_first_layer_nnet)
        after_second_layer_nnet_2 = nnet_2[1][1](after_first_layer_nnet_2)
        np.testing.assert_equal(cuda.to_cpu(after_second_layer_nnet.data), cuda.to_cpu(after_second_layer_nnet_2.data))
        
        assert (output_before_train != output_after_train).any()
        assert (output_before_train != output_after_load).any()
        np.testing.assert_equal(output_after_train, output_after_load)



    # def test_NNET_W2vec(self):
        # np.random.seed(0)
        # struct = 'w2vec(/project/nakamura-lab01/Work/truong-dq/chainer/vidnn/exp/word2vec_truecase_200/vectors.bin):lstm(200-3):linear(3-2)'
        # nnet = NNET_Model.parse_structure(struct)
        # nnet.save('test_save_model_w2vec')
        # another = NNET_Model.load('test_save_model_w2vec')

        # inp = chainer.Variable(np.asarray([1], dtype=np.int32))

        # np.testing.assert_equal(nnet(inp).data, another(inp).data)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
