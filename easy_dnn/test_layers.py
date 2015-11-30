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
import numpy as np

class Test_Links(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_embed(self):
        embed = L.EmbedID(5, 4)
        embed.save('embed.layer')
        another = L.EmbedID.load('embed.layer')
        
        inp = chainer.Variable(np.asarray([0], dtype=np.int32))
        print(embed(inp).data)
        print(another(inp).data)
        np.testing.assert_equal(embed(inp).data, another(inp).data)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
