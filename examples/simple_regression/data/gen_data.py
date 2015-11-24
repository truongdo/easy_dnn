#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 truong-d <truong-d@ahcsegpu01>
#
# Distributed under terms of the MIT license.

"""

"""
from __future__ import print_function
import numpy as np

x = np.random.rand(1000, 4)
y = x * 4 + 1

i = 0
for sub_x, sub_y in zip(x, y):
    print(' '.join([str(a) for a in sub_x]), end=' ')
    print(' '.join([str(a) for a in sub_y]))
    i += 1
    if i % 100 == 0:
        print('')
