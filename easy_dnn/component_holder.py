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
from collections import OrderedDict

class ComponentHolder:
    def __init__(self):
        """A class that hold NNET component object.
        Args:
            
        Attributes:
            _holder (OrderedDict): A dictionary that hold all components.

        Examples:
        """
        self._holder = OrderedDict()
    
    def _get_one(self, x):
        if isinstance(x, str):
            return self._holder[x]
        else:
            return self._holder_list[x]
    
    def __setitem__(self, key, value):
        """Set component, do not use int as key"""
        if isinstance(key, int):
            raise Exception('Do not support int as key')
        self._holder[key] = value
        
    def append_comp(self, name, comp):
        self.__setitem__(name, comp)

    def __getitem__(self, x):
        self._holder_list = [(key, value) for key, value in self._holder.items()]
        if isinstance(x, tuple):
            return [self._get_one(x_i) for x_i in x]
        else:
            return self._get_one(x)
    
    def items(self):
        return self._holder.items()
