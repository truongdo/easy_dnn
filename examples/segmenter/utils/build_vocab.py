#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 truong-d <truong-d@ahcsegpu01>
#
# Distributed under terms of the MIT license.

"""

"""
import sys
import codecs
import operator
import string
from collections import defaultdict, OrderedDict

counter = defaultdict(lambda: 0)
w_dic = OrderedDict()
t_dic = OrderedDict()
out_dir = sys.argv[-1]

for fname in sys.argv[1:-1]:
    for line in codecs.open(fname, 'r', 'utf-8'):
        line = line.strip()
        if not line:
            continue
        w, t = line.strip().split()
        if not w in w_dic:
            w_dic[w] = len(w_dic)
        if not t in t_dic:
            t_dic[t] = len(t_dic)
        
        counter[w] += 1

sorted_x = sorted(counter.items(), key=operator.itemgetter(1))
unk_id = w_dic[sorted_x[0][0]]
w_dic["<unk>"] = unk_id
for p in list(string.punctuation):
    w_dic[p] = unk_id

fout_dic = codecs.open(out_dir + '/words.int', 'w', 'utf-8')
for w, wid in w_dic.items():
    fout_dic.write("%s %d\n" % (w, wid))

fout_tag = codecs.open(out_dir + '/tags.int', 'w', 'utf-8')
for t, tid in t_dic.items():
    fout_tag.write("%s %d\n" % (t, tid))


