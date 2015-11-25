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

w_dic_file = sys.argv[1] + '/' + 'words.int'
t_dic_file = sys.argv[1] + '/' + 'tags.int'
text_file = sys.argv[2]
out_file = sys.argv[3]

w_dic = {}
t_dic = {}

error = 0
error_line = []
for line in codecs.open(w_dic_file, 'r', 'utf-8'):
    try:
        s, idx =  line.strip().split()
        w_dic[s] = idx
    except:
        error_line.append(line.strip())
        error += 1

for line in codecs.open(t_dic_file, 'r', 'utf-8'):
    s, idx =  line.strip().split()
    t_dic[s] = idx


fout = codecs.open(out_file, 'w', 'utf-8')
for line in codecs.open(text_file, 'r', 'utf-8'):
    line = line.strip()
    if not line:
        fout.write('\n')
    else:
        w, t = line.strip().split()
        out_w, out_t = "", ""
        if not w in w_dic:
            w = "<unk>"

        fout.write(w_dic[w] + ' %s\n' % t_dic[t])
