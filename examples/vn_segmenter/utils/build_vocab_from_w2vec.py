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
import sys
import codecs
import operator
from collections import defaultdict, OrderedDict
from gensim.models import Word2Vec
import string
from smartencoding import smart_unicode

def load_from_w2vec(fname):
    model = Word2Vec.load_word2vec_format(fname, binary=True)
    
    in_size, out_size = model.syn0.shape
    vocab = OrderedDict()
    for w, v in model.vocab.items():
        try:
            if ('\xc2\xa0' in w.encode('utf-8')) or ('\xe3\x80\x80' in w.encode('utf-8')):
                continue
            vocab[w] = v.index
        except:
            print('WARN: bad word:', end='')
            print(w)
            pass
    vocab[u"<unk>"] = -1

    for p in list(string.punctuation):
        vocab[p] = vocab[u"<unk>"]
    return vocab

t_dic = OrderedDict()
out_dir = sys.argv[-1]
train_file = sys.argv[1]
w2vec_file = sys.argv[2]

w_dic = load_from_w2vec(w2vec_file)

for line in codecs.open(train_file, 'r', 'utf-8'):
    line = line.strip()
    if not line:
        continue
    w, t = line.strip().split()
    if not t in t_dic:
        t_dic[t] = len(t_dic)

fout_dic = codecs.open(out_dir + '/words.int', 'w', 'utf-8')
error = 0
for w, wid in w_dic.items():
    w = smart_unicode(w)
    fout_dic.write("%s %d\n" % (smart_unicode(w), wid))
    # if not w or  len(w.split()) != 1:
        # continue
    # try: 
        # w = smart_unicode(w)
        # fout_dic.write("%s %d\n" % (smart_unicode(w), wid))
    # except:
        # print('WARN: bad word')
        # print(w)
        # error += 1
fout_tag = codecs.open(out_dir + '/tags.int', 'w', 'utf-8')
for t, tid in t_dic.items():
    fout_tag.write("%s %d\n" % (t, tid))
