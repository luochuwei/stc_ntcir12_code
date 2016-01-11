#-*- coding: utf-8 -*-
from gensim.models.word2vec import *
from gensim import models
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

infile = r'C:\Users\hkpuadmin\Desktop\stc\STC_task\repos\seg\f_all_sentence_seg.txt'
model = Word2Vec(LineSentence(infile), size=300, window=5, min_count=0, workers=8)
model.save_word2vec_format('all_word_vec.txt')

#model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)

print 'DONE'