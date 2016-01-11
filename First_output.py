import cPickle
import os
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.word2vec import *
import math
import numpy as np
import re
import time

def tf_idf_all(word, sentence):
    sp = sentence.split()
    tf = sp.count(word)/len(set(sp))
    #post + resp 一共有5844523条句子
    f = open(r'C:\Users\hkpuadmin\Desktop\stc\STC_task\repos\seg\f_all_sentence_seg.txt')
    n = 0
    for line in f:
        if word in line:
            n+=1
    if n == 0:
        return 1.0
    f.flush()
    f.close()
    idf = math.log(5844523/float(n) + 0.01)
    return tf*idf

def tf_idf_c(word, sentence, candidate, similar_post_dic):
    sp = sentence.split()
    tf = sp.count(word)/float(len(set(sp)))

    n = 0
    for line in candidate:
        if word in line[4]:
            n+=1
    for i,j in similar_post_dic.iteritems():
        if word in j[0]:
            n+=1
    if n == 0:
        return 1.0

    idf = math.log((len(candidate)+len(similar_post_dic))/float(n) + 0.01)
    return tf*idf

def tf_idf_post_data(word, sentence, all_post_vec):
    sp = sentence.split(' ')
    tf = sp.count(word)/len(set(sp))
    n = 0
    for line in all_post_vec:
        if word in line[1]:
            n+=1
    if n == 0:
        return 1.0
    idf = math.log(len(all_post_vec)/float(n)+0.01)
    return tf*idf



def get_similar_post(file_path):
    f = open(file_path)
    similar_post_dic = {}
    #similar_post_dic[对应文件打开后的post编号，比如repos-post-1001873310]=[对应的分词后的句子，对应的分数0.7823]
    for line in f:
        l=line[:-1].split(' ',1)
        f2 = open(l[1])
        f2read = f2.read()
        f2.flush()
        f2.close()
        f2s = f2read.split('\t',1)
        seg_f2s1 = jieba.cut(f2s[1][:-1].decode('utf-8'), cut_all=False)
        similar_post_dic[f2s[0]] = [' '.join(seg_f2s1), float(l[0].split('=')[-1])]
    return similar_post_dic

def get_similar_post_vec_based(model, test_txt, all_post_vec):
    test_txt_vec = get_sentence_vec2(model, test_txt, all_post_vec)
    for i in range(len(all_post_vec)):
        p_v = get_sentence_vec2(model, all_post_vec[i][1], all_post_vec)
        score = float(cosine_similarity(test_txt_vec, p_v))
        all_post_vec[i][2] = p_v
        all_post_vec[i][3] = score
    ranked_all_post_vec = sorted(all_post_vec, key = lambda x:x[3], reverse=True)
    similar_post_dic = {}
    for item in ranked_all_post_vec[:10]:
        similar_post_dic[item[0]] = [item[1],item[3]]
    return similar_post_dic



def get_resp_candidate(similar_post_dic, ID_r_answer):
    resp_candidate_list = []
    for post_id in similar_post_dic:
        for resp in ID_r_answer[post_id]:
            # return resp
            resp_id, resp_sentence = resp.split('\t',1)
            seg_list = jieba.cut(resp_sentence.decode('utf-8'), cut_all=False)
            resp_sentence_seg = ' '.join(seg_list)
            resp_candidate_list.append([post_id, similar_post_dic[post_id][0],'p_vec', resp_id, resp_sentence_seg, 'r_vec',similar_post_dic[post_id][1], 's2', 's3', 'rank_score'])
            #输出的response是分词后的 并且是unicode编码
    return resp_candidate_list

def get_sentence_vec(model, sentence, candidate_list, similar_post_dic):
    s_vec = np.zeros(model.layer1_size)
    for word in sentence.split(' '):
        if word in model.vocab:
            s_vec += tf_idf_c(word, sentence, candidate_list, similar_post_dic)*model[word]
    return s_vec

def get_sentence_vec2(model, sentence, all_post_vec):
    s_vec = np.zeros(model.layer1_size)
    for word in sentence.split(' '):
        if word in model.vocab:
            s_vec += model[word]
    return s_vec

def get_ranked_response(model, test_post_seg, candidate_list, similar_post_dic):
    test_post_seg_vec = get_sentence_vec(model, test_post_seg, candidate_list, similar_post_dic)
    for c in candidate_list:
        c_p_vec = get_sentence_vec(model, c[1], candidate_list, similar_post_dic)
        c_r_vec = get_sentence_vec(model, c[4], candidate_list, similar_post_dic)
        c[2] = c_p_vec
        c[5] = c_r_vec
        s2 = float(cosine_similarity(c_p_vec, c_r_vec))
        s3 = float(cosine_similarity(test_post_seg_vec, c_r_vec))
        c[7] = s2
        c[8] = s3
        # rank_score = 1000*c[6]*c[7]*c[8]
        rank_score = c[6]*0.5+c[7]*1.5+c[8]*2
        c[9] = rank_score
    rank_candidate = sorted(candidate_list, key = lambda l: l[-1])
    return rank_candidate









"""Main"""
"""
>>> print ID_r_answer['repos-post-1001470110'][1]
repos-cmnt-1014959310    看这话时候刚好喝了口水,差点喷出.
>>> print ID_r_answer['repos-post-1001470110'][2]
repos-cmnt-1020316050    什么时间播出呀?

output的txt：比如说1.txt如下：
Score=0.78466123 D:\STC\TXT\new\183117.txt
Score=0.7820012 D:\STC\TXT\new\40521.txt
Score=0.7617575 D:\STC\TXT\new\3121.txt
Score=0.7567925 D:\STC\TXT\new\5546.txt
Score=0.73571545 D:\STC\TXT\new\82555.txt
Score=0.68764776 D:\STC\TXT\new\73548.txt
Score=0.67144805 D:\STC\TXT\new\55411.txt
Score=0.65652466 D:\STC\TXT\new\178003.txt
Score=0.6400629 D:\STC\TXT\new\29873.txt
Score=0.6357535 D:\STC\TXT\new\100621.txt

D:\STC\TXT\new\1.txt 如下
repos-post-1001873310 四点半，625号，我想知道一号是几点来的。。。
"""


test_txt_path = r'D:\train.txt'
test_txt = open(test_txt_path)
test_txt_list = []
for line in test_txt:
    s_line = jieba.cut(line[:-1].decode('utf-8'), cut_all=False)
    test_txt_list.append(' '.join(s_line))
test_txt.close()

output_f_list = []
output_dir = r'D:\STC\output'

for fname in os.listdir(output_dir):
    output_f_list.append(output_dir+"\\"+fname)

assert len(test_txt_list) == len(output_f_list)

# r_id_answer = cPickle.load(open(r'r_id_answer.pkl', 'rb'))
ID_r_answer = cPickle.load(open(r'ID_r_id_answer.pkl', 'rb'))

model = Word2Vec.load_word2vec_format(r'C:\Users\hkpuadmin\Desktop\stc\STC_task\repos\all_word_vec.txt', binary=False)

ranked_output_dir = r'C:\Users\hkpuadmin\Desktop\stc\STC_task\repos\output_files//'

r = re.compile('[0-9]+')

for path in output_f_list:
    index = int(r.findall(path)[0])
    print index,'    start...'
    t1 = time.time()
    similar_post_dic = get_similar_post(path)
    candidate_list = get_resp_candidate(similar_post_dic, ID_r_answer)
    rank_candidate = get_ranked_response(model, test_txt_list[index], candidate_list, similar_post_dic)
    print index,'    get ranked candidate...'
    rank_candidate_file = open(ranked_output_dir+str(index)+'.txt', 'w')
    for c in reversed(rank_candidate):
        rank_candidate_file.write(c[3].encode('utf-8')+'\t'+c[4].encode('utf-8')+'\t'+str(c[6])+'\t'+str(c[7])+'\t'+str(c[8])+'\t'+str(c[9]))
        rank_candidate_file.write('\n')
    rank_candidate_file.flush()
    rank_candidate_file.close()
    print index, '    done...'
    t2 = time.time()
    print 'cost '+str(t2-t1)+'s'



"""
second ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# test_txt_path = r'D:\train.txt'
# test_txt = open(test_txt_path)
# test_txt_list = []
# for line in test_txt:
#     s_line = jieba.cut(line[:-1].decode('utf-8'), cut_all=False)
#     test_txt_list.append(' '.join(s_line))


# ID_r_answer = cPickle.load(open(r'ID_r_id_answer.pkl', 'rb'))
# model = Word2Vec.load_word2vec_format(r'C:\Users\hkpuadmin\Desktop\stc\STC_task\repos\all_word_vec.txt', binary=False)

# all_post_vec = []
# only_post_seg = open(r'C:\Users\hkpuadmin\Desktop\stc\STC_task\repos\seg\r-only-post-seg.txt')
# for line in only_post_seg:
#     post_seg_id, post_seg = line[:-1].split('\t',1)
#     all_post_vec.append([post_seg_id, post_seg.decode('utf-8'), 'post_vec', 'score'])
# only_post_seg.flush()
# only_post_seg.close()




# ranked_output_dir = r'C:\Users\hkpuadmin\Desktop\stc\STC_task\repos\output_files2//'