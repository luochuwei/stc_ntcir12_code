#-*- coding:utf-8 -*-

import cPickle



f = open(r'repos-id-post-cn')


r_post = []
r_only_post = {}

for line in f:
    a, b = line[:-1].split('\t', 1)
    r_post.append(a)
    r_only_post[a] = b



f.flush()
f.close()


# f2 = open(r'repos-id-cmnt-cn')


# r_cmnt = []
# for line in f2:
#     r_cmnt.append(line)



# f2.flush()
# f2.close()


# cPickle.dump(r_post, open(r'r_post.pkl', 'wb'))
# cPickle.dump(r_cmnt, open(r'r_cmnt.pkl', 'wb'))


f3 = open(r'r-only-post.txt', 'w')
for i,j in r_only_post.iteritems():
    f3.write(i)
    f3.write("\t")
    f3.write(j)
    f3.write('\n')

f3.flush()
f3.close()



r_id_answer = {}

assert len(r_cmnt) == len(r_post)

for i in range(len(r_post)):
    if r_post[i][0] in r_id_answer:
        r_id_answer[r_post[i][0]].append(r_cmnt[i][:-1])
    else:
        r_id_answer[r_post[i][0]] = [r_cmnt[i][:-1]]

cPickle.dump(r_id_answer, open(r'ID_r_id_answer.pkl', 'wb'), True)