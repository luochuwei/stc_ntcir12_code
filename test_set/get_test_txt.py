#-*- coding:utf-8 -*-


f = open(r'test-id-post-cn')

ft = open(r'test_txt.txt', 'w')



for line in f:
    ID, txt = line.split('\t')
    ft.write(txt)

f.close()
ft.close()