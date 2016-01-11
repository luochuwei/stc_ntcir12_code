import cPickle
f = open(r'test-id-post-cn')


a = {}

for n,line in enumerate(f):
    a[n],x = line.split('\t')

cPickle.dump(a, open(r'hang_testid.pkl', 'w'))