from lstm import *
sls=lstm("bestsem.p",load=True,training=False)

test=pickle.load(open("semtest.p",'rb'))
#Example
sa="A truly wise man"
sb="He is smart"
print sls.predict_similarity(sa,sb)*4.0+1.0