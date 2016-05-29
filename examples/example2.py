from lstm import *
sls=lstm("bestsem.p",load=True,training=False)
#Gradient compilatoin takes a long time, hence training=False since we're loading examples
test=pickle.load(open("semtest.p",'rb'))
print sls.chkterr2(test) #Mean Squared Error,Pearson, Spearman
