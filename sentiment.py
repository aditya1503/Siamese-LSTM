
# coding: utf-8

# In[7]:

from lstm import *

from sklearn.svm import SVC


# In[2]:

lst=lstm(False)
train=pickle.load(open("kaggle.p",'rb'))


# In[8]:

def prepare_data2(data):
    xa1=[]
    xb1=[]
    y2=[]
    for i in range(0,len(data)):
        xa1.append(data[i][0])
        #xb1.append(data[i][1])
        #y2.append(round(data[i][2],0))
        y2.append(data[i][1])
    lengths=[]
    for i in xa1:
        lengths.append(len(i.split()))
    #for i in xb1:
    #    lengths.append(len(i.split()))
    maxlen = numpy.max(lengths)
    emb1,mas1=getmtr(xa1,maxlen)
    #emb2,mas2=getmtr(xb1,maxlen)
    
    #y2=np.array(y2,dtype=np.float32)
    return emb1,mas1,y2
def getmtr(xa,maxlen):
    n_samples = len(xa)
    ls=[]
    x_mask = numpy.zeros((maxlen, n_samples)).astype(np.float32)
    for i in range(0,len(xa)):
        q=xa[i].split()
        for j in range(0,len(q)):
            x_mask[j][i]=1.0
        while(len(q)<maxlen):
            q.append(',')
        ls.append(q)
    xa=np.array(ls)
    return xa,x_mask
def fpro(mydata):
    count=[]
    num=len(mydata)
    px=[]
    yx=[]
    use_noise.set_value(0.)
    for i in range(0,num,64):
        q=[]
        x=i+64
        if x>num:
            x=num
        for j in range(i,x):
            q.append(mydata[j])
        x1,mas1,y2=prepare_data2(q)
        ls=[]
        ls2=[]
        for j in range(0,len(q)):
            ls.append(embed(x1[j]))
            #ls2.append(embed(x2[j]))
        trconv=np.dstack(ls)
        emb1=np.swapaxes(trconv,1,2)
        pred=lst.f_proj11(emb1,mas1)
        #dm1=np.ones(mas1.shape,dtype=np.float32)
        #dm2=np.ones(mas2.shape,dtype=np.float32)
        #corr=f_cost(emb1,mas1,emb2,mas2,y2)
        for z in range(0,len(q)):
            yx.append(y2[z])
            px.append(pred[z])
    px=np.array(px)
    yx=np.array(yx)
    return px,yx
def getacc(vlc,xl,yl):
    prd=vlc.predict(xl)
    sc=np.sum(yl==prd)/float(len(yl))*100.0
    return sc


# In[29]:

for i in range(0,10):
    shuffle(train)


# In[30]:

xdat,ydat=fpro(train)
c7=int(0.7*len(xdat))

xtr=xdat[0:c7]
ytr=ydat[0:c7]
xcr=xdat[c7:]
ycr=ydat[c7:]


# In[31]:

clf = SVC(C=100,gamma=3.1,kernel='rbf')
#xtr:training data
#xcr:cross validation data

# In[32]:

scl=clf.fit(xtr,ytr)
print "Training accuracy:",getacc(scl,xtr,ytr)
print "Cross validation accuracy:",getacc(scl,xcr,ycr)


# In[ ]:



