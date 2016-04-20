import numpy as np
import numpy
import pickle
import gensim
from gensim.models import word2vec
d2=pickle.load(open("synsem.p",'rb'))
dtr=pickle.load(open("dwords.p",'rb'))
print "Loading Word2Vec"
#model=np.load("modelgensim.npy").item()

model = word2vec.Word2Vec.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz",binary=True)



def prepare_data(data):
    xa1=[]
    xb1=[]
    y2=[]
    for i in range(0,len(data)):
        xa1.append(data[i][0])
        xb1.append(data[i][1])
        #y2.append(round(data[i][2],0))
        y2.append(data[i][2])
    lengths=[]
    for i in xa1:
        lengths.append(len(i.split()))
    for i in xb1:
        lengths.append(len(i.split()))
    maxlen = numpy.max(lengths)
    emb1,mas1=getmtr(xa1,maxlen)
    emb2,mas2=getmtr(xb1,maxlen)
    
    y2=np.array(y2,dtype=np.float32)
    return emb1,mas1,emb2,mas2,y2
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


# In[4]:

#new embed

def embed(stmx):
    #stmx=stmx.split()
    dmtr=numpy.zeros((stmx.shape[0],300),dtype=np.float32)
    count=0
    while(count<len(stmx)):
        if stmx[count]==',':
            count+=1
            continue
        if stmx[count] in dtr:
            dmtr[count]=model[dtr[stmx[count]]]
            count+=1
        else:
            dmtr[count]=model[stmx[count]]
            count+=1
    return dmtr


    


# In[5]:

def pfl(s):
    for i in dtr['syn'][0]:
        s.append(i)
    return s

def chsyn(s,trn):
    cnt=0
    global flg
    x2=s.split()
    x=[]
    
    for i in x2:
        x.append(i)
    for i in range(0,len(x)):
        q=x[i]
        mst=''
        if q not in d2:
            continue
        if flg==1:
            trn=pfl(trn)
            flg=0
        
        if q in cachedStopWords or q.title() in cachedStopWords or q.lower() in cachedStopWords:
            #print q,"skipped"
            continue
        if q in d2 or q.lower() in d2:
            if q in d2:
                mst=findsim(q)
            #print q,mst
            elif q.lower() in d2:
                mst=findsim(q)
            if q not in model:
                mst=''
                continue

        if mst in model:
            if q==mst:
                mst=''
                
                continue
            if model.similarity(q,mst)<0.6:
                continue
            #print x[i],mst
            x[i]=mst
            if q.find('ing')!=-1:
                if x[i]+'ing' in model:
                    x[i]+='ing'
                if x[i][:-1]+'ing' in model:
                    x[i]=x[i][:-1]+'ing'
            if q.find('ed')!=-1:
                if x[i]+'ed' in model:
                    x[i]+='ed'
                if x[i][:-1]+'ed' in model:
                    x[i]=x[i][:-1]+'ed'
            cnt+=1
            mst=''
    return ' '.join(x),cnt

def findsim(wd):
    syns=d2[wd]
    x=random.randint(0,len(syns)-1)
    return syns[x]

def check(sa,sb,dat):
    for i in dat:
        if sa==i[0] and sb==i[1]:
            return False
        if sa==i[1] and sb==i[0]:
            return False
    return True

def expand(data):
    n=[]
    for m in range(0,10):
        for i in data:
            sa,cnt1=chsyn(i[0],data)
            sb,cnt2=chsyn(i[1],data)
            if cnt1>0 and cnt2>0:
                l1=[sa,sb,i[2]]
                n.append(l1)
    print len(n)
    for i in n:
        if check(i[0],i[1],data):
            data.append(i)
    return data