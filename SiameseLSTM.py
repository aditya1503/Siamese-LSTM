
# coding: utf-8

# In[2]:




# In[1]:
import gensim
from gensim.models import word2vec
import pickle
import numpy as np
import numpy
import pickle
from random import *
import theano.tensor as T
def _p(pp, name):
    return '%s_%s' % (pp, name)
import re
from nltk.corpus import stopwords
import scipy.stats as meas

from gensim.models import word2vec


from collections import OrderedDict
import pickle as pkl
import random
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams



def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)
def zipp(params, tparams):
    
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def get_layer(name):
    fns = layers[name]
    return fns

# In[2]:

def genm(mu,sigma,n1,n2):
    return np.random.normal(mu,sigma,(n1,n2))

def getlayerx(d,pref,n,nin):
    mu=0.0
    sigma=0.2
    U = np.concatenate([genm(mu,sigma,n,n),genm(mu,sigma,n,n),genm(mu,sigma,n,n),genm(mu,sigma,n,n)])/np.sqrt(n)
    U=np.array(U,dtype=np.float32)
    W =np.concatenate([genm(mu,sigma,n,nin),genm(mu,sigma,n,nin),genm(mu,sigma,n,nin),genm(mu,sigma,n,nin)])/np.sqrt(np.sqrt(n*nin))
    W=np.array(W,dtype=np.float32)
    
    d[_p(pref, 'U')] = U
    #b = numpy.zeros((n * 300,))+1.5
    b = np.random.uniform(-0.5,0.5,size=(4*n,))
    b[n:n*2]=0.5
    d[_p(pref, 'W')] = W
    d[_p(pref, 'b')] = b.astype(config.floatX)
    return d

def creatrnnx():
    newp=OrderedDict()
    #print ("Creating neural network")
    newp=getlayerx(newp,'1lstm1',50,300)
    #newp=getlayerx(newp,'1lstm2',30,50)
    #newp=getlayerx(newp,'1lstm3',40,60)
    #newp=getlayerx(newp,'1lstm4',6)
    #newp=getlayerx(newp,'1lstm5',4)
    newp=getlayerx(newp,'2lstm1',50,300)
    #newp=getlayerx(newp,'2lstm2',20,10)
    #newp=getlayerx(newp,'2lstm3',10,20)
    #newp=getlayerx(newp,'2lstm4',6)
    #newp=getlayerx(newp,'2lstm5',4)
    #newp=getlayerx(newp,'2lstm3',4)
    #newp['2lstm1']=newp['1lstm1']
    #newp['2lstm2']=newp['1lstm2']
    #newp['2lstm3']=newp['1lstm3']
    return newp


# In[3]:


def dropout_layer(state_before, use_noise, rrng,rate):
    proj = tensor.switch(use_noise,
                         (state_before *rrng),
                         state_before * (1-rate))
    return proj

def getpl2(prevlayer,pre,mymask,used,rrng,size,tnewp):
    proj = lstm_layer2(tnewp, prevlayer, options,
                                        prefix=pre,
                                        mask=mymask,nhd=size)
    if used:
        print "Added dropout"
        proj = dropout_layer(proj, use_noise, rrng,0.5)
        
    return proj

def lstm_layer2(tparams, state_below, options, prefix='lstm', mask=None,nhd=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')].T)
        preact += x_
        preact += tparams[_p(prefix, 'b')]

        i = tensor.nnet.sigmoid(_slice(preact, 0, nhd))
        f = tensor.nnet.sigmoid(_slice(preact, 1, nhd))
        o = tensor.nnet.sigmoid(_slice(preact, 2, nhd))
        c = tensor.tanh(_slice(preact, 3, nhd))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return [h, c]

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')].T) +
                   tparams[_p(prefix, 'b')].T)
    #print "hvals"
    dim_proj = nhd
    [hvals,yvals], updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return hvals

def adadelta(lr, tparams, grads, emb11,mask11,emb21,mask21,y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, (0.95 * rg2 + 0.05* (g ** 2)))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([emb11,mask11,emb21,mask21,y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, (0.95 * ru2 + 0.05 * (ud ** 2)))
             for ru2, ud in zip(running_up2,updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

def sgd(lr, tparams, grads, emb11,mask11,emb21,mask21,y, cost):
    
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function([emb11,mask11,emb21,mask21,y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')
    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, emb11,mask11,emb21,mask21,y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([emb11,mask11,emb21,mask21,y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update
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



d2=pickle.load(open("synsem.p",'rb'))
dtr=pickle.load(open("dwords.p",'rb'))
#d2=dtr
#model=pickle.load(open("Semevalembed.p","rb"))


# In[7]:

## import pickle
#from random import shuffle


# In[9]:

prefix='lstm'
noise_std=0.
use_noise = theano.shared(numpy_floatX(0.))
flg=1
cachedStopWords=stopwords.words("english")
training=True #Loads best saved model if False
Syn_aug=True # If true, performs better on Test dataset but longer training time
print "Loading word2vec"
model = word2vec.Word2Vec.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz",binary=True)
options=locals().copy()


# In[ ]:



