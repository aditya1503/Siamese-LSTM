
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
from sentences import *


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
    b[n:n*2]=1.5
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




class lstm():    
    def __init__(self,nam,load=False,training=False):
        newp=creatrnnx()
        for i in newp.keys():
            if i[0]=='1':
                newp['2'+i[1:]]=newp[i]
        y = tensor.vector('y', dtype=config.floatX)
        mask11 = tensor.matrix('mask11', dtype=config.floatX)
        mask21 = tensor.matrix('mask21', dtype=config.floatX)
        emb11=theano.tensor.ftensor3('emb11')
        emb21=theano.tensor.ftensor3('emb21')
        if load==True:
            newp=pickle.load(open(nam,'rb'))
        tnewp=init_tparams(newp)
        trng = RandomStreams(1234)
        use_noise = theano.shared(numpy_floatX(0.))
    
        rate=0.5
        rrng=trng.binomial(emb11.shape,p=1-rate, n=1,dtype=emb11.dtype)
    
        proj11=getpl2(emb11,'1lstm1',mask11,False,rrng,50,tnewp)[-1]
        proj21=getpl2(emb21,'2lstm1',mask21,False,rrng,50,tnewp)[-1]
        dif=(proj21-proj11).norm(L=1,axis=1)
        s2=T.exp(-dif)
        sim=T.clip(s2,1e-7,1.0-1e-7)
        lr = tensor.scalar(name='lr')
        ys=T.clip((y-1.0)/4.0,1e-7,1.0-1e-7)
        cost=T.mean((sim - ys) ** 2)
        ns=emb11.shape[1]
        self.f2sim=theano.function([emb11,mask11,emb21,mask21],sim,allow_input_downcast=True)
        self.f_proj11=theano.function([emb11,mask11],proj11,allow_input_downcast=True)
        self.f_cost=theano.function([emb11,mask11,emb21,mask21,y],cost,allow_input_downcast=True)
        if training==True:
        
            gradi = tensor.grad(cost, wrt=tnewp.values())#/bts
            grads=[]
            l=len(gradi)
            for i in range(0,l/2):
                gravg=(gradi[i]+gradi[i+l/2])/(4.0)
            #print i,i+9
                grads.append(gravg)
            for i in range(0,len(tnewp.keys())/2):
                    grads.append(grads[i])
        
            self.f_grad_shared, self.f_update = adadelta(lr, tnewp, grads,emb11,mask11,emb21,mask21,y, cost)
    
    
    def train_lstm(self,train,max_epochs):
        print "Training"
        crer=[]
        cr=1.6
        freq=0
        batchsize=32
        dfreq=40#display frequency
        valfreq=800# Validation frequency
        lrate=0.0001
        precision=2
        for eidx in xrange(0,max_epochs):
            sta=time.time()
            #print self.chkterr2(test)
            num=len(train)
            nd=eidx
            sta=time.time()
            print 'Epoch',eidx
            rnd=sample(xrange(len(train)),len(train))
            for i in range(0,num,batchsize):
                q=[]
                x=i+batchsize
                if x>num:
                    x=num
                for z in range(i,x):
                    q.append(train[rnd[z]])
                #q=train[i:i+32]
                #shuffle(q)
                x1,mas1,x2,mas2,y2=prepare_data(q)
                
                ls=[]
                ls2=[]
                freq+=1
                use_noise.set_value(1.)
                for j in range(0,len(x1)):
                    ls.append(embed(x1[j]))
                    ls2.append(embed(x2[j]))
                trconv=np.dstack(ls)
                trconv2=np.dstack(ls2)
                emb2=np.swapaxes(trconv2,1,2)
                emb1=np.swapaxes(trconv,1,2)
               
                cst=self.f_grad_shared(emb2, mas2, emb1,mas1,y2)
                s=self.f_update(lrate)
                #s=f_update(lrate)
                if np.mod(freq,dfreq)==0:
                    print 'Epoch ', eidx, 'Update ', freq, 'Cost ', cst
            sto=time.time()
            print "epoch took:",sto-sta
        
    def chkterr2(self,mydata):
        count=[]
        num=len(mydata)
        px=[]
        yx=[]
        use_noise.set_value(0.)
        for i in range(0,num,256):
            q=[]
            x=i+256
            if x>num:
                x=num
            for j in range(i,x):
                q.append(mydata[j])
            x1,mas1,x2,mas2,y2=prepare_data(q)
            ls=[]
            ls2=[]
            for j in range(0,len(q)):
                ls.append(embed(x1[j]))
                ls2.append(embed(x2[j]))
            trconv=np.dstack(ls)
            trconv2=np.dstack(ls2)
            emb2=np.swapaxes(trconv2,1,2)
            emb1=np.swapaxes(trconv,1,2)
            pred=(self.f2sim(emb1,mas1,emb2,mas2))*4.0+1.0
            #dm1=np.ones(mas1.shape,dtype=np.float32)
            #dm2=np.ones(mas2.shape,dtype=np.float32)
            #corr=f_cost(emb1,mas1,emb2,mas2,y2)
            for z in range(0,len(q)):
                yx.append(y2[z])
                px.append(pred[z])
        #count.append(corr)
        px=np.array(px)
        yx=np.array(yx)
        #print "average error= "+str(np.mean(acc))
        return np.mean(np.square(px-yx)),meas.pearsonr(px,yx)[0],meas.spearmanr(yx,px)[0]
    def predict_similarity(self,sa,sb):
        q=[[sa,sb,0]]
        x1,mas1,x2,mas2,y2=prepare_data(q)
        ls=[]
        ls2=[]
        use_noise.set_value(0.)
        for j in range(0,len(x1)):
            ls.append(embed(x1[j]))
            ls2.append(embed(x2[j]))
        trconv=np.dstack(ls)
        trconv2=np.dstack(ls2)
        emb2=np.swapaxes(trconv2,1,2)
        emb1=np.swapaxes(trconv,1,2)
        return self.f2sim(emb1,mas1,emb2,mas2)

d2=pickle.load(open("synsem.p",'rb'))
dtr=pickle.load(open("dwords.p",'rb'))
#d2=dtr
#model=pickle.load(open("Semevalembed.p","rb"))




# In[9]:

prefix='lstm'
noise_std=0.
use_noise = theano.shared(numpy_floatX(0.))
flg=1
cachedStopWords=stopwords.words("english")
training=False #Loads best saved model if False
Syn_aug=True # If true, performs better on Test dataset but longer training time

options=locals().copy()


# In[ ]:



