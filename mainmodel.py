from siameseLSTM import *
prefix='lstm'
noise_std=0.
training=False #Loads best saved model if False
#model = word2vec.Word2Vec.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz",binary=True)
options=locals().copy()
newp=creatrnnx()
for i in newp.keys():
    if i[0]=='1':
        newp['2'+i[1:]]=newp[i]
y = tensor.vector('y', dtype=config.floatX)
mask11 = tensor.matrix('mask11', dtype=config.floatX)
mask21 = tensor.matrix('mask21', dtype=config.floatX)
emb11=theano.tensor.ftensor3('emb11')
emb21=theano.tensor.ftensor3('emb21')
if training==True:
    newp=pickle.load(open("bestsem.p",'rb'))
tnewp=init_tparams(newp)
trng = RandomStreams(1234)
use_noise = theano.shared(numpy_floatX(0.))

rate=0.5
rrng=trng.binomial(emb11.shape,p=1-rate, n=1,dtype=emb11.dtype)

proj11=getpl2(emb11,'1lstm1',mask11,False,rrng,50)[-1]
proj21=getpl2(emb21,'2lstm1',mask21,False,rrng,50)[-1]
dif=(proj21-proj11).norm(L=1,axis=1)
s2=T.exp(-dif)
sim=T.clip(s2,1e-7,1.0-1e-7)
lr = tensor.scalar(name='lr')
ys=T.clip((y-1.0)/4.0,1e-7,1.0-1e-7)
cost=T.mean((sim - ys) ** 2)
ns=emb11.shape[1]
f2sim=theano.function([emb11,mask11,emb21,mask21],sim,allow_input_downcast=True)
f_cost=theano.function([emb11,mask11,emb21,mask21,y],cost,allow_input_downcast=True)
if training==True:
    gradi = tensor.grad(cost, wrt=tnewp.values())#/bts
    grads=[]
    l=len(gradi)
    for i in range(0,l/2):
        gravg=(gradi[i]+gradi[i+l/2])/(2.0)
        #print i,i+9
        grads.append(gravg)
    for i in range(0,len(tnewp.keys())/2):
        grads.append(grads[i])





    f_grad_shared, f_update = adadelta(lr, tnewp, grads,
                                       emb11,mask11,emb21,mask21,y, cost)



