#!/usr/bin/env python
# coding: utf-8

# In[9]:


import model as mm


# In[13]:


# import some librariesw
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import supportingFunctions as sf
import model as mm 

tf.compat.v1.reset_default_graph()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True

#--------------------------------------------------------------
#% SET THESE PARAMETERS CAREFULLY
nLayers=5
epochs=50
batchSize=1
gradientMethod='AG'
K=1
sigma=0.01
restoreWeights=False
#%% to train the model with higher K values  (K>1) such as K=5 or 10,
# it is better to initialize with a pre-trained model with K=1.
if K>1:
    restoreWeights=True
    restoreFromModel='04Jun_0243pm_5L_1K_100E_AG'

if restoreWeights:
    wts=sf.getWeights('savedModels/'+restoreFromModel)
#--------------------------------------------------------------------------
# these are functions from mm #had trouble calling them before

# def createLayer(x, szW, trainning,lastLayer):
#     """
#     This function create a layer of CNN consisting of convolution, batch-norm,
#     and ReLU. Last layer does not have ReLU to avoid truncating the negative
#     part of the learned noise and alias patterns.
#     """
#     W=tf.compat.v1.get_variable('W',shape=szW,initializer=tf.contrib.layers.xavier_initializer())
#     x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#     xbn=tf.layers.batch_normalization(x,training=trainning,fused=True,name='BN')

#     if not(lastLayer):
#         return tf.nn.relu(xbn)
#     else:
#         return xbn

# ## drn func from model.py because its having trouble reading it
# def dw(inp,trainning,nLay):
#     """
#     This is the Dw block as defined in the Fig. 1 of the MoDL paper
#     It creates an n-layer (nLay) residual learning CNN.
#     Convolution filters are of size 3x3 and 64 such filters are there.
#     nw: It is the learned noise
#     dw: it is the output of residual learning after adding the input back.
#     """
#     lastLayer=False
#     nw={}
#     nw['c'+str(0)]=inp
#     szW={}
#     szW = {key: (3,3,64,64) for key in range(2,nLay)}
#     szW[1]=(3,3,2,64)
#     szW[nLay]=(3,3,64,2)

#     for i in np.arange(1,nLay+1):
#         if i==nLay:
#             lastLayer=True
#         with tf.compat.v1.variable_scope('Layer'+str(i)):
#             nw['c'+str(i)]=createLayer(nw['c'+str(i-1)],szW[i],trainning,lastLayer)

#     with tf.name_scope('Residual'):
#         shortcut=tf.identity(inp)
#         dw=shortcut+nw['c'+str(nLay)]
#     return dw

# # make model function 
# def makeModel(atb,csm,mask,training,nLayers,K,gradientMethod):
#     """
#     This is the main function that creates the model.

#     """
#     out={}
#     out['dc0']=atb
#     with tf.name_scope('myModel'):
#         with tf.compat.v1.variable_scope('Wts',reuse=tf.compat.v1.AUTO_REUSE):
#             for i in range(1,K+1):
#                 j=str(i)
#                 out['dw'+j]=dw(out['dc'+str(i-1)],training,nLayers)
#                 lam1=getLambda()
#                 rhs=atb + lam1*out['dw'+j]
#                 if gradientMethod=='AG':
#                     out['dc'+j]=dc(rhs,csm,mask,lam1)
#                 elif gradientMethod=='MG':
#                     if training:
#                         out['dc'+j]=dcManualGradient(rhs)
#                     else:
#                         out['dc'+j]=dc(rhs,csm,mask,lam1)
#     return out

#--------------------------------------

#%%Generate a meaningful filename to save the trainined models for testing
print ('*************************************************')
start_time=time.time()
saveDir='savedModels/'
cwd=os.getcwd()
directory=saveDir + datetime.now().strftime("%m-%d-%Y %H:%M%p")+ str(nLayers)+'L_'+str(K)+'K_'+str(epochs)+'E_'+gradientMethod

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName= directory+'/model'


#%% save test model
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

csmT = tf.compat.v1.placeholder(tf.complex64,shape=(None,12,256,232),name='csm')
maskT= tf.compat.v1.placeholder(tf.complex64,shape=(None,256,232),name='mask')
atbT = tf.compat.v1.placeholder(tf.float32,shape=(None,256,232,2),name='atb')

out= mm.makeModel(atbT,csmT,maskT,False,nLayers,K,gradientMethod) ## uses deleted package
predTst=out['dc'+str(K)]
predTst=tf.identity(predTst,name='predTst')
sessFileNameTst=directory+'/modelTst'

saver=tf.compat.v1.train.Saver()
with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    savedFile=saver.save(sess, sessFileNameTst,latest_filename='checkpointTst')
print ('testing model saved:' +savedFile)
#%% read multi-channel dataset
trnOrg,trnAtb,trnCsm,trnMask=sf.getData('training')
trnOrg,trnAtb=sf.c2r(trnOrg),sf.c2r(trnAtb)

#%%
tf.compat.v1.reset_default_graph()
csmP = tf.compat.v1.placeholder(tf.compat.v1.complex64,shape=(None,None,None,None),name='csm')
maskP= tf.compat.v1.placeholder(tf.compat.v1.complex64,shape=(None,None,None),name='mask')
atbP = tf.compat.v1.placeholder(tf.compat.v1.float32,shape=(None,None,None,2),name='atb')
orgP = tf.compat.v1.placeholder(tf.compat.v1.float32,shape=(None,None,None,2),name='org')


#%% creating the dataset
nTrn=trnOrg.shape[0]
nBatch= int(np.floor(np.float32(nTrn)/batchSize))
nSteps= nBatch*epochs

trnData = tf.compat.v1.data.Dataset.from_tensor_slices((orgP,atbP,csmP,maskP))
trnData = trnData.cache()
trnData=trnData.repeat(count=epochs)
trnData = trnData.shuffle(buffer_size=trnOrg.shape[0])
trnData=trnData.batch(batchSize)
trnData=trnData.prefetch(5)
iterator=trnData.make_initializable_iterator()
orgT,atbT,csmT,maskT = iterator.get_next('getNext')

#%% make training model

out= mm.makeModel(atbT,csmT,maskT,True,nLayers,K,gradientMethod) ## uses deleted package
predT=out['dc'+str(K)]
predT=tf.compat.v1.identity(predT,name='pred')
loss = tf.compat.v1.reduce_mean(tf.reduce_sum(tf.pow(predT-orgT, 2),axis=0))
tf.compat.v1.summary.scalar('loss', loss)
update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.compat.v1.name_scope('optimizer'):
    optimizer = tf.compat.v1.train.AdamOptimizer()
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.compat.v1.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    opToRun=optimizer.apply_gradients(capped_gvs)


#%% training code


print ('training started at', datetime.now().strftime("%m-%d-%Y %H:%M%p"))
print ('parameters are: Epochs:',epochs,' BS:',batchSize,'nSteps:',nSteps,'nSamples:',nTrn)

saver = tf.compat.v1.train.Saver(max_to_keep=100)
totalLoss,ep=[],0
lossT = tf.compat.v1.placeholder(tf.float32)
lossSumT = tf.compat.v1.summary.scalar("TrnLoss", lossT)

with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    if restoreWeights:
        sess=sf.assignWts(sess,nLayers,wts)

    feedDict={orgP:trnOrg,atbP:trnAtb, maskP:trnMask,csmP:trnCsm}
    sess.run(iterator.initializer,feed_dict=feedDict)
    savedFile=saver.save(sess, sessFileName)
    print("Model meta graph saved in::%s" % savedFile)

    writer = tf.compat.v1.summary.FileWriter(directory, sess.graph)
    for step in tqdm(range(nSteps)):
        try:
            tmp,_,_=sess.run([loss,update_ops,opToRun])
            totalLoss.append(tmp)
            if np.remainder(step+1,nBatch)==0:
                ep=ep+1
                avgTrnLoss=np.mean(totalLoss)
                lossSum=sess.run(lossSumT,feed_dict={lossT:avgTrnLoss})
                writer.add_summary(lossSum,ep)
                totalLoss=[] #after each epoch empty the list of total loos
        except tf.compat.v1.errors.OutOfRangeError:
            break
    savedfile=saver.save(sess, sessFileName,global_step=ep,write_meta_graph=True)
    writer.close()

end_time = time.time()
print ('Trianing completed in minutes ', ((end_time - start_time) / 60))
print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('*************************************************')

