"""
The following code was taken from the paper's GitHub respository. The purpose of this file is to train the models. It outputs a trained model that is 
saved under the savedModels folder to be used by testDemo.py. There were complications with running the code due to outdated packages and incorrect syntax. 
This code requires a lot of space and can only be run successfully through Google Colab. 
"""
###------------------------------------------------------------
# # the notebook was mounted to my current Google Drive and moved to my BIPN162 folder.
# # this was to make it so that this file could access all the other files it needed to run and that all of these files were in the same directory.
# from google.colab import drive
# drive.mount('/content/drive')
# %cd drive/MyDrive/BIPN162/

# # the tensorflow package had to be downgraded to version 1.14 because one module of tensorflow used in this file is not availible in any other versions
# %tensorflow_version 1.14

###-------------------------------------------------------------
# import some libraries
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

#%%Generate a meaningful filename to save the trainined models for testing
print ('*************************************************')
start_time=time.time()
saveDir='savedModels/'
cwd=os.getcwd()
directory=saveDir + datetime.now().strftime("%d-%b-%Y %I:%M %P") ## save the trained model in the savedModels folder and name it according to this format
if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName= directory+'/model'


#%% save test model
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

csmT = tf.compat.v1.placeholder(tf.complex64,shape=(None,12,256,232),name='csm')
maskT= tf.compat.v1.placeholder(tf.complex64,shape=(None,256,232),name='mask')
atbT = tf.compat.v1.placeholder(tf.float32,shape=(None,256,232,2),name='atb')

out= mm.makeModel(atbT,csmT,maskT,False,nLayers,K,gradientMethod) 
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
                totalLoss=[] #after each epoch empty the list of total loss
        except tf.compat.v1.errors.OutOfRangeError:
            break
    savedfile=saver.save(sess, sessFileName,global_step=ep,write_meta_graph=True)
    writer.close()

end_time = time.time()
print ('Trianing completed in minutes ', ((end_time - start_time) / 60))
print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('*************************************************')

