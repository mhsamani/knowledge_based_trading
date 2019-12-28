from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
 
# TensorFlow and tf.keras
import keras
import tensorflow as tf

from keras.utils import multi_gpu_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.92
keras.backend.set_session(tf.Session(config=config))
#define number of gpus for keras
import subprocess
gpun = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')


# Helper libraries
import matplotlib.pyplot as plt
#print(tf.__version__)
###################     Load Data From CSVs
'''fr1=open('cp_testf.csv', "rt",encoding="utf-8")
tmparr1=fr1.readlines()
trltmp1=[]
trtmp1=[]
for i in range(len(tmparr1)//36):

    if  tmparr1[i*36+2].replace('\n','')=='Hold':
        trltmp1.append(int(1))
    if  tmparr1[i*36+2].replace('\n','')=='Buy':
        trltmp1.append(int(2))
    if  tmparr1[i*36+2].replace('\n','')=='Sell':
        trltmp1.append(int(0))
    tmp1 = []
    for j in range(1,16):
        results1 = [float(k) for k in tmparr1[i * 36+2*j+4].split(',')[1:16]]
        tmp1.append(results1)
    trtmp1.append(tmp1)
testlabel=np.array(trltmp1)
testset=np.array(trtmp1)'''


nemads="'فولاژ1'"
##load data from database
import mysql.connector
connection = mysql.connector.connect(host='192.168.2.102',database='tse',user='root',password='S@d3ghi#2019')
sql_select_Query = "SELECT * from indicators where nemad in ("+nemads+") AND date(date)> '2012-02-01'"
print(sql_select_Query)
cursor = connection.cursor()
cursor.execute(sql_select_Query)
records = cursor.fetchall()

dtestlables=[]
dtestset=[]
for i in records:
    if  i[7]=='Hold':
        dtestlables.append(int(1))
    if  i[7]=='Buy':
        dtestlables.append(int(2))
    if  i[7]=='Sell':
        dtestlables.append(int(0))

    dt=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    for j in range(0,15):
        for k in range(0,15):
            dt[j][k]=i[j*15+k+9]
    dtestset.append(dt)
testlabel=np.array(dtestlables)
testset=np.array(dtestset)


'''fr=open('cp_trainf.csv', "rt",encoding="utf-8")
tmparr=fr.readlines()
trltmp=[]
trtmp=[]
sbc=1
mc=2
fc=0
for i in range(len(tmparr)//36):

    if  tmparr[i*36+2].replace('\n','')=='Hold':
        fc=fc+1
        if fc!=mc:
            trltmp.append(int(1))
    if  tmparr[i*36+2].replace('\n','')=='Buy':
        for p in range(sbc):
            trltmp.append(int(2))
    if  tmparr[i*36+2].replace('\n','')=='Sell':
        for p in range(sbc):
            trltmp.append(int(0))
    tmp = []
    if fc!=mc:
        for j in range(1,16):
            results = [float(k) for k in tmparr[i * 36+2*j+4].split(',')[1:16]]
            tmp.append(results)

        if tmparr[i * 36 + 2].replace('\n', '') != 'Hold':
            for p in range(sbc):
                trtmp.append(tmp)
        else:
            trtmp.append(tmp)
    if fc==mc:
        fc=0

trainlabel=np.array(trltmp)
trainset=np.array(trtmp)'''

trnemads="'فملي1','فولاد1','فاسمين1'"
##load data from database
trconnection = mysql.connector.connect(host='192.168.2.102',database='tse',user='root',password='S@d3ghi#2019')
trsql_select_Query = "SELECT * from indicators where nemad in ("+trnemads+") AND date(date)> '2012-02-01'"
print(trsql_select_Query)
trcursor = trconnection.cursor()
trcursor.execute(trsql_select_Query)
trecords = trcursor.fetchall()

dtrainlables=[]
dtrainset=[]
for i in trecords:
    if  i[7]=='Hold':
        dtrainlables.append(int(1))
    if  i[7]=='Buy':
        dtrainlables.append(int(2))
    if  i[7]=='Sell':
        dtrainlables.append(int(0))

    dt=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    for j in range(0,15):
        for k in range(0,15):
            dt[j][k]=i[j*15+k+9]
    dtrainset.append(dt)
trainlabel=np.array(dtrainlables)
trainset=np.array(dtrainset)

trainset=trainset.reshape([-1,15,15,1])
testset=testset.reshape([-1,15, 15,1])

###dispose arrays
'''del tmp[:]
del tmp1[:]
del tmparr[:]
#del tmparr1[:]
del trltmp[:]
del trltmp1[:]
del trtmp[:]
del trtmp1[:]'''





###########             Saving Class Names
class_names = ['Sell','Hold','Buy']
bts=4096*4
index=0
max=0
loss=1
dindex=0
dloss=1
dacc=0
buc=3
loss_acc=[]
for i in range(0,201):
#############   build the model
    keras.backend.clear_session()
    np.random.seed(43)
    tf.compat.v1.random.set_random_seed(43)
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(15,15,1), padding='same'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(3, activation='softmax'))


    ###########     Compile the model/Optimize function
    if (gpun<2):
        parallel_model = model
    else:
        parallel_model = multi_gpu_model(model,gpus=gpun)
    parallel_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


    ###########     fit the model
    parallel_model.fit(trainset, trainlabel, epochs=i*buc,batch_size=bts)

    #####evaluate accuracyker
    test_loss, test_acc = parallel_model.evaluate(testset, testlabel)

    print('Test accuracy:', test_acc)
    print('index is: '+str(i*buc)+' with accuracy: '+str(test_acc))
    loss_acc.append([i*buc,test_acc,test_loss])
    if test_acc>max:
        max=test_acc
        index = i
        loss=test_loss
    if test_loss<dloss:
        dindex=i
        dloss=test_loss
        dacc=test_acc




print('best epoch number based on accuracy is: '+str(index*buc)+' with accuracy: '+str(max)+' with loss: '+str(loss))
print('best epoch number based on loss is: '+str(dindex*buc)+' with accuracy: '+str(dacc)+' with loss: '+str(dloss))

w=open('gp_epoch_eval.csv', "wt",encoding="utf-8")

w.write('best epoch number based on accuracy is: '+str(index*buc)+' with accuracy: '+str(max)+' with loss: '+str(loss)+'\n')
w.write('best epoch number based on loss is: '+str(dindex*buc)+' with accuracy: '+str(dacc)+' with loss: '+str(dloss)+'\n')
w.write('epochs,accuracy,loss\n')
for i in range(len(loss_acc)):
    w.write(str(loss_acc[i][0])+','+str(loss_acc[i][1])+','+str(loss_acc[i][2])+'\n')
w.close()
