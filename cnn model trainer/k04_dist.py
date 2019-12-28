##### distributed keras
from __future__ import absolute_import, division, print_function, unicode_literals
import time

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
start_time = time.time()
# TensorFlow and tf.keras
import keras
import tensorflow as tf
#from tensorflow import keras
#import keras.utils.multi_gpu_model
#import keras.applications.Xception
#from keras.utils import multi_gpu_model
from keras.utils import multi_gpu_model

#define number of gpus for keras
import subprocess
gpun = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.92
keras.backend.set_session(tf.Session(config=config))
#server = tf.train.Server.create_local_server()
#sess = tf.Session(server.target)
#tf.keras.backend.set_session(sess)
#tf_config = tf.ConfigProto(allow_soft_placement=False)
#tf_config.gpu_options.allow_growth = True
#tf.config.optimizer_set_jit(True)
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
epc=1800
bts=4096*gpun
#keras.backend.clear_session()
np.random.seed(43)
#print(tf.__version__)
#from tensorflow import set_random_seed
tf.set_random_seed(43)

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

#############           Show PIC
'''plt.figure()
plt.imshow(trainset[0])
plt.colorbar()
plt.grid(False)
plt.show()'''

######### show top 25

'''plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainset[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[trainlabel[i]])
plt.show()'''

#############   build the model
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(15,15,1), padding='same'))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(3, activation='softmax'))
#model.summary()

###########     Compile the model/Optimize function
if (gpun<2):
    parallel_model = model
else:
    parallel_model = multi_gpu_model(model,gpus=gpun)
parallel_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

###########     fit the model
parallel_model.fit(trainset, trainlabel, epochs=epc,batch_size=bts)

#####evaluate accuracy
test_loss, test_acc =parallel_model.evaluate(testset, testlabel)

print('Test accuracy:', test_acc)


#############       make prediction


predictions = parallel_model.predict(testset)

###saving the model

parallel_model.save('cnn_'+str(epc)+'epochs.batchsize'+str(bts)+'.h5')

###output predictions

w=open('gp_prediction.csv', "wt",encoding="utf-8")
for i in range(len(dtestset)):
    w.write(str(records[i][0])+','+str(records[i][1])+','+str(records[i][2])+','+str(records[i][3])+','+str(records[i][4])+','+str(records[i][5])+','+str(records[i][6])+','+str(class_names[np.argmax(predictions[i])])+','+str(records[i][8])+','+str(predictions[i][0])+','+str(predictions[i][1])+','+str(predictions[i][2])+'\n')
w.close()

print("--- %s seconds ---" % (time.time() - start_time))

#######generate confusion matrix
'''conf1=[]
conf1.append([0,0,0,'Sell'])
conf1.append([0,0,0,'Hold'])
conf1.append([0,0,0,'Buy'])
for i in range(len(tmparr1)//36):
    if str(tmparr1[i * 36+2].replace('\n', ''))=='Sell':
        if class_names[np.argmax(predictions[i])]=='Sell':
            conf1[0][0]+=1
        if class_names[np.argmax(predictions[i])]=='Hold':
            conf1[0][1]+=1
        if class_names[np.argmax(predictions[i])]=='Buy':
            conf1[0][2]+=1
    if str(tmparr1[i * 36+2].replace('\n', ''))=='Hold':
        if class_names[np.argmax(predictions[i])]=='Sell':
            conf1[1][0]+=1
        if class_names[np.argmax(predictions[i])]=='Hold':
            conf1[1][1]+=1
        if class_names[np.argmax(predictions[i])]=='Buy':
            conf1[1][2]+=1
    if str(tmparr1[i * 36+2].replace('\n', ''))=='Buy':
        if class_names[np.argmax(predictions[i])]=='Sell':
            conf1[2][0]+=1
        if class_names[np.argmax(predictions[i])]=='Hold':
            conf1[2][1]+=1
        if class_names[np.argmax(predictions[i])]=='Buy':
            conf1[2][2]+=1
conf2=[]
conf2.append([0,0,0,'Precision'])
conf2.append([0,0,0,'Recall'])
conf2.append([0,0,0,'F_measure'])
#precision
try:
    conf2[0][0]=conf1[0][0]/(conf1[0][0]+conf1[1][0]+conf1[2][0])
except:
    conf2[0][0] = 1
try:
    conf2[0][1]=conf1[1][1]/(conf1[0][1]+conf1[1][1]+conf1[2][1])
except:
    conf2[0][1] = 1
try:
    conf2[0][2]=conf1[2][2]/(conf1[0][2]+conf1[1][2]+conf1[2][2])
except:
    conf2[0][2] = 1

#recall
conf2[1][0]=conf1[0][0]/(conf1[0][0]+conf1[0][1]+conf1[0][2])
conf2[1][1]=conf1[1][1]/(conf1[1][0]+conf1[1][1]+conf1[1][2])
conf2[1][2]=conf1[2][2]/(conf1[2][0]+conf1[2][1]+conf1[2][2])
#fmeasure

try:
    conf2[2][0]=2*(conf2[0][0]*conf2[1][0])/(conf2[0][0]+conf2[1][0])
except:
    conf2[2][0]=0
try:
    conf2[2][1] = 2 * (conf2[0][1] * conf2[1][1]) / (conf2[0][1] + conf2[1][1])
except:
    conf2[2][1]=0
try:
    conf2[2][2] = 2 * (conf2[0][2] * conf2[1][2]) / (conf2[0][2] + conf2[1][2])
except:
    conf2[2][2]=0

##total f1
f1=((conf2[2][0]*(conf1[0][0]+conf1[0][1]+conf1[0][2]))+(conf2[2][1]*(conf1[1][0]+conf1[1][1]+conf1[1][2]))+(conf2[2][2]*(conf1[2][0]+conf1[2][1]+conf1[2][2])))  /  (conf1[0][0]+conf1[0][1]+conf1[0][2]+conf1[1][0]+conf1[1][1]+conf1[1][2]+conf1[2][0]+conf1[2][1]+conf1[2][2])


#plotting confusion matrix
fig, axs =plt.subplots(2,1)
clust_data = conf1
collabel=("Sell", "Hold", "Buy", "prediction/Default")
#axs[0].axis('tight')
axs[0].axis('off')
axs[0].text(0.5,-0.1, "Total F-Measure is: "+str(f1), size=12, ha="center",transform=axs[0].transAxes)
the_table = axs[0].table(cellText=clust_data,colLabels=collabel,loc='center')

clust_data1 = conf2
collabel1=("Sell", "Hold", "Buy", "Label/Measure")
#axs[0].axis('tight')
axs[1].axis('off')
the_table1 = axs[1].table(cellText=clust_data1,colLabels=collabel1,loc='center')

plt.show()
'''

'''w1=open('tejarat_15testlabs.csv', "wt",encoding="utf-8")

for i in range(len(tmparr1)//36):
    w1.write(str(tmparr1[i * 36].replace('\n', ''))+','+str(tmparr1[i * 36+2].replace('\n', ''))+'\n')
w1.close()'''
'''
####evaluate graphs
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(3), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

###### plot ith image of test
'''

'''i = 100
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, testlabel, testset)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  testlabel)
plt.show()'''


##########  plot x images
'''num_rows = 15
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, testlabel, testset)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, testlabel)
plt.show()'''


'''
###########
r2=open('price_list.csv', "rt",encoding="utf-8")
strl=r2.readline()
price_list=[]
while strl!='':
    price_list.append(strl.split(','))
    strl=r2.readline()

def findprice(date,pricelist):
    for i in range(len(pricelist)):
        if date==price_list[i][0]:
            return price_list[i][1].replace('\n','')
    return ''

w=open('gp_tradecalc.csv', "wt",encoding="utf-8")
w.write('DATE,STATUS,CLOSE\n')
for i in range(len(tmparr1)//36):
    w.write(str(tmparr1[i * 36].replace('\n', '').split(',')[0])+','+str(class_names[np.argmax(predictions[i])])+','+str(findprice(str(tmparr1[i * 36].replace('\n', '').split(',')[0]),price_list))+'\n')
w.close()


########### algo trading
import pandas as pd
df=pd.read_csv('gp_tradecalc.csv')
timeframe=df.iloc[:,:]

Buy_commission = 0
Sell_commission = 0
bank = 10000000
shares = 0
wealth = []
buy = 0
sell = 0
hold = 0
xBuy = []
xSell = []
yBuy = []
ySell = []

days = len(timeframe)
start_price = timeframe['CLOSE'][0]
end_price = timeframe['CLOSE'][len(timeframe) - 1]

buy_hold_shares = int(bank / start_price)
buy_hold_bank = bank - buy_hold_shares * start_price

# In[107]:


for i in range(len(timeframe)):
    if timeframe.iloc[i, 1] == 'Buy' and shares == 0:
        share_price = timeframe.iloc[i, 2] * (1 + Buy_commission)
        shares = int(bank / share_price)
        bank = bank - int(bank / share_price) * share_price
        print('Date:{}\nbank:{}\nshares:{}\n'.format(timeframe.iloc[i, 0], bank, shares))
        wealth.append(bank + shares * share_price)
        xBuy.append(i)
        buy += 1

    elif timeframe.iloc[i, 1] == 'Sell' and shares != 0:
        share_price = timeframe.iloc[i, 2] * (1 - Sell_commission)
        bank = bank + share_price * shares
        shares = 0
        wealth.append(bank)
        print('Date:{}\nbank:{}\nshares:{}\n'.format(timeframe.iloc[i, 0], bank, shares))
        xSell.append(i)
        sell += 1
    else:
        share_price = timeframe.iloc[i, 2]
        wealth.append(bank + shares * share_price)
        hold += 1

for i in range(len(xBuy)):
    yBuy.append(wealth[xBuy[i]])

for i in range(len(xSell)):
    ySell.append(wealth[xSell[i]])

# In[108]:


buy_hold_wealth = []
for i in range(len(timeframe)):
    buy_hold_wealth.append(buy_hold_bank + buy_hold_shares * timeframe.iloc[i, 2])

# In[119]:


for x in xBuy:
    plt.axvline(x=x, color='gray')

for x in xSell:
    plt.axvline(x=x, color='gray')
plt.plot(wealth)
plt.plot(buy_hold_wealth, color='r')
plt.plot(xBuy, yBuy, '^', markersize=10, color='g')
plt.plot(xSell, ySell, 'v', markersize=10, color='r')
plt.show()

q=1
q=q+1
'''