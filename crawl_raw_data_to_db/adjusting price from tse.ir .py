#!/usr/bin/env python
# coding: utf-8


import urllib.request
import pandas as pd
#import xlrd
#import mysql.connector
import os
print('Beginning file download with urllib2...')

df=pd.read_csv('company1.csv')
code_list=list(df['0'])

error=[]
code_list1=[]
for i in range(len(code_list)):
    if 'IRO' in code_list[i]:
        code_list1.append(code_list[i])
        
for i in range(1,len(code_list1)):
    try:
        url='http://new.tse.ir/archive/Trade/Cash/SymbolTrade/SymbolTrade_{}.xls'.format(code_list1[i])
        urllib.request.urlretrieve(url, 'xls\{}.xls'.format(code_list1[i]))
        print(code_list1[i])
    except:
        error.append(code_list1[i])
        print('error {}'.format(code_list1[i]))
        
        

        
        
        


# In[ ]:





# In[ ]:


files_list=os.listdir('xls')
for t in range(len(files_list)):
    data = pd.read_html('xls/{}'.format(files_list[t]))
    data=data[0]
    
    print(files_list[t])
    
    adjusted_list=[]
    for k in range(1,len(data)):
        if data['قیمت روز قبل'][k-1]!=data['مقدار قیمت پایانی'][k]:
            adjusted_list.append([data['تاريخ'][k-1],data['قیمت روز قبل'][k-1]/data['مقدار قیمت پایانی'][k],k])

    final=list(data['مقدار قیمت پایانی'])
    high=list(data['بیشترین'])
    low=list(data['کمترین'])
    close=list(data['مقدار آخرین قیمت'])

    for i in range(len(adjusted_list)-1,-1,-1):
        for j in range(adjusted_list[i][2],len(data)):
            final[j]*=adjusted_list[i][1]
            high[j]*=adjusted_list[i][1]
            low[j]*=adjusted_list[i][1]
            close[j]*=adjusted_list[i][1]


    data['adj_final']=final
    data['adj_high']=high
    data['adj_low']=low
    data['adj_close']=close
    data.to_csv('csv/{}.csv'.format(files_list[t][:-4]))


# In[ ]:





# In[ ]:




