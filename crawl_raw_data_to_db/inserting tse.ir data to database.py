import urllib.request
import pandas as pd
import xlrd
import mysql.connector
import datetime
import os

def jalali_to_gregorian(jy,jm,jd):
 if(jy>979):
  gy=1600
  jy-=979
 else:
  gy=621
 if(jm<7):
  days=(jm-1)*31
 else:
  days=((jm-7)*30)+186
 days+=(365*jy) +((int(jy/33))*8) +(int(((jy%33)+3)/4)) +78 +jd
 gy+=400*(int(days/146097))
 days%=146097
 if(days > 36524):
  gy+=100*(int(--days/36524))
  days%=36524
  if(days >= 365):
   days+=1
 gy+=4*(int(days/1461))
 days%=1461
 if(days > 365):
  gy+=int((days-1)/365)
  days=(days-1)%365
 gd=days+1
 if((gy%4==0 and gy%100!=0) or (gy%400==0)):
  kab=29
 else:
  kab=28
 sal_a=[0,31,kab,31,30,31,30,31,31,30,31,30,31]
 gm=0
 while(gm<13):
  v=sal_a[gm]
  if(gd <= v):
   break
  gd-=v
  gm+=1
 return [gy,gm,gd]


files_list=os.listdir('csv')
for t in range(0,len(files_list)):
    df = pd.read_csv('csv/{}'.format(files_list[t]))
    
    
    cnx = mysql.connector.connect(user='root', password='S@d3ghi#2019',host='192.168.2.102',database='tse')#,auth_plugin='mysql_native_password')
    cursor = cnx.cursor()

    query = ("INSERT INTO daily_data_copy2"
                   "(miladi ,date, nemad, persian, adj_low, adj_high, adj_close, adj_final, volume, value, transaction_number, maximum, minimum, final_price, final_price_change, final_price_percent, last_price, last_price_change, last_price_percent, yesterday_final_price, market_value ) "
                   "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s )")

    for r in range(len(df)):
        date=df['تاريخ'][r]

        jalalidatelist=df['تاريخ'][r].split('/')
        miladidatelist=jalali_to_gregorian(int(jalalidatelist[0]),int(jalalidatelist[1]),int(jalalidatelist[2]))
        miladi = datetime.datetime(miladidatelist[0],miladidatelist[1],miladidatelist[2])

        nemad=df['نماد'][r]
        persian=df['نام فارسی'][r]
        volume=float(df['حجم'][r])
        value=float(df['ارزش'][r])
        transaction_number=float(df['دفعات معامله'][r])
        maximum=float(df['بیشترین'][r])
        minimum=float(df['کمترین'][r])
        final_price=float(df['مقدار قیمت پایانی'][r])
        final_price_change=float(df['تغییر قیمت پایانی'][r])
        final_price_percent=float(df['درصد قیمت پایانی'][r])
        last_price=float(df['مقدار آخرین قیمت'][r])
        last_price_change=float(df['تغییر آخرین قیمت'][r])
        last_price_percent=float(df['درصد آخرین قیمت'][r])
        yesterday_final_price=float(df['قیمت روز قبل'][r])
        market_value=float(df['ارزش بازار'][r])
        adj_low=float(df['adj_low'][r])
        adj_high=float(df['adj_high'][r])
        adj_close=float(df['adj_close'][r])
        adj_final=float(df['adj_final'][r])

        values = (miladi, date, nemad, persian, adj_low, adj_high, adj_close, adj_final, volume, value, transaction_number, maximum, minimum, final_price, final_price_change, final_price_percent, last_price, last_price_change, last_price_percent, yesterday_final_price, market_value)
        print(r,values)
        cursor.execute(query, values)

    cnx.commit()    
    cnx.close()


# In[ ]:




