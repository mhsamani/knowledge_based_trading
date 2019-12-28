import mysql.connector
import pandas as pd
from mysql.connector import Error
import ta
import talib
import pandas as pd
import numpy as np
import math
 
try:
    connection = mysql.connector.connect(host='192.168.2.102',database='tse',user='root',password='S@d3ghi#2019')
    nem = ('"فاسمين1"')
    sql_select_Query = "SELECT * FROM `daily_data_copy2` where nemad = " + nem
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    print("Total number of rows in Laptop is: ", cursor.rowcount)
    print("\nPrinting each laptop record")

    ddate = []
    adj_low = []
    adj_high = []
    adj_close = []
    adj_final = []
    nemad = []
    volume = []
    flag = []
    for i in range(len(records)):
        ddate.append(records[i][1])
        nemad.append(records[i][3])
        adj_low.append(records[i][4])
        adj_high.append(records[i][5])
        adj_close.append(records[i][6])
        adj_final.append(records[i][7])
        volume.append(records[i][8])
        if records[i][11] == records[i][12]:
            flag.append(1)
        else:
            flag.append(0)

except Error as e:
    print("Error reading data from MySQL table", e)
finally:
    if (connection.is_connected()):
        connection.close()
        cursor.close()
        print("MySQL connection is closed")

df = pd.DataFrame(list(zip(ddate, nemad, adj_low, adj_high, adj_close, adj_final, volume, flag)),
                  columns=['date', 'nemad', 'low', 'high', 'close', 'final', 'volume', 'flag'])



# SIMPLE LABELING 1
close = adj_close
result = [''] * len(close)
result_date = [''] * len(close)
windowsize = 11
counterrow = 0
minindex = 0
maxindex = 0
while counterrow < len(close):
    counterrow = counterrow + 1
    if (counterrow > windowsize):
        windowbeginindex = counterrow - windowsize - 1
        windowendindex = windowbeginindex + windowsize
        windowmiddleindex = (windowbeginindex + windowendindex - 1) / 2
        w = int(windowmiddleindex)
        m = close[windowbeginindex:windowendindex]
        sortedm = sorted(m)
        if close[w] == max(m):
            if close[w] - sortedm[0] > close[w] * 0.005:
                result[w] = 'Sell'
                result_date[w] = ddate[w]

        elif close[w] == min(m):
            if -close[w] + sortedm[-1] > close[w] * 0.005:
                result[w] = 'Buy'
                result_date[w] = ddate[w]

for i in range(len(result)):
    if result[i] == '':
        result[i] = 'Hold'
label = result
df['label'] = label

for i in range(6, 21):
    df['RSI{}'.format(i)] = talib.RSI(np.array(adj_close), i)
for i in range(6, 21):
    df['WR{}'.format(i)] = talib.WILLR(np.array(adj_high), np.array(adj_low), np.array(adj_close), i)
for i in range(6, 21):
    df['SMA{}'.format(i)] = (2 * (talib.SMA(np.array(adj_close), i) / adj_close) - 1) / 3
for i in range(6, 21):
    df['EMA{}'.format(i)] = (2 * (talib.EMA(np.array(adj_close), i) / adj_close) - 1) / 3
for i in range(6, 21):
    df['WMA{}'.format(i)] = (2 * (talib.WMA(np.array(adj_close), i) / adj_close) - 1) / 3
for i in range(6, 21):
    df['HMA{}'.format(i)] = (2 * (talib.WMA(2 * talib.WMA(np.array(adj_close), i) - talib.WMA(np.array(adj_close), 2 * i),math.sqrt(2 * i)) / adj_close) - 1) / 3
for i in range(6, 21):
    df['TEMA{}'.format(i)] = (2 * (talib.TEMA(np.array(adj_close)) / adj_close) - 1) / 3
for i in range(6, 21):
    df['CCI{}'.format(i)] = (talib.CCI(np.array(adj_high), np.array(adj_low), np.array(adj_close), i) + 400) / 800
for i in range(6, 21):
    df['CMO{}'.format(i)] = (talib.CMO(np.array(adj_close), i) + 100) / 200
for i in range(6, 21):
    df['MACD{}'.format(i)] = (talib.MACD(np.array(adj_close), fastperiod=i, slowperiod=2 * i + 2)[2] / adj_close) + 1 / 2
for i in range(6, 21):
    ppo = talib.PPO(np.array(adj_close), i, 2 * i + 2)
    df['PPO{}'.format(i)] = (ppo - talib.EMA(ppo, 9) / 100) + 1 / 2
for i in range(6, 21):
    df['ROC{}'.format(i)] = (talib.ROC(np.array(adj_close), i) + 100) / 400
for i in range(6, 21):
    df['CMFI{}'.format(i)] = (ta.chaikin_money_flow(pd.Series(adj_high), pd.Series(adj_low), pd.Series(adj_close), pd.Series(volume), n=i, fillna=False) + 1) / 2
for i in range(6, 21):
    df['DI{}'.format(i)] = (talib.PLUS_DI(np.array(adj_high), np.array(adj_low), np.array(adj_close),i) - talib.MINUS_DI(np.array(adj_high), np.array(adj_low),np.array(adj_close), i) + 100) / 200

df['SAR6'] = (talib.SAR(np.array(adj_close), np.array(adj_low))/adj_close)/2
for i in range(7, 21):
    sar = (talib.SAR(np.array(adj_close), np.array(adj_low))/adj_close)/2
    df['SAR{}'.format(i)] = [1] * (i - 6) + list(sar[:-i+6])







df=df.fillna(0)

cnx = mysql.connector.connect(user='root', password='S@d3ghi#2019',host='192.168.2.102',database='tse')
cursor = cnx.cursor()

query = ("INSERT INTO indicators"
               "(date, nemad, adj_low, adj_high, adj_close, adj_final, volume, label, flag, RSI6, RSI7, RSI8, RSI9, RSI10, RSI11, RSI12, RSI13, RSI14, RSI15, RSI16, RSI17, RSI18, RSI19, RSI20, WR6, WR7, WR8, WR9, WR10, WR11, WR12, WR13, WR14, WR15, WR16, WR17, WR18, WR19, WR20, WMA6, WMA7, WMA8, WMA9, WMA10, WMA11, WMA12, WMA13, WMA14, WMA15, WMA16, WMA17, WMA18, WMA19, WMA20, EMA6, EMA7, EMA8, EMA9, EMA10, EMA11, EMA12, EMA13, EMA14, EMA15, EMA16, EMA17, EMA18, EMA19, EMA20, SMA6, SMA7, SMA8, SMA9, SMA10, SMA11, SMA12, SMA13, SMA14, SMA15, SMA16, SMA17, SMA18, SMA19, SMA20, HMA6, HMA7, HMA8, HMA9, HMA10, HMA11, HMA12, HMA13, HMA14, HMA15, HMA16, HMA17, HMA18, HMA19, HMA20, TEMA6, TEMA7, TEMA8, TEMA9, TEMA10, TEMA11, TEMA12, TEMA13, TEMA14, TEMA15, TEMA16, TEMA17, TEMA18, TEMA19, TEMA20, CCI6, CCI7, CCI8, CCI9, CCI10, CCI11, CCI12, CCI13, CCI14, CCI15, CCI16, CCI17, CCI18, CCI19, CCI20, CMO6, CMO7, CMO8, CMO9, CMO10, CMO11, CMO12, CMO13, CMO14, CMO15, CMO16, CMO17, CMO18, CMO19, CMO20, MACD6, MACD7, MACD8, MACD9, MACD10, MACD11, MACD12, MACD13, MACD14, MACD15, MACD16, MACD17, MACD18, MACD19, MACD20, PPO6, PPO7, PPO8, PPO9, PPO10, PPO11, PPO12, PPO13, PPO14, PPO15, PPO16, PPO17, PPO18, PPO19, PPO20, ROC6, ROC7, ROC8, ROC9, ROC10, ROC11, ROC12, ROC13, ROC14, ROC15, ROC16, ROC17, ROC18, ROC19, ROC20, CMFI6, CMFI7, CMFI8, CMFI9, CMFI10, CMFI11, CMFI12, CMFI13, CMFI14, CMFI15, CMFI16, CMFI17, CMFI18, CMFI19, CMFI20, ADX6, ADX7, ADX8, ADX9, ADX10, ADX11, ADX12, ADX13, ADX14, ADX15, ADX16, ADX17, ADX18, ADX19, ADX20, SAR6, SAR7, SAR8, SAR9, SAR10, SAR11, SAR12, SAR13, SAR14, SAR15, SAR16, SAR17, SAR18, SAR19, SAR20) "
               "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")

for r in range(61,len(df)):
    date=df['date'][r]
    nemad=df['nemad'][r]
    adj_low=float(df['low'][r])
    adj_high=float(df['high'][r])
    adj_close=float(df['close'][r])
    adj_final=float(df['final'][r])
    volume=float(df['volume'][r])
    label=str(df['label'][r])
    flag=int(df['flag'][r])


    values = (date, nemad, adj_low, adj_high, adj_close, adj_final, volume, label, flag, df['RSI6'][r], df['RSI7'][r], df['RSI8'][r], df['RSI9'][r], df['RSI10'][r], df['RSI11'][r], df['RSI12'][r], df['RSI13'][r], df['RSI14'][r], df['RSI15'][r], df['RSI16'][r], df['RSI17'][r], df['RSI18'][r], df['RSI19'][r], df['RSI20'][r], df['WR6'][r], df['WR7'][r], df['WR8'][r], df['WR9'][r], df['WR10'][r], df['WR11'][r], df['WR12'][r], df['WR13'][r], df['WR14'][r], df['WR15'][r], df['WR16'][r], df['WR17'][r], df['WR18'][r], df['WR19'][r], df['WR20'][r], df['WMA6'][r], df['WMA7'][r], df['WMA8'][r], df['WMA9'][r], df['WMA10'][r], df['WMA11'][r], df['WMA12'][r], df['WMA13'][r], df['WMA14'][r], df['WMA15'][r], df['WMA16'][r], df['WMA17'][r], df['WMA18'][r], df['WMA19'][r], df['WMA20'][r], df['EMA6'][r], df['EMA7'][r], df['EMA8'][r], df['EMA9'][r], df['EMA10'][r], df['EMA11'][r], df['EMA12'][r], df['EMA13'][r], df['EMA14'][r], df['EMA15'][r], df['EMA16'][r], df['EMA17'][r], df['EMA18'][r], df['EMA19'][r], df['EMA20'][r], df['SMA6'][r], df['SMA7'][r], df['SMA8'][r], df['SMA9'][r], df['SMA10'][r], df['SMA11'][r], df['SMA12'][r], df['SMA13'][r], df['SMA14'][r], df['SMA15'][r], df['SMA16'][r], df['SMA17'][r], df['SMA18'][r], df['SMA19'][r], df['SMA20'][r], df['HMA6'][r], df['HMA7'][r], df['HMA8'][r], df['HMA9'][r], df['HMA10'][r], df['HMA11'][r], df['HMA12'][r], df['HMA13'][r], df['HMA14'][r], df['HMA15'][r], df['HMA16'][r], df['HMA17'][r], df['HMA18'][r], df['HMA19'][r], df['HMA20'][r], df['TEMA6'][r], df['TEMA7'][r], df['TEMA8'][r], df['TEMA9'][r], df['TEMA10'][r], df['TEMA11'][r], df['TEMA12'][r], df['TEMA13'][r], df['TEMA14'][r], df['TEMA15'][r], df['TEMA16'][r], df['TEMA17'][r], df['TEMA18'][r], df['TEMA19'][r], df['TEMA20'][r], df['CCI6'][r], df['CCI7'][r], df['CCI8'][r], df['CCI9'][r], df['CCI10'][r], df['CCI11'][r], df['CCI12'][r], df['CCI13'][r], df['CCI14'][r], df['CCI15'][r], df['CCI16'][r], df['CCI17'][r], df['CCI18'][r], df['CCI19'][r], df['CCI20'][r], df['CMO6'][r], df['CMO7'][r], df['CMO8'][r], df['CMO9'][r], df['CMO10'][r], df['CMO11'][r], df['CMO12'][r], df['CMO13'][r], df['CMO14'][r], df['CMO15'][r], df['CMO16'][r], df['CMO17'][r], df['CMO18'][r], df['CMO19'][r], df['CMO20'][r], df['MACD6'][r], df['MACD7'][r], df['MACD8'][r], df['MACD9'][r], df['MACD10'][r], df['MACD11'][r], df['MACD12'][r], df['MACD13'][r], df['MACD14'][r], df['MACD15'][r], df['MACD16'][r], df['MACD17'][r], df['MACD18'][r], df['MACD19'][r], df['MACD20'][r], df['PPO6'][r], df['PPO7'][r], df['PPO8'][r], df['PPO9'][r], df['PPO10'][r], df['PPO11'][r], df['PPO12'][r], df['PPO13'][r], df['PPO14'][r], df['PPO15'][r], df['PPO16'][r], df['PPO17'][r], df['PPO18'][r], df['PPO19'][r], df['PPO20'][r], df['ROC6'][r], df['ROC7'][r], df['ROC8'][r], df['ROC9'][r], df['ROC10'][r], df['ROC11'][r], df['ROC12'][r], df['ROC13'][r], df['ROC14'][r], df['ROC15'][r], df['ROC16'][r], df['ROC17'][r], df['ROC18'][r], df['ROC19'][r], df['ROC20'][r], df['CMFI6'][r], df['CMFI7'][r], df['CMFI8'][r], df['CMFI9'][r], df['CMFI10'][r], df['CMFI11'][r], df['CMFI12'][r], df['CMFI13'][r], df['CMFI14'][r], df['CMFI15'][r], df['CMFI16'][r], df['CMFI17'][r], df['CMFI18'][r], df['CMFI19'][r], df['CMFI20'][r], df['DI6'][r], df['DI7'][r], df['DI8'][r], df['DI9'][r], df['DI10'][r], df['DI11'][r], df['DI12'][r], df['DI13'][r], df['DI14'][r], df['DI15'][r], df['DI16'][r], df['DI17'][r], df['DI18'][r], df['DI19'][r], df['DI20'][r], df['SAR6'][r], df['SAR7'][r], df['SAR8'][r], df['SAR9'][r], df['SAR10'][r], df['SAR11'][r], df['SAR12'][r], df['SAR13'][r], df['SAR14'][r], df['SAR15'][r], df['SAR16'][r], df['SAR17'][r], df['SAR18'][r], df['SAR19'][r], df['SAR20'][r])
    a = [date, nemad, adj_low, adj_high, adj_close, adj_final, volume, label, flag]
    for i in range(9, len(values)):
        a.append(float(values[i]))

    values = tuple(a)

    print(r,values)
    cursor.execute(query, values)

cnx.commit()
cnx.close()
