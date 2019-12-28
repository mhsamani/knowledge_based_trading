#!/usr/bin/env python
# coding: utf-8


import ta
import pandas as pd
import numpy as np
from backtesting.test import SMA
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting import Backtest

'''nemad='GOLD'
df=pd.read_csv('C:/Users/Sina/desktop/csv/{}.csv'.format(nemad))
df.columns=['Date','Open','High','Low','Close','Volume']
df=df[['Date','Open','High','Low','Close','Volume']]
df['Date1']=pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='ignore')
df.set_index('Date1',inplace=True)'''


def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()


def RSI(valuse, n):

    return ta.rsi(pd.Series(values),n)

def MACD(values, fast = 12, slow = 26):
    
    return ta.trend.macd(pd.Series(values), n_fast= fast, n_slow= slow, fillna=False)

def WR(high, low, close):
    
    return ta.momentum.wr(pd.Series(high), pd.Series(low), pd.Series(close), lbp=14, fillna=False)

def CMF(high, low, close, volume):
    
    return ta.volume.chaikin_money_flow(pd.Series(high), pd.Series(low), pd.Series(close), pd.Series(volume), n = 20, fillna=True)

def PSAR(dates, high, low, close, iaf = 0.02, maxaf = 0.2):
    length = len(dates)
    dates = list(dates)
    high = list(high)
    low = list(low)
    close = list(close)
    psar = close[0:len(close)]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = iaf
    ep = low[0]
    hp = high[0]
    lp = low[0]
    for i in range(2,length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
        reverse = False
        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf
        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]
        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]
        
    
    sar_dic={"dates":dates, "high":high, "low":low, "close":close, "psar":psar, "psarbear":psarbear, "psarbull":psarbull}
    
    updown=[]
    
    for i in range(len(close)):
        if psar[i] > close[i] and psar[i-1] <= close[i-1]:
            updown.append(1)
        elif psar[i] < close[i] and psar[i-1] >= close[i-1]:
            updown.append(-1)
        else:
            updown.append(0)
    return pd.Series(updown, name='PSAR' , index=dates)


class SmaCross(Strategy):
    
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 10
    n2 = 30
    
    def init(self):
        # Precompute two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    
    def next(self):
        price = self.data.Close[-1]
        # If sma1 crosses above sma2, buy the asset
        if crossover(self.sma1, self.sma2) and not self.position:
            self.buy()

        # Else, if sma1 crosses below sma2, sell it
        elif crossover(self.sma2, self.sma1):
            self.position.close()

            
class SmaCross1(Strategy):
    
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 50
    n2 = 150
    
    def init(self):
        # Precompute two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    
    def next(self):
        price = self.data.Close[-1]
        # If sma1 crosses above sma2, buy the asset
        if self.sma1 > self.sma2 and not self.position:
            self.buy()

        # Else, if sma1 crosses below sma2, sell it
        elif self.sma2 > self.sma1:
            self.position.close()
            
            
class MACDCross(Strategy):
    
    def init(self):
        self.macd = self.I(MACD, self.data.Close)
            
    def next(self):
        if self.macd >0 and not self.position :
            self.buy()
            
        elif self.macd <=0 :
            self.position.close()

            
            

class WillamsR(Strategy):
    
    def init(self):
        self.WR = self.I(WR, self.data.High, self.data.Low, self.data.Close)
        
    def next(self):
        if self.WR <=-90 and not self.position:
            self.buy()
        elif self.WR >= -10 :
            self.position.close()
            
            
class ChaikinMoneyFlow(Strategy):
    
    def init(self):
        self.CMF = self.I(CMF, self.data.High, self.data.Low, self.data.Close, self.data.Volume)
        
    def next(self):
        if self.CMF <=-0.3 and not self.position:
            self.buy()
        elif self.CMF >= 0.3 :
            self.position.close()
            
            
            
class OverRSI(Strategy):
    n = 14
    
    def init(self):
        self.rsi = self.I(RSI, self.data.Close, self.n)
    
    def next(self) :
        if self.rsi > 70  :
            self.position.close()   
        elif self.rsi < 30 and not self.position:
            self.buy()

class OverRSITAHA(Strategy):
    n = 14
    
    def init(self):
        self.rsi = self.I(RSI, self.data.Close, self.n)
    
    def next(self):
        
        if self.rsi > 70 and not self.position :
            self.buy()
            
        elif self.rsi < 35 :
            self.position.close()
            
            
            
class SARSTRATEGY(Strategy):
    def init(self):
        self.sar = self.I(PSAR, self.data.index, self.data.High, self.data.Low, self.data.Close)
    
    def next(self):
        if self.sar == -1 and not self.position:
            self.buy()
            
        elif self.sar == 1:
            self.position.close()
            
strategy=[SmaCross,SmaCross1,MACDCross,WillamsR,OverRSI,OverRSITAHA,SARSTRATEGY]
report =[]
share_list=['لسرما','لبوتان','وتوس','فولاد','وتجارت','خودرو','ستران','آکنتور','زاگرس']

for j in share_list:
    try:
        connection = mysql.connector.connect(host='127.0.0.1',
                                             database='newtse',
                                             user='root',
                                             password='Sina12345',
                                             port='3306')
        nem = ('"{}"'.format(j))
        sql_select_Query = "SELECT * FROM `noavaran_d` where persian = "+nem

        cursor = connection.cursor()
        cursor.execute(sql_select_Query)
        records = cursor.fetchall()  
        date=[]
        adj_low=[]
        adj_high=[]
        adj_close=[]
        adj_open=[]
        nemad=[]
        volume=[]
        flag=[]

        for i in range(len(records)):
            date.append(records[i][0])
            nemad.append(records[i][10])
            adj_open.append(records[i][2])
            adj_high.append(records[i][3])
            adj_low.append(records[i][4])
            adj_close.append(records[i][5])
            volume.append(records[i][6])
            if records[i][3]==records[i][4]:
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
    
    df=pd.DataFrame(list(zip(date,nemad,adj_low,adj_high,adj_close,adj_open,volume,flag)),columns=['Date','nemad','Low','High','Close','Open','Volume','Flag']) 
    df=df[['Date','Open','High','Low','Close','Volume']]
    df['Date1']=pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='ignore')
    df.set_index('Date1',inplace=True)
    
    for i in strategy:
        print(str(i).split('.')[-1].split("'")[0])
        bt = Backtest(df, i, cash=100000, commission=0)
        stat=str(bt.run()).split('\n')
        start = stat[0].split(' ')[-2]
        end = stat[1].split(' ')[-2]
        Duration = stat[2].split(' ')[-3]
        Exposure = stat[3].split(' ')[-1]
        Equity_Final = stat[4].split(' ')[-1]
        Equity_Peak = stat[5].split(' ')[-1]
        Return_percent = stat[6].split(' ')[-1]
        Buy_Hold_Return_percent = stat[7].split(' ')[-1]
        Max_Drawdown_percent = stat[8].split(' ')[-1]
        Avg_Drawdown_percent = stat[9].split(' ')[-1]
        Max_Drawdown_Duration = stat[10].split(' ')[-3]
        Avg_Drawdown_Duration = stat[11].split(' ')[-3]
        Trades = stat[12].split(' ')[-1]
        Win_Rate = stat[13].split(' ')[-1]
        Best_Trade = stat[14].split(' ')[-1]
        Worst_Trade = stat[15].split(' ')[-1]
        Avg_Trade = stat[16].split(' ')[-1]
        Max_Trade_Duration = stat[17].split(' ')[-3]
        Avg_Trade_Duration = stat[18].split(' ')[-3]
        Expectancy = stat[19].split(' ')[-1]
        SQN = stat[20].split(' ')[-1]
        Sharpe_Ratio = stat[21].split(' ')[-1]
        Sortino_Ratio = stat[22].split(' ')[-1]
        Calmar_Ratio =  stat[23].split(' ')[-1]
        used_strategy = stat[24].split(' ')[-1]

        report.append([used_strategy,nemad[0],start,end,Duration,Exposure,Equity_Final,Equity_Peak,Return_percent,Buy_Hold_Return_percent,Max_Drawdown_percent,Avg_Drawdown_percent,Max_Drawdown_Duration,Avg_Drawdown_Duration,Trades,Win_Rate,Best_Trade,Worst_Trade,Avg_Trade,Max_Trade_Duration,Avg_Trade_Duration,Expectancy,SQN,Sharpe_Ratio,Calmar_Ratio])
        
        
pd.DataFrame(report ,columns=['used_strategy','nemad','start','end','Duration','Exposure','Equity_Final','Equity_Peak','Return_percent','Buy_Hold_Return_percent','Max_Drawdown_percent','Avg_Drawdown_percent','Max_Drawdown_Duration','Avg_Drawdown_Duration','Trades','Win_Rate','Best_Trade','Worst_Trade','Avg_Trade','Max_Trade_Duration','Avg_Trade_Duration','Expectancy','SQN','Sharpe_Ratio','Calmar_Ratio'])
df.to_csv('C:/Users/Sina/Desktop/report.csv',encoding="utf-8-sig")

