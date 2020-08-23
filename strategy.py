# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 08:47:17 2018

@author: sonng
"""
from finance_util import symbol_to_path, get_info_stock, get_info_stock_cp68_mobile, get_info_stock_bsc
import numpy as np
import pandas as pd
import talib
import datetime
from collections import Counter
import yfinance as yf

from pandas_datareader import data as pdr
from alpha_vantage.timeseries import TimeSeries

yf.pdr_override()

def run_backtest(df, ticker, trade = 'Long'):
    if (trade == 'Long'):
        df['Buy'] = df['Long']
    else:
        if (trade == 'Bottom'):
            df['Buy'] = df['Bottom']
        if (trade == 'MarkM'):
            df['Buy'] = df['MarkM']    
        if (trade == 'Short'):
            df['Buy'] = df['Short']
        if (trade == 'ShortTerm'):
            df['Buy'] = df['ShortTerm']
        if (trade == 'Breakout'):
            df['Buy'] = df['Breakout']
        if (trade == 'LongShortTrend'):
            df['Buy'] = df['LongShortTrend']  
        if (trade == 'Sideway'):
            df['Buy'] = df['Sideway'] 
        if (trade == 'EarlySignal'):
            df['Buy'] = df['EarlySignal'] 
        if (trade == 'SidewayBreakout'):
            df['Buy'] = df['SidewayBreakout'] 
            
    df['5Days'] = df['Close'].shift(-5)
    df['10Days'] = df['Close'].shift(-10)
    df['Back_test'] = 1* (df['Buy'] & (df['10Days'] > df['Close']) & (df['5Days'] > df['Close']) ) + -1* (df['Buy'] & (df['10Days'] <= df['Close'])& (df['5Days'] <= df['Close']))
    vals = df['Back_test'] .values.tolist()
    str_vals = [str(i) for i in vals]
    print('Back test result:', Counter(str_vals), 'symbol: ', ticker)
    print('***************************************************************')

def get_statistic_index(days, start, end, update = False, source = "cp68", exchange = "ALL"):
    
    benchmark = []
    if (exchange == "ALL"):
        benchmark = ["^VNINDEX", "^HASTC", "^UPCOM"]
    else:
        benchmark = [exchange]
   
    for ticker in benchmark:
        try:
            print(' Index information: ', ticker)
            df = process_data(ticker = ticker, start = start, end = end, realtime = update, source = source)
            print('  Actual Close/Low/High/Open: ', df['Close'].iloc[-days], df['Low'].iloc[-days], df['High'].iloc[-days], df['Open'].iloc[-days])
            print('  PCT_Change last 3 days: ', round(100*df['PCT_Change'].iloc[-days-2],2),round(100*df['PCT_Change'].iloc[-days-1],2), round(100*df['PCT_Change'].iloc[-days],2))
            print('  Volume/volume(MA30) ratio: ', round(df['Volume'].iloc[-days]/df['VolMA30'].iloc[-days],2))
            print('  RSI indicator: ', df['RSI'].iloc[-days])
            print('  Rate of change last 3 days: ', df['ROC'].iloc[-days]) 
            
            if ((df['Close'].iloc[-days] > df['EMA18'].iloc[-days] > df['EMA50'].iloc[-days])):
                print('  Market UPTREND!')
            else:
                if ((df['Close'].iloc[-days] < df['EMA18'].iloc[-days] < df['EMA50'].iloc[-days])):
                    print('  Market DOWNTREND!')
                else:
                    print('  Market UNSTABLE!')
            if (df['MACD_UP'].iloc[-days]):
                print('  Momentum UP')
            else:
                print('  Momentum DOWN')
            print('----------------------------------------------------------------')
        except Exception as e:
            print (e)
            print("Error in reading symbol: ", ticker)
            pass



def mean_reversion(ticker, start, end, realtime = False, source = "cp68", market = None):

    df = process_data(ticker = ticker, start = start, end = end, realtime = realtime, source = source)
    
    n_fast = 3
    n_slow = 6
    nema = 20
    df['MACD_3_6'], df['MACDSign20'], _= compute_MACD(df, n_fast, n_slow, nema)
    
    
    n_fast = 12
    n_slow = 26
    nema = 9
    df['MACD_12_26'], df['MACDSign9'], _ = compute_MACD(df, n_fast, n_slow, nema)
    
    
    df['Long']= ((df['Close'] > 1.02*df['Close'].shift(1)) & (df['Close'] > df['Open'])
    & (df['MACD_3_6'].shift(1) < df['MACDSign20'].shift(1)) & (df['RSI'] >=50)
    & ((df['MACD_3_6'] > df['MACDSign20']) | ((df['MACD_3_6'] > 0)  & (df['MACD_3_6'].shift(1) <0)))
    & ((df['Close']*df['Volume'] >= 3000000) | ((df['Volume'] > 1.3*df['VolMA30']) |(df['Volume'] > 250000))))
    
    
    # df['Canslim'] = (((df['MACD_12_26'] > df['MACDSign9']) | (df['MACD_12_26'] > 0.85*df['MACDSign9'])) & \
    # (df['Close']> 1.02*df['Close'].shift(1)) & (df['Close'] > df['Open']) & \
    ## (df['Close'] >= df['MID']) & (1.05*df['Close'].shift(2) > df['Close'].shift(1)) & \
    # ((df['Close']*df['Volume'] >= 3000000)) & (df['RSI'] >=50) &\
    # (((df['Volume'] > 1.3*df['VolMA30']) |(df['Volume'] > 500000)) & ((df['Volume'].shift(1) < df['Volume']) | (0.95*df['Volume'].shift(1) < df['Volume'] ))) & \
    # (df['Close'] > df['SMA30']) & ((df['Close']> df['Max6M']) | (df['Close']> df['Max3M']) |(df['Close']> df['Max4D'])))
    #
    
    df['Signal'] = 1*(df['Long'])
    
    hm_days = 2
    back_test = False
    for i in range(1,hm_days+1):
        if (df['Long'].iloc[-i]):
            print(" Mean reversion trading ", str(i), "days before ", df.iloc[-i].name , ticker)
            print_statistic(df, i)
            back_test = True
            if (market != None):
                    get_statistic_index(i, start, end, update = False, source = "cp68", exchange = market)

    if back_test:
        run_backtest(df, ticker)
        
    return df

def hung_canslim(ticker, start, end, realtime = False, source = "cp68", market = None , ndays = 2, typetrade = 'Long'):
    
    df = process_data(ticker = ticker, start = start, end = end, realtime = realtime, source = source)
    
    
    df['SMA15'] = df['Close'].rolling(window=15).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA30'] = df['Close'].rolling(window=30).mean()
    
    
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA150'] = df['Close'].rolling(window=150).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
#    n_fast = 12
#    n_slow = 26
#    nema = 9
#    df['MACD_12_26'], df['MACDSign9'], _ = compute_MACD(df, n_fast, n_slow, nema)
    
    
    
    df['Max10D'] = df['Close'].shift(1).rolling(window = 10).max()
    
    
    
    df['MID'] = (df['High'] + df['Low']) /2
#    print('Max 120 days :', max_120)
#    & (df['Close'] > (df['High'] + df['Low'])/2)
    # df['Volume'] > df['Volume'].shift(1)
    
#    
#     HHV(C,5) <1.055* LLV(C,5)
#    AND C * V >= 3000000 
#    AND C*V < 500000000
#    
#    
#    AND MA(V,30)>=50000
#    AND Ref(V,-5)>=50000
#    AND Ref(V,-10)>=50000
#    AND Ref(V,-20)>=50000
#    AND RSI(14)>=40
   
    
    
    df['Long'] = ((df['Close']> 1.02*df['Close'].shift(1)) & (df['Close'] > df['Open'])  & \
                  (df['Close'] >= (df['High'] + df['Low'])/2)  &\
                 (1.05*df['Close'].shift(2) >= df['Close'].shift(1)) & (df['Volume'] >= df['Volume'].shift(1)) &\
                 ((df['Close']*df['Volume'] >= 3E6)) & (df['RSI'] >=50) &\
                 (((df['Volume'] >= 1.3*df['VolMA15']) |(df['Volume'] > 2*250000))) &\
                 (df['Close'] >= df['SMA30']) & ((df['Close']> df['Max6M']) | (df['Close']> df['Max3M']) |(df['Close']>= df['High4D'])) &\
                 (df['PCT_HL'] <= 15))
    
    df['LongShortTrend'] = ((df['Close']> 1.02*df['Close'].shift(1)) & (df['Close'] >= df['Open'])  & \
                  (df['Close'] >= (df['High'] + df['Low'])/2)  &\
                 (1.05*df['Close'].shift(2) >= df['Close'].shift(1)) & (df['Volume'] >= df['Volume'].shift(1)) &\
                 ((df['Close']*df['Volume'] >= 3E6)) &\
                 (((df['Volume'] >= 1.3*df['VolMA15']) |(df['Volume'] > 2*250000))) &\
                 (df['Close'] >= df['SMA30']) & ((df['Close']>= df['High3D']) | (df['Close']>= df['High4D'])) &\
                 (df['PCT_HL'] <= 30))    
        
    df['T4'] = ((df['Close']> df['Close'].shift(1)) & (df['Close'] > df['Close'].shift(4))  &\
                  (df['Close'] > df['Close'].shift(2))  & (df['Close'] > df['Close'].shift(3)) &\
                 (df['PCT_HL'] <= 50)) 
        
    df['MA30'] = ((df['Close']> 1.015*df['Close'].shift(1)) & (df['Close'] >= df['Open']) & (df['Close']*df['Volume'] >= 3E6) & (df['Close'] >= df['SMA30'])  & ((df['Close'] - df['SMA30'])/df['SMA30'] <= 0.05) &\
                 (df['PCT_HL'] <= 50))  
        
    df['Sideway'] = ((df['Close']> 1.015*df['Close'].shift(1)) & (df['Close'] > df['Close'].shift(4))  &\
                  (df['Close'] > df['Close'].shift(2))  & (df['Close'] > df['Close'].shift(3)) &\
                  (df['Close']*df['Volume'] >= 3E6) & (df['Close'] >= df['SMA30'])  & ((df['Close'] - df['SMA30'])/df['SMA30'] <= 0.05) &\
                    (df['PCT_HL'] <= 50)) 
        
    df['ROC4'] = talib.ROC(df['Close'].values, timeperiod = 4)
    
    df['MarkM'] = ((df['Close']> 1.02*df['Close'].shift(1)) & (df['Close'] > df['Open'])  & \
                  (df['Close'] > (df['High'] + df['Low'])/2)  &\
                 (1.05*df['Close'].shift(2) >= df['Close'].shift(1)) & (df['Volume'] >= df['Volume'].shift(1)) &\
                 ((df['Close']*df['Volume'] >= 3E6)) & (df['RSI'] >=50) &\
                 (((df['Volume'] >= 1.3*df['VolMA30']) |(df['Volume'] > 2*250000))) &\
                 ((df['Close'] >= df['SMA50']) & (df['SMA50']>= df['SMA150']) & (df['SMA150']>= df['SMA200']) & \
                 (df['Close']>= 1.25*df['Min12M']) & (df['Close']>= 0.75*df['Max12M']) &\
                 (df['PCT_HL'] <= 15)))
   
    df['MarkM_tickers'] = (((df['Close'] >= df['SMA50']) & (df['SMA50']>= df['SMA150']) & (df['SMA150']>= df['SMA200']) & \
                 (df['Close']>= 1.25*df['Min12M']) & (df['Close']>= 0.75*df['Max12M'])))
    
    df['Breakout'] = ((df['Close']*df['Volume'] >= 3E6) & (df['ValueMA30']> 1E6) &\
                  (df['Close'] > 1.02*df['Close'].shift(1))  & (df['Close'] > df['Open']) &\
                 (df['PCT_HL'] < 15) & (df['Volume'] >= 1.3*df['VolMA30']) & \
                 (df['High15D'] > 1.05*df['Low15D'])  & \
                 ((df['Close'] > df['SMA30']) & (df['Close']>= df['High15D'])))
    
    df['EarlySignal'] = ((df['Close'] >= 1.01* df['Close'].shift(1))  &\
                         (df['Close'].shift(1) < 1.05* df['Close'].shift(2)) & (df['RSI'] >=50) & \
                      (df['RSI'] > df['RSI'].shift(1)) & (df['Volume'] >= 0.5*df['VolMA15']))
  
    df['SidewayBreakout'] = (((df['Close']-df['Low5D'])/df['Low5D'] <=0.1) & (df['Close'] > df['SMA30'])  &\
                            (df['Max5D'] < 1.07* df['Min5D']) & (df['Max10D'] < 1.07* df['Min10D']))
  
    
#    & ((df['Close']> df['Max6M']) | (df['Close']> df['Max3M']))
    df['Bottom'] = ((df['Close'] > df['Open']) & (df['Close']*df['Volume'] > 1E7) & 
#                  ((df['RSI'] < 31) | ((df['RSI'].shift(1) < df['RSI']) & df['RSI'].shift(1) < 31)) & 
                  ((df['RSI'] < 31) | (df['RSI'].shift(1) < 31)) & (df['RSI'].shift(1) < df['RSI']) &
                  ((df['ROC4'].shift(1) <-10)| (df['ROC'].shift(1) <-10)))
    
#    ban = (C < Ref(L,-1)AND C < Ref(L,-2)AND C < Ref(L,-3)AND C < Ref(L,-4))
#OR HHV(C,10) >1.15*C
#    df['Short'] = ((df['Close']< df['Low'].shift(1)) & (df['Close']< df['Low'].shift(2)) & (df['Close']< df['Low'].shift(3)) & (df['Close']< df['Low'].shift(4)))  | \
#                 (df['Max10D'] > 1.15* df['Close'])
                 
    df['Short'] =  ((df['Close'] < df['SMA50']) | (df['Close'] < 0.96*df['Close'].shift(1))  \
                    | ((df['Close'] < 0.97*df['Close'].shift(1)) & (df['Volume'] > df['VolMA30'])) \
                    | ((df['Close'] < df['Close'].shift(1)) & (df['Close'].shift(1) < df['Close'].shift(2)))   \
                    | ((df['Close'] >= 1.01*df['Close'].shift(1)) & (1.02*df['Close'].shift(1) >= df['Close'])) & (df['Volume'] >= 1.5*df['VolMA30']))
    
    df['Signal'] = 1* (df['LongShortTrend'] | df['Long']) + -1*df['Short']
    hm_days = ndays

    back_test = False
    for i in range(1,hm_days+1):
        if (df['Long'].iloc[-i] & (typetrade == 'Long')):
                print(" Canslim trading ", str(i), "days before ", df.iloc[-i].name ,  ticker)  
                back_test = True
                print_statistic(df, i)
                if (market != None):
                    get_statistic_index(i, start, end, update = False, source = "cp68", exchange = market)
        
        if (df['LongShortTrend'].iloc[-i] & (typetrade == 'LongShortTrend')):
                print(" Short trend trading ", str(i), "days before ", df.iloc[-i].name ,  ticker)  
                back_test = True
                print_statistic(df, i)
                if (market != None):
                    get_statistic_index(i, start, end, update = False, source = "cp68", exchange = market)

        if (df['Sideway'].iloc[-i] & (typetrade == 'Sideway')):
                print(" Sideway trading ", str(i), "days before ", df.iloc[-i].name ,  ticker)  
                back_test = True
                print_statistic(df, i)
                if (market != None):
                    get_statistic_index(i, start, end, update = False, source = "cp68", exchange = market)

        if (df['EarlySignal'].iloc[-i] & (typetrade == 'EarlySignal')):
                print(" Early breakout signal trading ", str(i), "days before ", df.iloc[-i].name ,  ticker)  
                back_test = True
                print_statistic(df, i)
                if (market != None):
                    get_statistic_index(i, start, end, update = False, source = "cp68", exchange = market)

        if (df['MarkM'].iloc[-i] & (typetrade == 'MarkM')):
                print(" Mark Minervini trading ", str(i), "days before ", df.iloc[-i].name ,  ticker)  
#                print(ticker)  
                back_test = True
                print_statistic(df, i)
                if (market != None):
                    get_statistic_index(i, start, end, update = False, source = "cp68", exchange = market)

        if (df['Bottom'].iloc[-i] & (typetrade == 'Bottom')):
                print(" Bottom trading ", str(i), "days before ", df.iloc[-i].name ,  ticker)   
                print_statistic(df, i)
                back_test = True
                if (market != None):
                    get_statistic_index(i, start, end, update = False, source = "cp68", exchange = market)
##   

        if (df['SidewayBreakout'].iloc[-i] & (typetrade == 'SidewayBreakout')):
                print(" Sideway filter ", str(i), "days before ", df.iloc[-i].name ,  ticker)   
                print_statistic(df, i)
                back_test = True
                if (market != None):
                    get_statistic_index(i, start, end, update = False, source = "cp68", exchange = market)
   
        if (df['Short'].iloc[-i] & (typetrade == 'Short')):
                print(" Short selling canslim ", str(i), "days before ", df.iloc[-i].name ,  ticker)   
                print_statistic(df, i)
                back_test = True
                if (market != None):
                    get_statistic_index(i, start, end, update = False, source = "cp68", exchange = market)
        
        if (df['Breakout'].iloc[-i] & (typetrade == 'Breakout')):
                print(" Breakout canslim ", str(i), "days before ", df.iloc[-i].name ,  ticker)   
                print_statistic(df, i)
                back_test = True
                if (market != None):
                    get_statistic_index(i, start, end, update = False, source = "cp68", exchange = market)
   
    
#    back_test = True
#    if back_test == False:
#        back_test = df['Buy'].sum() > 0 
    if back_test:
        run_backtest(df, ticker, trade = typetrade)
#     
    return df


def momentum_strategy(ticker, start, end, realtime = False, source = "cp68", market = None):
    
    df = process_data(ticker = ticker, start = start, end = end, realtime = realtime, source = source)
    
    
    df['SMA15'] = df['Close'].rolling(window=15).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA30'] = df['Close'].rolling(window=30).mean()
    
    
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA150'] = df['Close'].rolling(window=150).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    
    df['Max10D'] = df['Close'].shift(1).rolling(window = 10).max()
    
    
    df['MID'] = (df['High'] + df['Low']) /2
  
    
    
    df['Long'] = ((df['Close']> 1.02*df['Close'].shift(1)) & (df['Close'] > df['Open'])  & \
                  (df['Close'] > (df['High'] + df['Low'])/2)  &\
                 (1.05*df['Close'].shift(2) >= df['Close'].shift(1)) & (df['Volume'] >= df['Volume'].shift(1)) &\
                 ((df['Close']*df['Volume'] >= 3E6)) & (df['RSI'] >=50) &\
                 (((df['Volume'] >= 1.3*df['VolMA30']) |(df['Volume'] > 2*250000))) &\
                 (df['Close'] >= df['SMA200']) & (df['Close'] >= df['SMA30']) & ((df['Close']> df['Max6M']) | (df['Close']> df['Max3M']) |(df['Close']>= df['High4D'])) &\
                 (df['PCT_HL'] <= 15))
        
    df['LongShortTrend'] = ((df['Close']> 1.02*df['Close'].shift(1)) & (df['Close'] > df['Open'])  & \
                  (df['Close'] > (df['High'] + df['Low'])/2)  &\
                 (1.05*df['Close'].shift(2) >= df['Close'].shift(1)) & (df['Volume'] >= df['Volume'].shift(1)) &\
                 ((df['Close']*df['Volume'] >= 3E6)) &\
                 (((df['Volume'] >= 1.3*df['VolMA30']) |(df['Volume'] > 2*250000))) &\
                 (df['Close'] >= df['SMA20']) & (df['Close'] >= df['SMA30']) & (df['Close']>= df['High4D']) &\
                 (df['PCT_HL'] <= 30))       
    
    # df['Long4D'] = ((df['Close'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2)) & (df['High'].shift(2) > df['High'].shift(3)))
    
    df['ROC4'] = talib.ROC(df['Close'].values, timeperiod = 4)
        
   
    df['Bottom'] = ((df['Close'] > df['Open']) & (df['Close']*df['Volume'] > 1E7) & 
#                  ((df['RSI'] < 31) | ((df['RSI'].shift(1) < df['RSI']) & df['RSI'].shift(1) < 31)) & 
                  ((df['RSI'] < 31) | (df['RSI'].shift(1) < 31)) & (df['RSI'].shift(1) < df['RSI']) &
                  ((df['ROC4'].shift(1) <-10)| (df['ROC'].shift(1) <-10)))
    
#    ban = (C < Ref(L,-1)AND C < Ref(L,-2)AND C < Ref(L,-3)AND C < Ref(L,-4))
#OR HHV(C,10) >1.15*C
#    df['Short'] = ((df['Close']< df['Low'].shift(1)) & (df['Close']< df['Low'].shift(2)) & (df['Close']< df['Low'].shift(3)) & (df['Close']< df['Low'].shift(4)))  | \
#                 (df['Max10D'] > 1.15* df['Close'])
                 
    df['Short'] =  (((df['Close'] < df['Low']).shift(1)) & ((df['Close'] < df['Low']).shift(2)) &\
                    ((df['Close'] < df['Low']).shift(2)) & ((df['Close'] < df['Low']).shift(3))) |\
                    (df['Max10D'] > 1.15* df['Close'])
                    
                    
    df['Buy'] = 1* (df['Long'] | df['Bottom'] | df['LongShortTrend'] )
    df['Sell'] = -1*(df['Short'])
        
   
    return df


def canslim_usstock(ticker, start, end, realtime = False, source = "cp68", market = None, ndays = 2):
    
    df = process_data(ticker = ticker, start = start, end = end, realtime = realtime, source = source)
    
    
    df['SMA15'] = df['Close'].rolling(window=15).mean()
    df['SMA30'] = df['Close'].rolling(window=30).mean()
    
   
    
    
#    n_fast = 12
#    n_slow = 26
#    nema = 9
#    df['MACD_12_26'], df['MACDSign9'], _ = compute_MACD(df, n_fast, n_slow, nema)
    
    
    
    df['ValueM30'] = df['Value'].rolling(window = 30).mean()
   
   
    df['Max10D'] = df['Close'].shift(1).rolling(window = 10).max()
    
    df['Long'] = ((df['Close']> 1.01*df['Close'].shift(1)) & (df['Close'] > df['Open'])  & \
                 (1.05*df['Close'].shift(2) >= df['Close'].shift(1)) & \
                 ((df['Close']*df['Volume'] >= 5000000)) & (df['RSI'] >=50) &\
                 ((df['Volume'] > df['VolMA30'])) & \
                 (df['Close'] > df['SMA30']) & ((df['Close']> df['Max6M']) | (df['Close']> df['Max3M']) |(df['Close']>= df['High4D'])))
   
    
    
#    df['Long'] = ((df['ValueM30']> 5000000) & (df['Close'] > df['Open'])  & (df['Close'] > df['Low']) &\
#                 (df['PCT_HL'] < 10) & (df['Volume'] > 1.2*df['VolMA30']) & \
#                 (df['High15D'] > 1.05*df['Low15D'])  & \
#                 ((df['Close'] > df['SMA30']) & (df['Close']> df['High15D'])))
    
    df['ROC4'] = talib.ROC(df['Close'].values, timeperiod = 4)
    

  
    
    df['Signal'] = 1* (df['Long']) 
    hm_days = ndays

    back_test = False
    for i in range(1,hm_days+1):
        if (df['Long'].iloc[-i] ):
                print(" US stocks canslim trading ", str(i), "days before ", df.iloc[-i].name ,  ticker)  
                back_test = True
                print_statistic(df, i)
                
               
#        if (df['Bottom'].iloc[-i] ):
#                print(" Bottom trading ", str(i), "days before ", df.iloc[-i].name ,  ticker)   
#                print_statistic(df, i)
#                back_test = True
#                if (market != None):
#                    get_statistic_index(i, start, end, update = False, source = "cp68", exchange = market)
##   
#        if (df['Outperform'].iloc[-i] ):
#                print(" Outperform filter ", str(i), "days before ", df.iloc[-i].name ,  ticker)   
#                print_statistic(df, i)
#                back_test = True
#                if (market != None):
#                    get_statistic_index(i, start, end, update = False, source = "cp68", exchange = market)
#   
    
#    back_test = True
#    if back_test == False:
#        back_test = df['Buy'].sum() > 0 
    if back_test:
        run_backtest(df, ticker, trade = 'Long')
#     
    return df

def short_selling(ticker, start, end, realtime = False, source = "cp68", market = None , ndays = 2, typetrade = 'Short'):
       
    df = process_data(ticker = ticker, start = start, end = end, realtime = realtime, source = source)
        
  
    #((L-XAVGC3)/L) 
    df['PRICE_D'] = (df['Low'] - df['EMA3']) /df['Low']
    n_fast = 3
    n_slow = 6
    nema = 9
    df['MACD_3_6'], df['MACDSign369'], df['MACDDiff369'] = compute_MACD(df, n_fast, n_slow, nema)
    
#     MDR (Mean-Divergence Reading) (XAVGC3.1-XAVGC6.1)/(XAVG(XAVGC3.1,9)-XAVG(XAVGC6.1,9))
    
    df['MDR'] = df['MACD_3_6'].shift(1)/df['MACDSign369'].shift(1)
    
#    18-EMA-D (18-EMA Divergence) ((L-XAVGC18)/L)
    df['18_EMA_D'] = (df['Low'] - df['EMA18']) /df['Low']
#    50-EMA-D (50-EMA Divergence) ((L-XAVGC50)/L)
    df['50_EMA_D'] = (df['Low'] - df['EMA50']) /df['Low']
#    %R-Short (Last Interval Risk Percentage) (((L-H)+0.05)/L)
    df['Risk'] = (df['Low'] - df['High'] + 0.05) /df['Low']
#    R-18 (Number of R to Target 1) ((L-XAVGC18)/L)/(((L-H)+0.05)/L)
    df['R_18'] = df['18_EMA_D']/ df['Risk']
#    R-50 (Number of R to Target 2) ((L-XAVGC50)/L)/(((L-H)+0.05)/L)
    df['R_50'] = df['50_EMA_D']/ df['Risk']
    
    df['EMA_UP'] = ((df['EMA18'] > df['EMA50']) & (df['EMA50'] > df['EMA100']) & (df['EMA100'] > df['EMA200']))
   
#    (XAVGC3-XAVGC6)>0AND
#(XAVGC12-XAVGC26)>0AND
#C>XAVGC18AND
#XAVGC18>XAVGC50AND
#XAVGC50>XAVGC100AND
#XAVGC100>XAVGC200AND
    n_fast = 12
    n_slow = 26
    nema = 9
    df['MACD_12_26'],df['MACDSign12269'], _ = compute_MACD(df, n_fast, n_slow, nema)
    
    df['CONTEXT'] = (df['MACD_3_6'] > 0) & (df['MACD_12_26'] > 0) & (df['Close'] > df['EMA18']) & df['EMA_UP'] 

#     UNIVERSAL SCAN TRIGGER (Standard Constructive Reversal)
#     C<O AND
#     C1>O1 AND
#     L>L1 
    df['Divergence'] = np.where((df['MACD_3_6'] < df['MACD_3_6'].shift(1)) & (df['MACD_3_6'].shift(2) < df['MACD_3_6'].shift(1)), df['MACD_3_6'].shift(1), np.nan)
    df['Divergence'] = np.where((df['MACD_3_6'] > df['MACDSign369']), df['MACD_3_6'], np.nan)
    
    df['Divergence'].fillna(method ="ffill", inplace = True)

    df['SHORT_SELL'] =  df['CONTEXT'] & (df['Divergence'] < df['Divergence'].shift(1)) &\
                (df['MACD_3_6'] < df['MACD_3_6'].shift(1)) & (df['Close'] < df['Open']) &\
                  (df['Close'].shift(1) > df['Open'].shift(1)) & \
                                        (df['Low'] > df['Low'].shift(1))
    

    hm_days = ndays
    back_test = False
    for i in range(1,hm_days+1):
        if (df['SHORT_SELL'].iloc[-i]):
#            if (df['Close'].iloc[-i] < df['Open'].iloc[-i]):
                print(" Short selling", str(i), " days before ", df.iloc[-i].name ,  ticker)
#                print(" Risk ", df['Risk'].iloc[-i])
#                print(" Price divergence ", df['PRICE_D'].iloc[-i])
                print_statistic(df, i)
                back_test = True
                if (market != None):
                    get_statistic_index(i, start, end, update = False, source = "cp68", exchange = market)
       
    df['Signal'] = -1*df['SHORT_SELL']    
    df['Short'] = df['SHORT_SELL']  
    if back_test:
        run_backtest(df, ticker,  trade = typetrade)

    return df    
                
def process_data(ticker, start, end, realtime = False, source = "cp68"):
    
#    print(source)
    if source == "ssi":
        file_path = symbol_to_path(ticker, base_dir = source)
        df = pd.read_csv(file_path, index_col ="DATE", parse_dates = True,  dayfirst=True,
                     usecols = ["DATE", "OPEN","CLOSE","HIGHEST","LOWEST","TOTAL VOLUMN", "TOTAL VALUES"], na_values = "nan")
        df = df.reset_index()
        df = df.rename(columns = {'DATE': 'Date', "OPEN": 'Open', 'HIGHEST': 'High',
                                  'LOWEST': 'Low','CLOSE' : 'Close', 'TOTAL VOLUMN': 'Volume', 'TOTAL VALUES': 'Values'})
        df = df.set_index('Date')
        
    if source == 'cp68':
        file_path = symbol_to_path(ticker, base_dir = source)
        df = pd.read_csv(file_path, index_col ="<DTYYYYMMDD>", parse_dates = True, 
                     usecols = ["<DTYYYYMMDD>", "<OpenFixed>","<HighFixed>","<LowFixed>","<CloseFixed>","<Volume>", "<VolumeDeal>","<VolumeFB>", "<VolumeFS>"], na_values = "nan")
        df = df.reset_index()
        df = df.rename(columns = {'<DTYYYYMMDD>': 'Date', "<OpenFixed>": 'Open', '<HighFixed>': 'High',
                                  '<LowFixed>': 'Low','<CloseFixed>' : 'Close', '<Volume>': 'Volume', '<VolumeDeal>':'Deal', '<VolumeFB>': 'FB', '<VolumeFS>': 'FS'})
        df = df.set_index('Date')
    
    if source == 'alpha':
        if realtime: 
            ts = TimeSeries(key='9ODDY4H8J5P847TA', output_format='pandas')
            df, _ = ts.get_daily(symbol=ticker, outputsize='full')
            df = df.reset_index()
            df = df.rename(columns = {'date': 'Date', "1. open": 'Open', '2. high': 'High',
                                      '3. low': 'Low','4. close' : 'Close', '5. volume': 'Volume'})
           
            df = df.set_index('Date')
        else:
            
            file_path = symbol_to_path(ticker, base_dir = source)
            df = pd.read_csv(file_path, index_col ="date", parse_dates = True, 
                         usecols = ["date", "1. open", "2. high","3. low","4. close", "5. volume"], na_values = "nan")
            df = df.reset_index()
            df = df.rename(columns = {'date': 'Date', "1. open": 'Open', '2. high': 'High',
                                      '3. low': 'Low','4. close' : 'Close', '5. volume': 'Volume'})
           
            df = df.set_index('Date')
#    if (source == 'yahoo'):
#        file_path = symbol_to_path(ticker, base_dir = source)
#        df = pd.read_csv(file_path, index_col ="Date", parse_dates = True,  
#                     usecols = ["Date", "Open", "High","Low","Close", "Volume"], na_values = "nan")
#        df = df.reset_index()
#        df = df.set_index('Date')
        
    if (source == 'yahoo'):
        if realtime:          
#            df = yf.download(ticker, start, end)
            df = pdr.get_data_yahoo(ticker, start=start, end=end,  as_panel = False) 
        else:            
            file_path = symbol_to_path(ticker, base_dir = source)
            df = pd.read_csv(file_path, index_col ="Date", parse_dates = True,  
                         usecols = ["Date", "Open", "High","Low","Close", "Volume"], na_values = "nan")
            df = df.reset_index()
            df = df.set_index('Date')
        
    # columns order for backtrader type
    columnsOrder=["Open","High","Low","Close", "Volume", "OpenInterest", "FB", "FS"]
    # change the index by new index
    df = df.reindex(columns = columnsOrder)  
    # change date index to increasing order
    df = df.sort_index()   
    # take a part of dataframe
    df = df.loc[start:end]
    
    if (realtime & ((source == 'cp68') | (source == 'ssi'))):
#        print(ticker)
        # actual_price = get_info_stock(ticker)
        actual_price = get_info_stock_cp68_mobile(ticker)
        # actual_price = get_info_stock_bsc(ticker)
        today = datetime.datetime.today()
        next_date = today
        df.loc[next_date] = ({ 'Open' : actual_price['Open'].iloc[-1],
                        'High' : actual_price['High'].iloc[-1], 
                        'Low' : actual_price['Low'].iloc[-1],
                        'Close' : actual_price['Close'].iloc[-1],
                        'Volume' : actual_price['Volume'].iloc[-1],
                        'OpenInterest': np.nan,
                        'FB': np.nan,
                        'FS': np.nan})
        df = df.reset_index()
        df = df.rename(columns = {'index': 'Date'}) 
        df = df.set_index('Date')
        
    df['VolMA30'] = df['Volume'].rolling(window = 30, center = False).mean()
    df['VolMA15'] = df['Volume'].rolling(window = 15, center = False).mean()
    df['VolTMA3'] = 3*df['Volume'].rolling(window = 3, center = False).mean()
    
    df['Volatility'] = df['Close'].rolling(window=5,center=False).std()
    df['PCT_Change'] = df['Close'].pct_change()
    df['Value'] = df['Volume']*df['Close']   
    df['ValueMA30'] = df['Value'].rolling(window = 30, center = False).mean()
   
    df['RSI'] = talib.RSI(df['Close'].values, timeperiod = 14)
    df['ROC'] = talib.ROC(df['Close'].values, timeperiod = 3)
#    df['RSW'] = 0.4* talib.ROC(df['Close'].values, timeperiod = 65) + \
#    0.3* talib.ROC(df['Close'].values, timeperiod = 130) + \
#    0.3*talib.ROC(df['Close'].values, timeperiod = 260)
    
    df['RSW'] = 40*df['Close'].pct_change(periods = 63).fillna(0) \
             + 20*df['Close'].pct_change(periods = 126).fillna(0) \
             + 20*df['Close'].pct_change(periods = 189).fillna(0) \
             + 20*df['Close'].pct_change(periods = 252).fillna(0) 
             
    df['Max5D'] = df['Close'].shift(1).rolling(window = 5).max() 
    df['Max10D'] = df['Close'].shift(1).rolling(window = 10).max() 
    df['Min5D'] = df['Close'].shift(1).rolling(window = 5).min()
    df['Min10D'] = df['Close'].shift(1).rolling(window = 10).min()
    
    df['Low5D'] = df['Low'].shift(1).rolling(window = 5).min()
    
    df['Sideways'] = (df['RSI'] >=40) & (df['Close']*df['Volume'] >= 3000000) &\
    (df['Max5D'] <= 1.055*df['Min5D'])
    
    
    df['Low15D'] = df['Low'].shift(1).rolling(window = 15).min()
    df['High15D'] = df['High'].shift(1).rolling(window = 15).max()
    df['PCT_HL'] = ((df['High15D'] -df['Low15D'])/df['Low15D'])*100
    df['HL_PCT'] = (df['High'] - df['Low'])/df['Low']*100
    
    df['Max3M'] = df['Close'].shift(1).rolling(window = 63).max()
    df['Max6M'] = df['Close'].shift(1).rolling(window = 126).max() 
   
    df['Max9M'] = df['Close'].shift(1).rolling(window = 189).max() 
    df['Max12M'] = df['Close'].shift(1).rolling(window = 252).max() 
    
    df['Min6M'] = df['Close'].shift(1).rolling(window = 126).min()
    df['Min9M'] = df['Close'].shift(1).rolling(window = 189).min() 
    df['Min12M'] = df['Close'].shift(1).rolling(window = 252).min() 
    
    df['High4D'] = df['High'].shift(1).rolling(window = 4).max()
    df['High3D'] = df['High'].shift(1).rolling(window = 3).max()
    
    df['EMA3'] = pd.Series(pd.Series.ewm(df['Close'], span = 3, min_periods = 3-1).mean()) 
    df['EMA6'] = pd.Series(pd.Series.ewm(df['Close'], span = 6, min_periods = 6-1).mean()) 
    df['EMA18'] = pd.Series(pd.Series.ewm(df['Close'], span = 18,  min_periods = 18-1).mean()) 
    df['EMA50'] = pd.Series(pd.Series.ewm(df['Close'], span = 50,  min_periods = 50-1).mean()) 
    df['EMA100'] = pd.Series(pd.Series.ewm(df['Close'], span = 100,  min_periods = 100-1).mean())
    df['EMA200'] = pd.Series(pd.Series.ewm(df['Close'], span = 200,  min_periods = 200-1).mean())
    
    n_fast = 12
    n_slow = 26
    nema = 9
    df['MACD_12_26'], df['Sign12_26_9'], _ = compute_MACD(df, n_fast, n_slow, nema)
    
    
     # MOMENTUM 
    df['MACD_UP'] = (df['MACD_12_26'] > df['Sign12_26_9'])
    df['MACD_DOWN'] = (df['MACD_12_26'] < df['Sign12_26_9'])
    
    
    
    # SUPPORT AND RESISTANCE
   
    # df['Support'] = np.where((df['Low'] >= df['Low'].shift(1)) & (df['Low'].shift(2) >= df['Low'].shift(1)), df['Low'].shift(1), np.nan)
    df['Support'] = np.where((df['Close'] >= df['Close'].shift(1)) & (df['Close'].shift(2) >= df['Close'].shift(1)), df['Close'].shift(1), np.nan)
    
    df['Support'].fillna(method ="backfill", inplace = True)
    df['Support'].fillna(method ="ffill", inplace = True)
    
    
    # df['Resistance'] = np.where((df['High'].shift(1) >= df['High']) & (df['High'].shift(2) <= df['High'].shift(1)), df['High'].shift(1), np.nan)
    df['Resistance'] = np.where((df['Close'].shift(1) >= df['Close']) & (df['Close'].shift(2) <= df['Close'].shift(1)), df['Close'].shift(1), np.nan)
    
    df['Resistance'].fillna(method ="backfill", inplace = True)
    df['Resistance'].fillna(method ="ffill", inplace = True)
    return df

def compute_support_resistance(df, i):
    
    S1 = np.nan
    S2 = np.nan
    S3 = np.nan
    R1 = np.nan
    R2 = np.nan
    R3 = np.nan
    
    support = df['Support'][:-i] 
    S0 = min(support.iloc[-i], df['Close'].iloc[-i-1], df['Low'].iloc[-i])
    ind = 1  
    
    
    while ((support.iloc[-ind] >= S0) & (ind < len(support))):
        ind = ind +1
    if (ind < len(support)):
        S1 =  support.iloc[-ind]  
    
    while ((support.iloc[-ind] >= S1) & (ind < len(support))):
        ind = ind +1
    if (ind < len(support)):
        S2 =  support.iloc[-ind]  
  
    while ((support.iloc[-ind] >= S2) & (ind < len(support))):
        ind = ind +1 
    if (ind < len(support)):
        S3 =  support.iloc[-ind]
    
    
    resistance = df['Resistance'][:-i]   
    R0 = max(resistance.iloc[-i], df['Close'].iloc[-i-1], df['High'].iloc[-i])    
    ind = 1

    while ((resistance.iloc[-ind] <= R0) & (ind < len(resistance))):
        ind = ind +1
    if (ind < len(resistance)):
        R1 =  resistance.iloc[-ind] 
    
    
    while ((resistance.iloc[-ind] <= R1) & (ind < len(resistance))):
        ind = ind +1
    if (ind < len(resistance)):
        R2 =  resistance.iloc[-ind]
    while ((resistance.iloc[-ind] <= R2) & (ind < len(resistance))):
        ind = ind +1
    if (ind < len(resistance)):
        R3 =  resistance.iloc[-ind]
    
    return S0, S1, S2, S3, R0, R1, R2, R3

def print_statistic(df, i):
#   
#    sddr = df['Close'].pct_change().std()
#    print('  Volatility last week: ', round(df['Volatility'].iloc[-i],2), "over all: ", round(sddr,2), "ratio  :", round(df['Volatility'].iloc[-i]/sddr,2)) 
    max_all = df['Close'].max()
   
    S0, S1, S2, S3, R0, R1, R2, R3 = compute_support_resistance(df, i)
    
    
    print('  Volume/volume(MA15) ratio: ', round(df['Volume'].iloc[-i]/df['VolMA15'].iloc[-i],2))
    print('  RSI indicator: ', df['RSI'].iloc[-i])
#    print('  Rate of change last 3 days: ', df['ROC'].iloc[-i])
    print('  Trading value (billion VND/million USD): ', round(df['Value'].iloc[-i]/1E6, 2), ' MA30 :', round(df['ValueMA30'].iloc[-i]/1E6, 2))
    print('  Money flow in last 5 days: ',round(df['Value'].iloc[-i-4]/1E6, 2), 
                                          round(df['Value'].iloc[-i-3]/1E6, 2),
                                          round(df['Value'].iloc[-i-2]/1E6, 2),
                                          round(df['Value'].iloc[-i-1]/1E6, 2), 
                                          round(df['Value'].iloc[-i]/1E6, 2))
    print('  Relative strength RSW: ', df['RSW'].iloc[-i])
    print('  Side ways status last 5 days: ',df['Sideways'].iloc[-i-4], df['Sideways'].iloc[-i-3], df['Sideways'].iloc[-i-2], df['Sideways'].iloc[-i-1], df['Sideways'].iloc[-i])
    print('  Price max 3M/6M/9M/12M: ', df['Max3M'].iloc[-i],df['Max6M'].iloc[-i], df['Max9M'].iloc[-i], df['Max12M'].iloc[-i])
    print('  Actual price Close/Low/High/Open:', df['Close'].iloc[-i], df['Low'].iloc[-i], df['High'].iloc[-i], df['Open'].iloc[-i])
   
    print('  PCT_Change last 7 days:',round(100*df['PCT_Change'].iloc[-i-6], 2),
                                      round(100*df['PCT_Change'].iloc[-i-5], 2),
                                      round(100*df['PCT_Change'].iloc[-i-4], 2), 
                                      round(100*df['PCT_Change'].iloc[-i-3], 2),
                                      round(100*df['PCT_Change'].iloc[-i-2], 2),
                                      round(100*df['PCT_Change'].iloc[-i-1], 2), 
                                      round(100*df['PCT_Change'].iloc[-i], 2))
    print('  Variation last 7 days max/min (H/L): ', round(df['PCT_HL'].iloc[-i-6], 2), round(df['PCT_HL'].iloc[-i-5], 2),
          round(df['PCT_HL'].iloc[-i-4], 2), round(df['PCT_HL'].iloc[-i-3], 2), round(df['PCT_HL'].iloc[-i-2], 2), round(df['PCT_HL'].iloc[-i-1], 2), round(df['PCT_HL'].iloc[-i], 2))
   
    print('  Variation last 7 days  (H/L) : ', round(df['HL_PCT'].iloc[-i-6], 2), round(df['HL_PCT'].iloc[-i-5], 2),
          round(df['HL_PCT'].iloc[-i-4], 2), round(df['HL_PCT'].iloc[-i-3], 2), round(df['HL_PCT'].iloc[-i-2], 2), round(df['HL_PCT'].iloc[-i-1], 2), round(df['HL_PCT'].iloc[-i], 2))
   
    print('  Volume last 7 days (100K):',round(df['Volume'].iloc[-i-6]/1E5, 2),
                                  round(df['Volume'].iloc[-i-5]/1E5, 2),
                                  round(df['Volume'].iloc[-i-4]/1E5, 2), 
                                  round(df['Volume'].iloc[-i-3]/1E5, 2),
                                  round(df['Volume'].iloc[-i-2]/1E5, 2),
                                  round(df['Volume'].iloc[-i-1]/1E5, 2), 
                                  round(df['Volume'].iloc[-i]/1E5, 2))
    print('  Volume MA3 last 7 days (100K):',round(df['VolTMA3'].iloc[-i-6]/1E5, 2),
                                  round(df['VolTMA3'].iloc[-i-5]/1E5, 2),
                                  round(df['VolTMA3'].iloc[-i-4]/1E5, 2), 
                                  round(df['VolTMA3'].iloc[-i-3]/1E5, 2),
                                  round(df['VolTMA3'].iloc[-i-2]/1E5, 2),
                                  round(df['VolTMA3'].iloc[-i-1]/1E5, 2), 
                                  round(df['VolTMA3'].iloc[-i]/1E5, 2))
    print('  Ratio vs max H3D/H4D/3M/6M/9M/12M/all_time:', round(df['Close'].iloc[-i]/df['High3D'].iloc[-i], 2),
                                                      round(df['Close'].iloc[-i]/df['High4D'].iloc[-i], 2),
                                                       round(df['Close'].iloc[-i]/df['Max3M'].iloc[-i], 2),
                                                       round(df['Close'].iloc[-i]/df['Max6M'].iloc[-i], 2),
                                                       round(df['Close'].iloc[-i]/df['Max9M'].iloc[-i], 2),
                                                       round(df['Close'].iloc[-i]/df['Max12M'].iloc[-i], 2),
                                                       round(df['Close'].iloc[-i]/max_all,2),)
    
#    print('  Hurst exponent in this period ', hurst_f(df['Close']))
    T5 = round((df['Close'].shift(-5).iloc[-i]/df['Close'].iloc[-i]-1)*100, 2)
    T6 = round((df['Close'].shift(-6).iloc[-i]/df['Close'].iloc[-i]-1)*100, 2)
    T7 = round((df['Close'].shift(-7).iloc[-i]/df['Close'].iloc[-i]-1)*100, 2)
    T8 = round((df['Close'].shift(-8).iloc[-i]/df['Close'].iloc[-i]-1)*100, 2)
    T9 = round((df['Close'].shift(-9).iloc[-i]/df['Close'].iloc[-i]-1)*100, 2)
    T10 = round((df['Close'].shift(-10).iloc[-i]/df['Close'].iloc[-i]-1)*100, 2)
    T1 = round((df['Close'].shift(-1).iloc[-i]/df['Close'].iloc[-i]-1)*100, 2)
    T2 = round((df['Close'].shift(-2).iloc[-i]/df['Close'].iloc[-i]-1)*100, 2)
    T4 = round((df['Close'].shift(-4).iloc[-i]/df['Close'].iloc[-i]-1)*100, 2)
    print('  Support S0 S1 S2 S3 :', S0, S1, S2, S3)
    print('  Resistance R0 R1 R2 R3 :', R0, R1, R2, R3)
    print('  Loss/gain T+1/T+2/T+3/T+4 :', T1, T2, round(df['ROC'].shift(-3).iloc[-i], 2), T4)
    print('  Back test T+5 : T+10:', T5, T6, T7, T8, T9, T10)    
    
    # R = df['High'].iloc[-i] - df['Low'].iloc[-i]
    targetR1 = round((R1-df['Close'].iloc[-i]) /df['Close'].iloc[-i]*100, 2)
    targetR0 = round((R0-df['Close'].iloc[-i]) /df['Close'].iloc[-i]*100, 2)
    cutlossS0 = round((df['Close'].iloc[-i]-S0) /df['Close'].iloc[-i]*100, 2)
    stoploss = max(df['Close'].iloc[-i]*0.96, df['Low'].iloc[-i])
    sl_pct = round((df['Close'].iloc[-i]-stoploss) /df['Close'].iloc[-i]*100, 2)
    risk = df['Close'].iloc[-i] - stoploss
    target = df['Close'].iloc[-i] + 2*risk
    tp_pct = round((target - df['Close'].iloc[-i]) /df['Close'].iloc[-i]*100, 2)
    print('  Support S0 (%) :', cutlossS0, '%', ' Resistance R0 R1 (%) :', targetR0, targetR1) 
    print('  Stop loss :', stoploss, sl_pct, '%  Take profit:', target, tp_pct, '%')    
    print('----------------------------------------------------------------')
   

def ninja_trading(ticker, start, end, realtime = False, source = "cp68"):
       
    df = process_data(ticker = ticker, start = start, end = end, realtime = realtime, source = source)
    
    df['R'] = (df['High'] - df['Low'] + 0.04)
    df['Target_SELL'] = df['R']*3 + df['Close']
    df['Target_STOPLOSS'] = - df['R'] + df['Close']
    df['Risk'] = df['R'] /df['Low']
    df['Reward'] = df['Target_SELL']/df['Close']
    
  
    
#    macd, signal, hist = talib.MACD(df['Close'].values, fastperiod=12, slowperiod= 26, signalperiod=9)
#    MACD = pd.Series(macd,  index = df.index, name = 'MACD_12_26')     
#    #Signal line 9 EMA of EMA12- EMA26
#    MACDsign = pd.Series(signal,  index = df.index, name = 'MACDSign9')  
#    # Histo = diff from (EMA12-EMA26) - EMA9(EMA12- EMA26)
#    MACDdiff = pd.Series(hist,  index = df.index, name = 'MACDDiff')
    
    
  
          
    df['18_LONG']= (swing_high(df) & check_crossover(df, high = 'Close', low = 'EMA18'))    
#    & (df['Close'] > df['EMA18']) & (df['C_1d'] > df['EMA18_1d']) & (df['C_2d'] < df['EMA18_2d']))
    
    df['18_SHORT']= (swing_low(df)  & check_crossover(df, high = 'EMA18', low = 'Close')) 
#    & (df['Close'] < df['EMA18']) & (df['C_1d'] < df['EMA18_1d']) & (df['C_2d'] > df['EMA18_2d']))
       
    df['3_18_LONG']= (swing_high(df)  & check_crossover(df, high = 'EMA3', low = 'EMA18'))
#    & (df['EMA3'] > df['EMA18']) & (df['EMA3_1d'] > df['EMA18_1d']) & (df['EMA3_2d'] < df['EMA18_2d']))    
    
    df['3_18_SHORT']= (swing_low(df) & check_crossover(df, high = 'EMA18', low = 'EMA3'))
#    & (df['EMA3'] < df['EMA18']) & (df['EMA3_1d'] < df['EMA18_1d']) & (df['EMA3_2d'] > df['EMA18_2d']))    
    
    df['3_6_LONG']= (swing_high(df) & check_crossover(df, high = 'EMA3', low = 'EMA6'))
#    & (df['EMA3'] > df['EMA6']) & (df['EMA3_1d'] > df['EMA6_1d']) & (df['EMA3_2d'] < df['EMA6_2d']))
    
    df['3_6_SHORT']= (swing_low(df) & check_crossover(df, high = 'EMA6', low = 'EMA3'))
#    & (df['EMA3'] < df['EMA6']) & (df['EMA3_1d'] < df['EMA6_1d']) & (df['EMA3_2d'] > df['EMA6_2d']))    
    
    df['6_18_LONG']= (swing_high(df) & check_crossover(df, high = 'EMA6', low = 'EMA18'))
#    & (df['EMA6'] > df['EMA18']) & (df['EMA6_1d'] > df['EMA18_1d']) & (df['EMA6_2d'] < df['EMA18_2d']))
    
    df['6_18_SHORT']= (swing_low(df) & check_crossover(df, high = 'EMA18', low = 'EMA6'))
#    & (df['EMA6'] < df['EMA18']) & (df['EMA6_1d'] < df['EMA18_1d']) & (df['EMA6_2d'] > df['EMA18_2d']))        
    
    df['3_50_LONG']= (swing_high(df) & check_crossover(df, high = 'EMA3', low = 'EMA50'))
#    & (df['EMA3'] > df['EMA50']) & (df['EMA3_1d'] > df['EMA50_1d']) & (df['EMA3_2d'] < df['EMA50_2d']))
    
    df['3_50_SHORT']= (swing_low(df) & check_crossover(df, high = 'EMA50', low = 'EMA3'))
#    & (df['EMA3'] < df['EMA50']) & (df['EMA3_1d'] < df['EMA50_1d']) & (df['EMA3_2d'] > df['EMA50_2d']))
    
    df['6_50_LONG']= (swing_high(df) & check_crossover(df, high = 'EMA6', low = 'EMA50'))
#    & (df['EMA6'] > df['EMA50']) & (df['EMA6_1d'] > df['EMA50_1d']) & (df['EMA6_2d'] < df['EMA50_2d']))
  
    df['6_50_SHORT']= (swing_low(df)  & check_crossover(df, high = 'EMA50', low = 'EMA6'))
#    & (df['EMA6'] < df['EMA50']) & (df['EMA6_1d'] < df['EMA50_1d']) & (df['EMA6_2d'] > df['EMA50_2d']))
    
    df['18_50_LONG']= (swing_high(df) & check_crossover(df, high = 'EMA18', low = 'EMA50'))
#    & (df['EMA18'] > df['EMA50']) & (df['EMA18_1d'] > df['EMA50_1d']) & (df['EMA18_2d'] < df['EMA50_2d']))
    
    df['18_50_SHORT']= (swing_low(df) & check_crossover(df, high = 'EMA50', low = 'EMA18'))
#    & (df['EMA18'] < df['EMA50']) & (df['EMA18_1d'] < df['EMA50_1d']) & (df['EMA18_2d'] > df['EMA50_2d']))

    df['3_6_18_LONG']= (swing_high(df) & check_crossover(df, high = 'EMA3', low = 'EMA6')
        & check_crossover(df, high = 'Close', low = 'EMA18'))
#    
#    & (df['EMA3'] > df['EMA6']) & (df['EMA3_1d'] > df['EMA6_1d']) & (df['EMA3_2d'] < df['EMA6_2d'])
#    & (df['Close'] > df['EMA18']) & (df['C_1d'] > df['EMA18_1d']) & (df['C_2d'] < df['EMA18_2d']))
    
    df['3_6_18_SHORT']= (swing_low(df) & check_crossover(df, high = 'EMA6', low = 'EMA3')
        & check_crossover(df, high = 'EMA18', low = 'Close'))
#    & (df['EMA3'] < df['EMA6']) & (df['EMA3_1d'] < df['EMA6_1d']) & (df['EMA3_2d'] > df['EMA6_2d'])
#    & (df['Close'] < df['EMA18']) & (df['C_1d'] < df['EMA18_1d']) & (df['C_2d'] > df['EMA18_2d']))
    
    df['MACD_SIGNAL_LONG']= (swing_high(df) & check_crossover(df, high = 'MACD_12_26', low = 'Sign12_26_9'))
#    & (df['MACD_12_26'] > df['MACDSign9']) & (df['MACD_12_26_1d'] > df['MACDSign9_1d'])
#    & (df['MACD_12_26_2d'] < df['MACDSign9_2d']))
    
    df['MACD_SIGNAL_SHORT']= (swing_low(df) & check_crossover(df, high = 'Sign12_26_9', low = 'MACD_12_26'))
#    & (df['MACD_12_26'] < df['MACDSign9']) & (df['MACD_12_26_1d'] < df['MACDSign9_1d'])
#    & (df['MACD_12_26_2d'] > df['MACDSign9_2d']))
        
    
    df['MACD_ZERO_LONG']= (swing_high(df) & check_over_zero(df, column = 'MACD_12_26'))
#    & (df['MACD_12_26'] > 0) & (df['MACD_12_26_1d'] > 0) & (df['MACD_12_26_2d'] < 0))
    
    df['MACD_ZERO_SHORT']= (swing_low(df) & check_below_zero(df, column = 'MACD_12_26'))
#    & (df['MACD_12_26'] < 0) & (df['MACD_12_26_1d'] < 0) & (df['MACD_12_26_2d'] > 0))
    
    # CONTEXTUAL RISK
    df['EMA_UP'] = ((df['EMA18'] > df['EMA50']) & (df['EMA50'] > df['EMA100']) & (df['EMA100'] > df['EMA200']))
    df['EMA_DOWN'] = ((df['EMA18'] < df['EMA50']) & (df['EMA50'] < df['EMA100']) & (df['EMA100'] < df['EMA200']))
    # MOMENTUM
    df['MACD_UP'] = ((df['MACD_12_26'] > df['Sign12_26_9']))
    df['MACD_DOWN'] = ((df['MACD_12_26'] < df['Sign12_26_9']))
    
    
    df['L18'] = (df['18_LONG'] & df['MACD_UP']) & df['EMA_UP'] 
    df['L3_18'] = (df['3_18_LONG'] & df['MACD_UP']) & df['EMA_UP'] 
    df['L3_6'] = (df['3_6_LONG'] &  df['MACD_UP'])  & df['EMA_UP'] 
    df['L6_18'] = (df['6_18_LONG'] &  df['MACD_UP']) & df['EMA_UP'] 
    df['L3_50'] = (df['3_50_LONG'] & df['MACD_UP']) & df['EMA_UP'] 
    df['L6_50'] = (df['6_50_LONG'] &  df['MACD_UP']) & df['EMA_UP'] 
    df['L18_50'] = (df['18_50_LONG'] &  df['MACD_UP']) & df['EMA_UP'] 
    df['L3_6_18'] = (df['3_6_18_LONG'] &  df['MACD_UP']) & df['EMA_UP']    
    df['L_MACD_SIGNAL']=  (df['MACD_SIGNAL_LONG'] &  df['MACD_UP']) & df['EMA_UP'] 
    df['L_MACD_ZERO']=  (df['MACD_ZERO_LONG'] &   df['MACD_UP']) & df['EMA_UP'] 
    
    df['L_EMA_FAN'] =  (swing_high(df) & (df['EMA_UP'] &   df['MACD_UP'])) 
    
    df['S18'] = (df['18_SHORT'] &  df['MACD_DOWN']) & df['EMA_DOWN']
    df['S3_18'] = (df['3_18_SHORT'] & df['MACD_DOWN'])  & df['EMA_DOWN']
    df['S3_6'] = (df['3_6_SHORT'] &  df['MACD_DOWN']) & df['EMA_DOWN'] 
    df['S6_18'] = (df['6_18_SHORT'] &  df['MACD_DOWN']) & df['EMA_DOWN'] 
    df['S3_50'] = (df['3_50_SHORT'] &  df['MACD_DOWN']) & df['EMA_DOWN']
    df['S6_50'] = (df['6_50_SHORT'] & df['MACD_DOWN']) & df['EMA_DOWN'] 
    df['S18_50'] = (df['18_50_SHORT'] &  df['MACD_DOWN']) & df['EMA_DOWN'] 
    df['S3_6_18'] = (df['3_6_18_SHORT'] &  df['MACD_DOWN']) & df['EMA_DOWN']
    df['S_MACD_SIGNAL']=  (df['MACD_SIGNAL_SHORT'] &  df['MACD_DOWN'])  & df['EMA_DOWN']
    df['S_MACD_ZERO']=  (df['MACD_ZERO_SHORT'] &  df['MACD_DOWN']) & df['EMA_DOWN'] 
    
    df['S_EMA_FAN'] = (swing_low(df)) & (df['MACD_DOWN'] & df['EMA_DOWN'])
    # 3 days checking: SH + 2 pullbacks or SH + IB + 2 pullbacks

   
    
    hm_days = 5
#    for i in range(1,hm_days+1):
#        if (df['L18'].iloc[-i] | df['L3_18'].iloc[-i] 
#            | df['L3_6'].iloc[-i] 
#            | df['L6_18'].iloc[-i] | df['L3_50'].iloc[-i]
#            | df['L6_50'].iloc[-i] | df['L18_50'].iloc[-i] | df['L3_6_18'].iloc[-i] 
#            | df['L_MACD_SIGNAL'].iloc[-i]
#            | df['L_MACD_ZERO'].iloc[-i]):
##            | df['L_EMA_FAN'].iloc[-i]):
##            if (df['Close'].iloc[-i] > df['Open'].iloc[-i]):
#                print(" Ninja trading", str(i), " days before", df.iloc[-i].name ,  ticker)
##                print(" Target sell", df['Target_SELL'].iloc[-i])
##                print(" Target STOP LOSS", df['Target_STOPLOSS'].iloc[-i])
##                print(" Risk ", df['Risk'].iloc[-i])
##                print(" Reward ", df['Reward'].iloc[-i])
##            print(" Price at that day : ", df.i[loc[-i][0:4])

#            
    for i in range(1,hm_days+1):
     if ((df['L18'].iloc[-i] & check_bounce(df, ind = i, nema = 18))
        | (df['L3_18'].iloc[-i] & check_bounce(df, ind = i, nema = 18))
        | (df['L3_6'].iloc[-i] & check_bounce(df, ind = i, nema = 6))
        | (df['L6_18'].iloc[-i] & check_bounce(df, ind = i, nema = 18))
        | (df['L3_50'].iloc[-i] & check_bounce(df, ind = i, nema = 50))
        | (df['L6_50'].iloc[-i] & check_bounce(df, ind = i, nema = 50))
        | (df['L18_50'].iloc[-i] & check_bounce(df, ind = i, nema = 50))
        | (df['L3_6_18'].iloc[-i] & check_bounce(df, ind = i, nema = 18))):
#         if (df['Close'].iloc[-i] > df['Open'].iloc[-i]) :
            print(" Advanced ninja trading LONG ", str(i), "days before", df.iloc[-i].name ,  ticker)
            print_statistic(df, i)
##            print(" Target sell", df['Target_SELL'].iloc[-i])
##            print(" Target STOP LOSS", df['Target_STOPLOSS'].iloc[-i])
##            print(" Risk ", df['Risk'].iloc[-i])
##            print(" Reward ", df['Reward'].iloc[-i])
      
     for i in range(1,hm_days+1):
         if ((df['S18'].iloc[-i] & check_bounce(df, ind = i, nema = 18))
            | (df['S3_18'].iloc[-i] & check_bounce(df, ind = i, nema = 18))
            | (df['S3_6'].iloc[-i] & check_bounce(df, ind = i, nema = 6))
            | (df['S6_18'].iloc[-i] & check_bounce(df, ind = i, nema = 18))
            | (df['S3_50'].iloc[-i] & check_bounce(df, ind = i, nema = 50))
            | (df['S6_50'].iloc[-i] & check_bounce(df, ind = i, nema = 50))
            | (df['S18_50'].iloc[-i] & check_bounce(df, ind = i, nema = 50))
            | (df['S3_6_18'].iloc[-i] & check_bounce(df, ind = i, nema = 18))):
#         if (df['Close'].iloc[-i] > df['Open'].iloc[-i]) :
            print(" Advanced ninja trading SHORT ", str(i), "days before", df.iloc[-i].name ,  ticker)
            print_statistic(df, i)
##            print(" Target sell", df['Target_SELL'].iloc[-i])
##            print(" Target STOP LOSS", df['Target_STOPLOSS'].iloc[-i])
##            print(" Risk ", df['Risk'].iloc[-i])
##            print(" Reward ", df['Reward'].iloc[-i])
        
#    
#    for i in range(1,hm_days+1):
#        if (df['L_EMA_FAN'].iloc[-i] & check_bounce(df, ind = i, nema = 6)):
#            if (df['Close'].iloc[-i] > df['Open'].iloc[-i]):
#                print(" Ninja trading EMA FAN", str(i), "days before", df.iloc[-i].name ,  ticker)
##                print(" Target sell", df['Target_SELL'].iloc[-i])
#                print(" Target STOP LOSS", df['Target_STOPLOSS'].iloc[-i])
#                print(" Risk ", df['Risk'].iloc[-i])
#            
    df['Signal'] = 1*(df['L18'] | df['L3_6'] |df['L3_18'] | df['L6_18'] | df['L3_50'] | df['L6_50'] | df['L18_50'] |  df['L3_6_18'] | df['L_MACD_SIGNAL'] | df['L_MACD_ZERO'] | df['L_EMA_FAN'])  +\
     -1*(df['S18'] | df['S3_6']| df['S3_18'] | df['S6_18'] | df['S3_50'] | df['S6_50'] | df['S18_50'] |  df['S3_6_18'] | df['S_MACD_SIGNAL'] | df['S_MACD_ZERO'] | df['S_EMA_FAN']) 
       
#    df['Buy'] = (df['L18'] | df['L3_18'] | df['L6_18'] | df['L3_50'] | df['L6_50'] | df['L18_50'] |  df['L3_6_18'] | df['L_MACD_SIGNAL'] | df['L_MACD_ZERO'] | df['L_EMA_FAN']) & (df['Close'].shift(-1) > df['Open'].shift(-1)) & (df['Close'] > df['Open'])
    
    df['1PB_RG'] = ((df['Close'] < df['Open']) & (df['Close'].shift(-1) > df['Open'].shift(-1))  & (df['Close'].shift(-2) > df['Open'].shift(-2))  )
    df['2PBIB_RRG'] = ((df['Close'] < df['Open']) & 
      (df['Close'].shift(-1) < df['Open'].shift(-1)) & 
      (df['Close'].shift(-2) > df['Open'].shift(-2)) &
      (df['Close'].shift(-3) > df['Open'].shift(-3)))
    
    df['1IB2PB_RRRG'] = (((df['High'] < df['High'].shift(1)) & df['Low'] > df['Low'].shift(1)) & 
    (df['Close'].shift(-1) < df['Open'].shift(-1)) &
    (df['Close'].shift(-2) < df['Open'].shift(-2)) & 
    (df['Close'].shift(-3) > df['Open'].shift(-3)) &
    (df['Close'].shift(-4) > df['Open'].shift(-4)))
    
    df['2PBIB_RRRG'] = ((df['Close'] < df['Open']) & 
      (df['Close'].shift(-1) < df['Open'].shift(-1)) & 
      (((df['High'].shift(-1) < df['High'].shift(-2)) & df['Low'].shift(-1) > df['Low'].shift(-2))) & 
      (df['Close'].shift(-3) > df['Open'].shift(-3)) &
      (df['Close'].shift(-4) > df['Open'].shift(-4)))
   
    df['PBIBPB_RRRG'] = ((df['Close'] < df['Open'])  & 
      ((df['High'] < df['High'].shift(-1)) & (df['Low'] > df['Low'].shift(-1)))  & 
        (df['Close'].shift(-2) < df['Open'].shift(-2)) &
        (df['Close'].shift(-3) > df['Open'].shift(-3)) &
        (df['Close'].shift(-4) > df['Open'].shift(-4)))
    
    df['IBPBIB_RRRG'] = (((df['High'] < df['High'].shift(1)) & df['Low'] > df['Low'].shift(1)) & 
      (df['Close'].shift(-1) < df['Open'].shift(-1)) & 
      ((df['High'].shift(-1) < df['High'].shift(-2)) & (df['Low'].shift(-1) > df['Low'].shift(-2)))  &        
        (df['Close'].shift(-3) > df['Open'].shift(-3)) &
        (df['Close'].shift(-4) > df['Open'].shift(-4)))
#    
    df['Buy'] = (df['L18'] | df['L3_6'] | df['L3_18'] | df['L6_18'] | df['L3_50'] | df['L6_50'] | df['L18_50'] |  df['L3_6_18'] | df['L_MACD_SIGNAL'] | df['L_MACD_ZERO'] | df['L_EMA_FAN'])  & (df['1PB_RG'] | df['2PBIB_RRG'] | df['1IB2PB_RRRG'] | df['2PBIB_RRRG'] |  df['PBIBPB_RRRG'] | df['IBPBIB_RRRG'] )
#    
#    back_test = df['Buy'].sum() > 0 
#    if back_test:        
#        df['5Days'] = df['Close'].shift(-5)
#        df['10Days'] = df['Close'].shift(-10)
#        df['Back_test'] = 1* (df['Buy'] & (df['10Days'] > df['Close']) & (df['5Days'] > df['Close'])  ) + -1* (df['Buy'] & (df['10Days'] <= df['Close'])& (df['5Days'] <= df['Close']))        
#        vals = df['Back_test'] .values.tolist()
#        str_vals = [str(i) for i in vals]
#        print('Back test ninja trading:', Counter(str_vals), 'symbol: ', ticker)
#    
#    df['Long'] = df['Buy']
#    back_test = df['Buy'].sum() > 0 
#    if back_test:
#        run_backtest(df, ticker)
    
    return df

   

def swing_high(df):
    return ((df['High'] < df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2)))
def swing_low(df):
    return ((df['Low'] > df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2)))

def check_bounce(df, ind = 5, nema = 18):    
    return ((df['Open'].iloc[-ind] > df['EMA{}'.format(nema)].iloc[-ind]) 
               & (df['Close'].iloc[-ind] > df['EMA{}'.format(nema)].iloc[-ind])
               &  (df['Low'].iloc[-ind] < df['EMA{}'.format(nema)].iloc[-ind]))
    
def check_crossover(df, high = 'Close', low = 'EMA18'):
#    ((df['Close'] > df['EMA18']) & (df['C_1d'] > df['EMA18_1d']) & (df['C_2d'] < df['EMA18_2d']))
    return ((df[high] > df[low]) & (df[high].shift(1) > df[low].shift(1)) & (df[high].shift(2) < df[low].shift(2)))

def check_over_zero(df, column = 'MACD_12_26'):
    return ((df[column] > 0) & (df[column].shift(1) > 0) & (df[column].shift(2) < 0))   
    
def check_below_zero(df, column = 'MACD_12_26'):
    return ((df[column] < 0) & (df[column].shift(1) < 0) & (df[column].shift(2) > 0)) 


def hedgefund_trading(ticker, start, end, realtime = False, source = "cp68"):
       
    df = process_data(ticker = ticker, start = start, end = end, realtime = realtime, source = source)
    
        
    n_fast = 3
    n_slow = 6
    nema = 20
    df['MACD_3_6'], df['Sign3_6_20'], _ =  compute_MACD(df, n_fast, n_slow, nema)
       
        
    n_fast = 50
    n_slow = 100
    nema = 9 
    df['MACD_50_100'], df['Sign50_100_9'], _ = compute_MACD(df, n_fast, n_slow, nema)
    
         

    # TREND TREND LONG
    df['TT_LONG']= ((df['Close'] > df['EMA18']) 
    & (df['MACD_3_6'] < df['Sign3_6_20']) & (df['MACD_3_6'] <0) & (df['Sign3_6_20'] > 0) 
    & ((df['MACD_50_100'] > df['Sign50_100_9']) | (df['MACD_50_100'] > 0.95*df['Sign50_100_9'])) & (df['MACD_50_100'] > 0))
    # TREND TREND SHORT
    df['TT_SHORT']= ((df['Close'] < df['EMA18']) 
    & (df['MACD_3_6'] > df['Sign3_6_20']) & (df['MACD_3_6'] > 0) & (df['Sign3_6_20'] < 0) 
    & ((df['MACD_50_100'] > df['Sign50_100_9']) | (df['MACD_50_100'] > 0.95*df['Sign50_100_9'])) & (df['MACD_50_100'] < 0))
   
    # TREND COUNTER TREND LONG
    df['TCT_LONG']= ((df['Close'] > df['EMA18']) 
    & (df['MACD_3_6'] < df['Sign3_6_20']) & (df['MACD_3_6'] <0) & (df['Sign3_6_20'] > 0) 
    & ((df['MACD_50_100'] > df['Sign50_100_9']) | (df['MACD_50_100'] > 0.95*df['Sign50_100_9'])) & (df['MACD_50_100'] < 0)) 
    
    # TREND COUNTER TREND SHORT
    df['TCT_SHORT']= ((df['Close'] < df['EMA18']) 
    & (df['MACD_3_6'] > df['Sign3_6_20']) & (df['MACD_3_6'] > 0) & (df['Sign3_6_20'] < 0) 
    & ((df['MACD_50_100'] > df['Sign50_100_9']) | (df['MACD_50_100'] > 0.95*df['Sign50_100_9'])) & (df['MACD_50_100'] > 0))
   
    
    
    # PRICE_ACTION
    df['1TT_LONG'] = ((df['High']< df['High'].shift(1)) 
    & (df['Close'] > df['EMA18']) & (df['Close'].shift(1) < df['EMA18'].shift(1)))
    
    df['2TT_LONG'] = ((df['High']< df['High'].shift(1)) 
    & (df['Close'] > df['EMA18']) & (df['Open'] < df['EMA18'])
    & (df['Close'].shift(1) < df['EMA18'].shift(1)) & (df['Open'].shift(1) > df['EMA18'].shift(1)))
    
    df['B_LONG'] = ((df['High']< df['High'].shift(1)) 
    & (df['Close'] > df['EMA18']) & (df['Open'] > df['EMA18']) & (df['Low'] < df['EMA18'])
    & (df['Close'].shift(1) > df['EMA18'].shift(1)) & (df['Open'].shift(1) > df['EMA18'].shift(1))
    & (df['Low'].shift(1) < df['EMA18'].shift(1)))
    
    
    df['S2BR_LONG'] = ((df['Low']< df['Low'].shift(1)) & (df['High'].shift(1) < df['High'].shift(2))
    & (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1))
    & (df['1TT_LONG'] | df['2TT_LONG'] | df['B_LONG']))
    
    df['PAR_LONG'] = ((df['Low']< df['Low'].shift(1)) & (df['High'].shift(1) < df['High'].shift(2))
    & (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1))
    & (df['Open'] > df['Low'].shift(1)) & (df['Close'] > df['Low'].shift(1))
    & (df['1TT_LONG'] | df['2TT_LONG'] | df['B_LONG']))
    
    df['PARW_LONG'] = ((df['Low']< df['Low'].shift(1)) & (df['High'].shift(1) < df['High'].shift(2))
    & (df['Open'] > df['Low'].shift(1)) & (df['Close'] > df['Low'].shift(1))
    & (df['1TT_LONG'] | df['2TT_LONG'] | df['B_LONG']))
    
    df['LIBR_LONG'] = ((df['Low'] > df['Low'].shift(1)) & (df['High'].shift(1) < df['High'].shift(2))
    & (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1))
    & (df['1TT_LONG'] | df['2TT_LONG'] | df['B_LONG']))
    
    df['LIBPR_LONG'] =  ((df['Low']< df['Low'].shift(1)) & (df['High'].shift(1) < df['High'].shift(2))
    & (df['Low'].shift(1) > df['Low'].shift(2)) & (df['High'].shift(2) < df['High'].shift(3))
    & (df['Low'].shift(2) < df['Low'].shift(3))
    & (df['1TT_LONG'] | df['2TT_LONG'] | df['B_LONG']))
    
    df['LHPR_LONG'] = ((df['Low']< df['Low'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
    & (df['Low'].shift(1) > df['Low'].shift(2)) & (df['High'].shift(2) < df['High'].shift(3))
    & (df['Low'].shift(2) < df['Low'].shift(3))
    & (df['1TT_LONG'] | df['2TT_LONG'] | df['B_LONG']))
    
    
    
    df['LTT'] = df['TT_LONG'] & (df['1TT_LONG'] | df['2TT_LONG'] | df['B_LONG'])
    df['LCTT'] = df['TCT_LONG'] & (df['1TT_LONG'] | df['2TT_LONG'] | df['B_LONG'])
    
    df['LTT_A'] = df['TT_LONG'] & (df['S2BR_LONG']  | df['PAR_LONG'] | df['PARW_LONG'] | df['LIBR_LONG'] | df['LIBPR_LONG']
    | df['LHPR_LONG'] )
    df['LCTT_A'] = df['TCT_LONG'] & (df['S2BR_LONG']  | df['PAR_LONG'] | df['PARW_LONG'] | df['LIBR_LONG'] | df['LIBPR_LONG']
    | df['LHPR_LONG'])
    
    df['Signal'] = 1*(df['LTT'] | df['LCTT'] | df['LTT_A'] | df['LCTT_A'])
    
    df['Long'] = df['Signal']
    
    df['1TT_SHORT'] = ((df['Low']> df['Low'].shift(1)) 
    & (df['Close'] < df['EMA18']) & (df['Close'].shift(1) > df['EMA18'].shift(1)))
    
    df['2TT_SHORT'] = ((df['Low'] > df['Low'].shift(1)) 
    & (df['Close'] < df['EMA18']) & (df['Open'] > df['EMA18'])
    & (df['Close'].shift(1) > df['EMA18'].shift(1)) & (df['Open'].shift(1) < df['EMA18'].shift(1)))
    
    df['B_SHORT'] = ((df['Low'] > df['Low'].shift(1)) 
    & (df['Close'] < df['EMA18']) & (df['Open'] < df['EMA18']) & (df['High'] > df['EMA18'])
    & (df['Close'].shift(1) < df['EMA18'].shift(1)) & (df['Open'].shift(1) < df['EMA18'].shift(1))
    & (df['High'].shift(1) > df['EMA18'].shift(1)))
    
    
        
    
    hm_days = 2
    back_test = False
    for i in range(1,hm_days+1):
        if (df['LTT'].iloc[-i] | df['LTT_A'].iloc[-i] ):
                print(" Slingshot trading TT", str(i), "days before ", df.iloc[-i].name ,  ticker)   
                print_statistic(df, i)
                back_test = True
        if (df['LCTT'].iloc[-i] | df['LCTT_A'].iloc[-i]  ):
                print(" Slingshot trading TCT", str(i), "days before ", df.iloc[-i].name ,  ticker)
                print_statistic(df, i)
                back_test = True
    df['Buy'] = (df['LTT'] | df['LCTT'] | df['LTT_A'] | df['LCTT_A'])
    
    
    if back_test:
        run_backtest(df, ticker,  trade = 'Long')
        
    return df

def bollinger_bands(ticker, start, end, realtime = False, source = "cp68",):
    
    df = process_data(ticker = ticker, start = start, end = end, realtime = realtime, source = source)
    
    period = 20
    nstd = 2.5
    rolling_mean = df['Close'].rolling(window=period,center=False).mean()
    rolling_std = df['Close'].rolling(window=period,center=False).std()
    
    df['Bollinger High'] = rolling_mean + (rolling_std * nstd)
    df['Bollinger Low'] = rolling_mean - (rolling_std * nstd)
    
      
    df['Signal'] = -1*((df['Close'] > df['Bollinger High']) & (df['Close'].shift(1)< df['Bollinger High'].shift(1))  ) + \
                   1 *((df['Close'] < df['Bollinger Low']) & (df['Close'].shift(1) > df['Bollinger Low'].shift(1))  )
            
    
    hmdays = 10
    back_test = False
    for i in range(1,hmdays+1):    
#        if (df['Close'].iloc[-i] > df['Bollinger High'].iloc[-i]) & (df['Close'].iloc[-i-1] < df['Bollinger High'].iloc[-i-1]):
#            print(" Bollinger trading sell", str(i), " days before", df.iloc[-i].name ,  ticker)
#            print_statistic(df, i)
#            back_test = True
        
        if (df['Close'].iloc[-i] < df['Bollinger Low'].iloc[-i]) & (df['Close'].iloc[-i-1] > df['Bollinger Low'].iloc[-i-1]):
            print(" Bollinger trading buy", str(i), "days before", df.iloc[-i].name ,  ticker)
            print_statistic(df, i)
            back_test = True
    df['Long'] =  (df['Close'] < df['Bollinger Low']) & (df['Close'].shift(1) > df['Bollinger Low'].shift(1))
    
#    back_test = df['Buy'].sum() > 0 
    if back_test:
        run_backtest(df, ticker,  typetrade = 'Long')
#    
    return df

def hurst_f(input_ts, lags_to_test = 20):
    tau = []
    lagvec = []
    for lag in range(2, lags_to_test):
        pp = np.subtract(input_ts[lag:],input_ts[:-lag])
        lagvec.append(lag)
        tau.append(np.sqrt(np.std(pp)))
    m = np.polyfit(np.log10(lagvec), np.log10(tau),1)
    hurst = m[0]*2
    return hurst


def compute_MACD(df, n_fast, n_slow, nema = 9):  
    EMAfast = pd.Series(pd.Series.ewm(df['Close'], span = n_fast, min_periods = n_fast - 1).mean())  
    EMAslow = pd.Series(pd.Series.ewm(df['Close'], span = n_slow, min_periods = n_slow - 1).mean())  
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(pd.Series.ewm(MACD, span = nema, min_periods = nema-1).mean(), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))  
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))  
    
#    MACD, MACDsign, MACDdiff = talib.MACD(df['Close'].values, fastperiod=n_fast, slowperiod= n_slow, signalperiod=nema)
    return MACD, MACDsign, MACDdiff

