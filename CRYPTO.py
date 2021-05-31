# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:35:57 2017

@author: sonng
"""
import pandas as pd
from finance_util import get_data_from_web, get_data, fill_missing_values, get_RSI
from strategy import crypto
import time
import os


def getliststocks(typestock = "CRYPTO"):
    if typestock == "CRYPTO":
        symbols = ['ETH-USD','ZEC-USD', 'BNB-USD','EOS-USD', 'ETC-USD', 'DASH-USD', 'XLM-USD',
                   'LINK-USD', 'LTC-USD', 'UNI3-USD', 'XRP-USD', 'ADA-USD','NEO-USD','DASH-USD',
                   'BCH-USD','MIOTA-USD', 'TRX-USD', 'XTZ-USD','DOGE-USD','MATIC-USD','SOL1-USD',
                   'ATOM1-USD','COMP-USD','VET-USD','THETA-USD','FIL-USD','AAVE-USD',
                   'XMR-USD','AVAX-USD', 'DOT1-USD','BTC-USD']    
    symbols = pd.unique(symbols).tolist()
    symbols = sorted(symbols)    
    return symbols
  
def analysis_trading(tickers, start, end, update = False, nbdays = 1, source = "cp68", trade = 'Long'):
    result = pd.DataFrame(columns =['Ticker', 'Advise','PCT', 'Close'])
    result = result.set_index('Ticker')
    for ticker in tickers:            
        try:
             res = crypto(ticker, start, end, realtime = update, source = source, ndays = nbdays, typetrade = trade)
             if len(res) > 1:
                 result.loc[res[0]] = [res[1], 100*res[2], res[3]]
                 # print(res[2], type(res[2]))
        except Exception as e:
            print (e)
            print("Error in reading symbol: ", ticker)
            pass
    return result 

import datetime
def passive_strategy(start_date, end_date, market = None, symbols = None, realtime = True, source = 'yahoo'):

    if symbols == None:
        symbols = getliststocks(typestock = market)
        
    if realtime:
        end_date = datetime.datetime.today()
        
    dates = pd.date_range(start_date, end_date)  # date range as index
    df_data = get_data(symbols, dates, benchmark = market, colname = 'Adj Close', realtime = realtime, source = source)  # get data for each symbol
    # Fill missing values
    fill_missing_values(df_data)
    df_volume = get_data(symbols, dates, benchmark = market, colname = 'Volume', realtime = realtime, source = source)  # get data for each symbol
    df_rsi = get_RSI(symbols, df_data)
    df_volume = df_volume.fillna(0)

    df_result = pd.DataFrame(index = symbols)    

    df_result['Ticker'] = symbols
    df_result['Close'] = df_data[symbols].iloc[-1,:].values
    df_result['PCT_C'] = 100*(df_data[symbols].iloc[-1,:].values - df_data[symbols].iloc[0,:].values)/df_data[symbols].iloc[0,:].values
    df_result['Volume'] = df_volume[symbols].iloc[-1,:].values + df_volume[symbols].iloc[-2,:].values

    df_result['Value'] = df_result['Close'] * df_result['Volume']   

    df_result ['Volatility'] = df_data[symbols].pct_change().std() 

    df_result ['PCT_3D'] = df_data[symbols].pct_change().iloc[-4,:].values*100
    df_result ['PCT_2D'] = df_data[symbols].pct_change().iloc[-3,:].values*100
    df_result ['PCT_1D'] = df_data[symbols].pct_change().iloc[-2,:].values*100
    df_result ['PCT_0D'] = df_data[symbols].pct_change().iloc[-1,:].values*100
   
    
    relative_strength = 40*df_data[symbols].pct_change(periods = 63).fillna(0) \
                     + 20*df_data[symbols].pct_change(periods = 126).fillna(0) \
                     + 20*df_data[symbols].pct_change(periods = 189).fillna(0) \
                     + 20*df_data[symbols].pct_change(periods = 252).fillna(0) 
    
    relative_strength1M = 100*df_data[symbols].pct_change(periods = 21).fillna(0)            
    relative_strength2M = 100*df_data[symbols].pct_change(periods = 42).fillna(0)      
    
    df_result ['RSW'] = relative_strength.iloc[-1,:].values
    
    df_result ['RSW1M'] = relative_strength1M.iloc[-1,:].values
    df_result ['RSW2M'] = relative_strength2M.iloc[-1,:].values
    df_result['RSI'] = df_rsi[symbols].iloc[-1,:].values
    
    
    return df_result, df_data

    
if __name__ == "__main__":#
    end_date =   "2021-5-28"
    start_date = "2020-5-3"    
    symbols = getliststocks(typestock = "CRYPTO")
    # get_data_from_web(tickers = symbols, start = start_date, end = end_date, source ='yahoo', redownload = True)
    # analysis_trading(symbols, start = start_date , end = end_date, update = False, source = "yahoo")
    
    # df_result, df_data = passive_strategy(start_date, end_date, market = None, symbols = None, realtime = True, source = 'cp68'):
    
    # df_result, df_data = passive_strategy(start_date, end_date, market = None, symbols = symbols, realtime = False, source = "yahoo")
       
    
    t0 = time.time()
    trade_type = ['EarlySignal','Bottom','SidewayBreakout']
    idx = 0 # EarlySignal
    realtime = not True
    if not realtime:
        get_data_from_web(tickers = symbols, start = start_date, end = end_date, source ='yahoo', redownload = True)
        df_result, df_data = passive_strategy(start_date, end_date, market = None, symbols = symbols, realtime = False, source = "yahoo")
       
    datasource = "yahoo"
    
    
    while True: 
        if realtime:
            os.system('cls')
            print('TRADING SYSTEM SIGNAL...............',time.asctime(time.localtime(time.time())))
            res = analysis_trading(tickers = symbols, start = start_date , end = end_date, update = realtime, nbdays = 1, source = datasource, trade = trade_type[idx])
            print("WAIT FOR 5 MINUTES ............................",time.asctime(time.localtime(time.time())))
            print(res.to_string())
            time.sleep(300.0 - ((time.time() - t0) % 300.0))
        else:            
            os.system('cls')
            print('OFF-LINE TRADING SIGNAL ............!')
            print('TRADING SYSTEM SIGNAL...............',time.asctime(time.localtime(time.time())))
            res = analysis_trading(tickers = symbols, start = start_date , end = end_date, update = realtime, nbdays = 1, source =datasource, trade = trade_type[idx])
            print(res.to_string())            
            break

    
   
