# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:35:57 2017

@author: sonng
"""
import pandas as pd
from finance_util import get_data_from_web
from strategy import crypto
import time
import os


def getliststocks(typestock = "CRYPTO"):
    if typestock == "CRYPTO":
        symbols = ['ETH-USD','ZEC-USD', 'BNB-USD','EOS-USD', 'ETC-USD', 'DASH-USD', 'XLM-USD',
                   'LINK-USD', 'LTC-USD', 'UNI3-USD', 'XRP-USD', 'ADA-USD','NEO-USD','DASH-USD',
                   'BCH-USD','MIOTA-USD', 'TRX-USD', 'XTZ-USD','DOGE-USD']    
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

    
if __name__ == "__main__":#
    end_date =   "2021-5-25"
    start_date = "2020-5-3"    
    symbols = getliststocks(typestock = "CRYPTO")
    # get_data_from_web(tickers = symbols, start = start_date, end = end_date, source ='yahoo', redownload = True)
    # analysis_trading(symbols, start = start_date , end = end_date, update = False, source = "yahoo")
    
    t0 = time.time()
    trade_type = ['EarlySignal','Bottom','SidewayBreakout']
    idx = 0 # EarlySignal
    realtime = True
    if not realtime:
        get_data_from_web(tickers = symbols, start = start_date, end = end_date, source ='yahoo', redownload = True)
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

    
   
