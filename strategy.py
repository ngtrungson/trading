# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 08:47:17 2018

@author: sonng
"""
from finance_util import symbol_to_path, get_info_stock
import numpy as np
import pandas as pd
import talib
import datetime
from datetime import date

def ninja_trading(ticker, start, end, realtime = False):
       
    file_path = symbol_to_path(ticker)
    df = pd.read_csv(file_path, index_col ="<DTYYYYMMDD>", parse_dates = True, 
                 usecols = ["<DTYYYYMMDD>", "<OpenFixed>","<HighFixed>","<LowFixed>","<CloseFixed>","<Volume>"], na_values = "nan")
    df = df.rename(columns = {'<DTYYYYMMDD>': 'Date', "<OpenFixed>": 'Open', '<HighFixed>': 'High',
                              '<LowFixed>': 'Low','<CloseFixed>' : 'Close', '<Volume>': 'Volume'})
  
    # columns order for backtrader type
    columnsOrder=["Open","High","Low","Close", "Volume", "OpenInterest"]
    # change the index by new index
    df = df.reindex(columns = columnsOrder)  
    # change date index to increasing order
    df = df.sort_index()   
    # take a part of dataframe
    df = df.loc[start:end]
    
    if realtime:
        actual_price = get_info_stock(ticker)
        today = datetime.datetime.today()
        next_date = today
        df.loc[next_date] = ({ 'Open' : actual_price['Open'].iloc[-1],
                        'High' : actual_price['High'].iloc[-1], 
                        'Low' : actual_price['Low'].iloc[-1],
                        'Close' : actual_price['Close'].iloc[-1],
                        'Volume' : actual_price['Volume'].iloc[-1],
                        'OpenInterest': np.nan})
    
    df['R'] = (df['High'] - df['Low']) + 0.04
    df['Target_SELL'] = df['R']*3 + df['Close']
    df['Target_STOPLOSS'] = - df['R']*3 + df['Close']
    
    df['EMA3'] = pd.Series(pd.Series.ewm(df['Close'], span = 3, min_periods = 3-1).mean()) 
    df['EMA6'] = pd.Series(pd.Series.ewm(df['Close'], span = 6, min_periods = 6-1).mean()) 
    df['EMA18'] = pd.Series(pd.Series.ewm(df['Close'], span = 18,  min_periods = 18-1).mean()) 
    df['EMA50'] = pd.Series(pd.Series.ewm(df['Close'], span = 50,  min_periods = 50-1).mean()) 
    df['EMA100'] = pd.Series(pd.Series.ewm(df['Close'], span = 100,  min_periods = 100-1).mean())
    df['EMA200'] = pd.Series(pd.Series.ewm(df['Close'], span = 200,  min_periods = 200-1).mean())
    
#    macd, signal, hist = talib.MACD(df['Close'].values, fastperiod=12, slowperiod= 26, signalperiod=9)
#    MACD = pd.Series(macd,  index = df.index, name = 'MACD_12_26')     
#    #Signal line 9 EMA of EMA12- EMA26
#    MACDsign = pd.Series(signal,  index = df.index, name = 'MACDSign9')  
#    # Histo = diff from (EMA12-EMA26) - EMA9(EMA12- EMA26)
#    MACDdiff = pd.Series(hist,  index = df.index, name = 'MACDDiff')
    df['MFI'] = talib.MFI(df['High'].values, df['Low'].values,df['Close'].values, df['Volume'].values.astype(np.float64), timeperiod=14)
    
    n_fast = 12
    n_slow = 26
    nema = 9
    EMAfast = pd.Series(pd.Series.ewm(df['Close'], span = n_fast, min_periods = n_fast - 1).mean())  
    EMAslow = pd.Series(pd.Series.ewm(df['Close'], span = n_slow, min_periods = n_slow - 1).mean())  
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(pd.Series.ewm(MACD, span = nema, min_periods = nema-1).mean(), name = 'MACDSign_' + str(nema))  
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDDiff') 
         
    hm_days = 2 
    df['MACD_12_26'] = MACD
    df['MACDSign9'] = MACDsign
    df['MACDDiff'] = MACDdiff
    for i in range(1,hm_days+1):
        df['H_{}d'.format(i)] = df['High'].shift(i) 
        df['C_{}d'.format(i)] = df['Close'].shift(i) 
        df['L_{}d'.format(i)] = df['Low'].shift(i) 
        df['EMA3_{}d'.format(i)] = df['EMA3'].shift(i) 
        df['EMA6_{}d'.format(i)] = df['EMA6'].shift(i) 
        df['EMA18_{}d'.format(i)] = df['EMA18'].shift(i) 
        df['EMA50_{}d'.format(i)] = df['EMA50'].shift(i) 
        df['EMA100_{}d'.format(i)] = df['EMA100'].shift(i) 
        df['EMA200_{}d'.format(i)] = df['EMA200'].shift(i) 
        df['MACD_12_26_{}d'.format(i)] = df['MACD_12_26'].shift(i)
        df['MACDSign9_{}d'.format(i)] = df['MACDSign9'].shift(i)
        df['MACDDiff_{}d'.format(i)] = df['MACDDiff'].shift(i)
        
#    df.fillna(0, inplace=True)
       
#    3-18* LONG
#    H<H1AND
#    H1>H2AND
#    XAVGC3>XAVGC18AND
#    XAVGC3.1>XAVGC18.1AND
#    XAVGC3.2<XAVGC18.2AND
#    AVGV60>250000 
    
    df['18_LONG']= ((df['High'] < df['H_1d']) & (df['H_1d'] > df['H_2d'])  # Swing High pattern  
    & (df['Close'] > df['EMA18']) & (df['C_1d'] > df['EMA18_1d']) & (df['C_2d'] < df['EMA18_2d']))
    
    df['18_SHORT']= ((df['Low'] > df['L_1d']) & (df['L_1d'] < df['L_2d'])  # Swing Low pattern  
    & (df['Close'] < df['EMA18']) & (df['C_1d'] < df['EMA18_1d']) & (df['C_2d'] > df['EMA18_2d']))
    
    df['3_18_LONG']= ((df['High'] < df['H_1d']) & (df['H_1d'] > df['H_2d']) # Swing High pattern
    & (df['EMA3'] > df['EMA18']) & (df['EMA3_1d'] > df['EMA18_1d']) & (df['EMA3_2d'] < df['EMA18_2d']))
    
    
    df['3_18_SHORT']= ((df['Low'] > df['L_1d']) & (df['L_1d'] < df['L_2d']) # Swing Low pattern
    & (df['EMA3'] < df['EMA18']) & (df['EMA3_1d'] < df['EMA18_1d']) & (df['EMA3_2d'] > df['EMA18_2d']))
    
    
    df['3_6_LONG']= ((df['High'] < df['H_1d']) & (df['H_1d'] > df['H_2d']) # Swing High pattern
    & (df['EMA3'] > df['EMA6']) & (df['EMA3_1d'] > df['EMA6_1d']) & (df['EMA3_2d'] < df['EMA6_2d']))
    
    df['3_6_SHORT']= ((df['Low'] > df['L_1d']) & (df['L_1d'] < df['L_2d']) # Swing Low pattern
    & (df['EMA3'] < df['EMA6']) & (df['EMA3_1d'] < df['EMA6_1d']) & (df['EMA3_2d'] > df['EMA6_2d']))
    
    
    df['6_18_LONG']= ((df['High'] < df['H_1d']) & (df['H_1d'] > df['H_2d']) # Swing High pattern
    & (df['EMA6'] > df['EMA18']) & (df['EMA6_1d'] > df['EMA18_1d']) & (df['EMA6_2d'] < df['EMA18_2d']))
    
    df['6_18_SHORT']= ((df['Low'] > df['L_1d']) & (df['L_1d'] < df['L_2d']) # Swing Low pattern
    & (df['EMA6'] < df['EMA18']) & (df['EMA6_1d'] < df['EMA18_1d']) & (df['EMA6_2d'] > df['EMA18_2d']))
    
    
    
    df['3_50_LONG']= ((df['High'] < df['H_1d']) & (df['H_1d'] > df['H_2d']) # Swing High pattern
    & (df['EMA3'] > df['EMA50']) & (df['EMA3_1d'] > df['EMA50_1d']) & (df['EMA3_2d'] < df['EMA50_2d']))
    
    df['3_50_SHORT']= ((df['Low'] > df['L_1d']) & (df['L_1d'] < df['L_2d']) # Swing Low pattern
    & (df['EMA3'] < df['EMA50']) & (df['EMA3_1d'] < df['EMA50_1d']) & (df['EMA3_2d'] > df['EMA50_2d']))
    
    df['6_50_LONG']= ((df['High'] < df['H_1d']) & (df['H_1d'] > df['H_2d']) # Swing High pattern
    & (df['EMA6'] > df['EMA50']) & (df['EMA6_1d'] > df['EMA50_1d']) & (df['EMA6_2d'] < df['EMA50_2d']))
    
    
    df['6_50_SHORT']= ((df['Low'] > df['L_1d']) & (df['L_1d'] < df['L_2d']) # Swing Low pattern
    & (df['EMA6'] < df['EMA50']) & (df['EMA6_1d'] < df['EMA50_1d']) & (df['EMA6_2d'] > df['EMA50_2d']))
    
    df['18_50_LONG']= ((df['High'] < df['H_1d']) & (df['H_1d'] > df['H_2d']) # Swing High pattern
    & (df['EMA18'] > df['EMA50']) & (df['EMA18_1d'] > df['EMA50_1d']) & (df['EMA18_2d'] < df['EMA50_2d']))
    
    df['18_50_SHORT']= ((df['Low'] > df['L_1d']) & (df['L_1d'] < df['L_2d']) # Swing Low pattern
    & (df['EMA18'] < df['EMA50']) & (df['EMA18_1d'] < df['EMA50_1d']) & (df['EMA18_2d'] > df['EMA50_2d']))


    df['3_6_18_LONG']= ((df['High'] < df['H_1d']) & (df['H_1d'] > df['H_2d']) 
    & (df['EMA3'] > df['EMA6']) & (df['EMA3_1d'] > df['EMA6_1d']) & (df['EMA3_2d'] < df['EMA6_2d'])
    & (df['Close'] > df['EMA18']) & (df['C_1d'] > df['EMA18_1d']) & (df['C_2d'] < df['EMA18_2d']))
    
    df['3_6_18_SHORT']= ((df['Low'] > df['L_1d']) & (df['L_1d'] < df['L_2d'])
    & (df['EMA3'] < df['EMA6']) & (df['EMA3_1d'] < df['EMA6_1d']) & (df['EMA3_2d'] > df['EMA6_2d'])
    & (df['Close'] < df['EMA18']) & (df['C_1d'] < df['EMA18_1d']) & (df['C_2d'] > df['EMA18_2d']))
    
    df['MACD_SIGNAL_LONG']= ((df['High'] < df['H_1d']) & (df['H_1d'] > df['H_2d']) 
    & (df['MACD_12_26'] > df['MACDSign9']) & (df['MACD_12_26_1d'] > df['MACDSign9_1d'])
    & (df['MACD_12_26_2d'] < df['MACDSign9_2d']))
    
    df['MACD_SIGNAL_SHORT']= ((df['Low'] > df['L_1d']) & (df['L_1d'] < df['L_2d'])
    & (df['MACD_12_26'] < df['MACDSign9']) & (df['MACD_12_26_1d'] < df['MACDSign9_1d'])
    & (df['MACD_12_26_2d'] > df['MACDSign9_2d']))
    
    
    
    df['MACD_ZERO_LONG']= ((df['High'] < df['H_1d']) & (df['H_1d'] > df['H_2d']) 
    & (df['MACD_12_26'] > 0) & (df['MACD_12_26_1d'] > 0) & (df['MACD_12_26_2d'] < 0))
    
    df['MACD_ZERO_SHORT']= ((df['Low'] > df['L_1d']) & (df['L_1d'] < df['L_2d'])
    & (df['MACD_12_26'] < 0) & (df['MACD_12_26_1d'] < 0) & (df['MACD_12_26_2d'] > 0))
    # CONTEXTUAL RISK
    df['EMA_UP'] = ((df['EMA18'] > df['EMA50']) & (df['EMA50'] > df['EMA100']) & (df['EMA100'] > df['EMA200']))
    df['EMA_DOWN'] = ((df['EMA18'] < df['EMA50']) & (df['EMA50'] < df['EMA100']) & (df['EMA100'] < df['EMA200']))
    # MOMENTUM
    df['MACD_UP'] = ((df['MACD_12_26'] > df['MACDSign9']))
    df['MACD_DOWN'] = ((df['MACD_12_26'] < df['MACDSign9']))
    
    
    df['L18'] = (df['18_LONG'] & (df['EMA_UP'] | df['MACD_UP']))
    df['L3_18'] = (df['3_18_LONG'] & (df['EMA_UP'] | df['MACD_UP']))
    df['L3_6'] = (df['3_6_LONG'] & ( df['EMA_UP'] | df['MACD_UP']))
    df['L6_18'] = (df['6_18_LONG'] & (df['EMA_UP'] | df['MACD_UP']))
    df['L3_50'] = (df['3_50_LONG'] & (df['EMA_UP'] | df['MACD_UP']))
    df['L6_50'] = (df['6_50_LONG'] & (df['EMA_UP'] | df['MACD_UP']))
    df['L18_50'] = (df['18_50_LONG'] & (df['EMA_UP'] | df['MACD_UP']))
    df['L3_6_18'] = (df['3_6_18_LONG'] & (df['EMA_UP'] | df['MACD_UP']))    
    df['L_MACD_SIGNAL']=  (df['MACD_SIGNAL_LONG'] &  df['EMA_UP'])
    df['L_MACD_ZERO']=  (df['MACD_ZERO_LONG'] &  (df['EMA_UP'] | df['MACD_UP']))
    
    
    df['S18'] = (df['18_SHORT'] & (df['EMA_DOWN'] | df['MACD_DOWN']))
    df['S3_18'] = (df['3_18_SHORT'] & (df['EMA_DOWN'] | df['MACD_DOWN']))
    df['S3_6'] = (df['3_6_SHORT'] & ( df['EMA_DOWN'] | df['MACD_DOWN']))
    df['S6_18'] = (df['6_18_SHORT'] & (df['EMA_DOWN'] | df['MACD_DOWN']))
    df['S3_50'] = (df['3_50_SHORT'] & (df['EMA_DOWN'] | df['MACD_DOWN']))
    df['S6_50'] = (df['6_50_SHORT'] & (df['EMA_DOWN'] | df['MACD_DOWN']))
    df['S18_50'] = (df['18_50_SHORT'] & (df['EMA_DOWN'] | df['MACD_DOWN']))
    df['S3_6_18'] = (df['3_6_18_SHORT'] & (df['EMA_DOWN'] | df['MACD_DOWN']))    
    df['S_MACD_SIGNAL']=  (df['MACD_SIGNAL_SHORT'] &  df['EMA_DOWN'])
    df['S_MACD_ZERO']=  (df['MACD_ZERO_SHORT'] &  (df['EMA_DOWN'] | df['MACD_DOWN']))
    
    # 3 days checking: SH + 2 pullbacks
    hm_days = 10
    for i in range(1,hm_days+1):
        if (df['L18'].iloc[-i] | df['L3_18'].iloc[-i] | df['L3_6'].iloc[-i] 
            | df['L6_18'].iloc[-i] | df['L3_50'].iloc[-i]
            | df['L6_50'].iloc[-i] | df['L18_50'].iloc[-i] | df['L3_6_18'].iloc[-i] 
            | df['L_MACD_SIGNAL'].iloc[-i]
            | df['L_MACD_ZERO'].iloc[-i]):
            print(" Time for ninja trading ", str(i), " days before ", df.iloc[-i].name ,  ticker)
#        if ((df['High'].iloc[-i] < df['H_1d'].iloc[-i]) & (df['H_1d'].iloc[-i] > df['H_2d'].iloc[-i])):
#            print(" Swing high ", str(i), " days before ", df.iloc[-i].name , ticker)
            
    for i in range(1,hm_days+1):
        if ((df['L18'].iloc[-i] & (df['EMA18'].iloc[-i] > df['Low'].iloc[-i]) & (min(df['Open'].iloc[-i], df['Close'].iloc[-i]) > df['EMA18'].iloc[-i]))
            | (df['L3_18'].iloc[-i] & (df['EMA18'].iloc[-i] > df['Low'].iloc[-i]) & (min(df['Open'].iloc[-i], df['Close'].iloc[-i]) > df['EMA18'].iloc[-i]))
            | (df['L3_6'].iloc[-i] & (df['EMA6'].iloc[-i] > df['Low'].iloc[-i]) & (min(df['Open'].iloc[-i], df['Close'].iloc[-i]) > df['EMA6'].iloc[-i]))
            | (df['L6_18'].iloc[-i] & (df['EMA18'].iloc[-i] > df['Low'].iloc[-i]) & (min(df['Open'].iloc[-i], df['Close'].iloc[-i]) > df['EMA18'].iloc[-i]))
            | (df['L3_50'].iloc[-i] & (df['EMA50'].iloc[-i] > df['Low'].iloc[-i]) & (min(df['Open'].iloc[-i], df['Close'].iloc[-i]) > df['EMA50'].iloc[-i]))
            | (df['L6_50'].iloc[-i] & (df['EMA50'].iloc[-i] > df['Low'].iloc[-i]) & (min(df['Open'].iloc[-i], df['Close'].iloc[-i]) > df['EMA50'].iloc[-i]))
            | (df['L18_50'].iloc[-i] & (df['EMA50'].iloc[-i] > df['Low'].iloc[-i]) & (min(df['Open'].iloc[-i], df['Close'].iloc[-i]) > df['EMA50'].iloc[-i]))
            | (df['L3_6_18'].iloc[-i] & (df['EMA18'].iloc[-i] > df['Low'].iloc[-i]) & (min(df['Open'].iloc[-i], df['Close'].iloc[-i]) > df['EMA18'].iloc[-i]))):

            print(" Advanced ninja trading ", str(i), " days before ", df.iloc[-i].name ,  ticker)
#            print(" Open : ", df['Open'].iloc[-i], " Close: ", df['Close'].iloc[-i])
#            print(" Low: ", df['Low'].iloc[-i], " EMA 18 :", df['EMA18'].iloc[-i] )
       
    df['Signal'] = 1*(df['L18'] | df['L3_18'] | df['L3_6'] | df['L6_18'] | df['L3_50'] | df['L6_50'] | df['L18_50'] |  df['L3_6_18'] | df['L_MACD_SIGNAL'] | df['L_MACD_ZERO'])  +\
     -1*(df['S18'] | df['S3_18'] | df['S3_6'] | df['S6_18'] | df['S3_50'] | df['S6_50'] | df['S18_50'] |  df['S3_6_18'] | df['S_MACD_SIGNAL'] | df['S_MACD_ZERO']) 
#    
#    if ((df['MACD_UP'].iloc[-1] | df['EMA_UP'].iloc[-1]) & 
#         (min(df['Open'].iloc[-1], df['Close'].iloc[-1]) > df['EMA6'].iloc[-1]) & (df['EMA6'].iloc[-1] > df['Low'].iloc[-1])):
#        print(" Last day: Bouncing EMA 6", df.iloc[-1].name , ticker)  
#    if ((df['MACD_UP'].iloc[-1] | df['EMA_UP'].iloc[-1]) & 
#         (min(df['Open'].iloc[-1], df['Close'].iloc[-1])> df['EMA18'].iloc[-1]) & (df['EMA18'].iloc[-1] > df['Low'].iloc[-1])):
#        print(" Last day: Bouncing EMA 18", df.iloc[-1].name ,  ticker)
#    if ((df['MACD_UP'].iloc[-1] | df['EMA_UP'].iloc[-1]) & 
#         (min(df['Open'].iloc[-1], df['Close'].iloc[-1]) > df['EMA50'].iloc[-1]) & (df['EMA50'].iloc[-1] > df['Low'].iloc[-1])):
#        print(" Last day: Bouncing EMA 50", df.iloc[-1].name , ticker)
  
#    for i in range (1, len(df['Signal'])):
#        if (df['Signal'].iloc[i] ==0 & df['Close'] > )
        
   
    df['L18'] = 1*  df['L18']
    df['L3_18'] = 1*  df['L3_18']
    df['L3_6'] = 1*df['L3_6']
    df['L6_18'] = 1* df['L6_18']
    df['L3_50'] = 1* df['L3_50']
    df['L6_50'] = 1* df['L6_50']
    df['L18_50'] = 1* df['L18_50']
    df['L3_6_18'] = 1* df['L3_6_18']
    df['L_MACD_SIGNAL']= 1* df['L_MACD_SIGNAL']
    df['L_MACD_ZERO'] = 1* df['L_MACD_ZERO']
    
    df['S18'] = -1*  df['S18']
    df['S3_18'] = -1*  df['S3_18']
    df['S3_6'] = -1*df['S3_6']
    df['S6_18'] = -1* df['S6_18']
    df['S3_50'] = -1* df['S3_50']
    df['S6_50'] = -1* df['S6_50']
    df['S18_50'] = -1* df['S18_50']
    df['S3_6_18'] = -1* df['S3_6_18']
    df['S_MACD_SIGNAL']= -1* df['S_MACD_SIGNAL']
    df['S_MACD_ZERO'] = -1* df['S_MACD_ZERO']
    
    
    return df

   
def check_swing(df):    
    swing_signal = -1* ((df['Low'] > df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))) + \
                    1*((df['High'] < df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2)))    
    return swing_signal

def check_bounce(df, nema):
    bouncing_signal = 1* ((df['Open'] > df['EMA{}'.format(nema)]) 
                       & (df['Close'] > df['EMA{}'.format(nema)])
                       &  (df['Low'] < df['EMA{}'.format(nema)]))
    return bouncing_signal
    

def hedgefund_trading(ticker, start, end, realtime = False):
       
    file_path = symbol_to_path(ticker)
    df = pd.read_csv(file_path, index_col ="<DTYYYYMMDD>", parse_dates = True, 
                 usecols = ["<DTYYYYMMDD>", "<OpenFixed>","<HighFixed>","<LowFixed>","<CloseFixed>","<Volume>"], na_values = "nan")
    df = df.rename(columns = {'<DTYYYYMMDD>': 'Date', "<OpenFixed>": 'Open', '<HighFixed>': 'High',
                              '<LowFixed>': 'Low','<CloseFixed>' : 'Close', '<Volume>': 'Volume'})
    
    # columns order for backtrader type
    columnsOrder=["Open","High","Low","Close", "Volume", "OpenInterest"]
    # change the index by new index
    df = df.reindex(columns = columnsOrder)  
    # change date index to increasing order
    df = df.sort_index()   
    # take a part of dataframe
    df = df.loc[start:end]
    
    if realtime:
        actual_price = get_info_stock(ticker)
        today = datetime.datetime.today()
        next_date = today
        df.loc[next_date] = ({ 'Open' : actual_price['Open'].iloc[-1],
                        'High' : actual_price['High'].iloc[-1], 
                        'Low' : actual_price['Low'].iloc[-1],
                        'Close' : actual_price['Close'].iloc[-1],
                        'Volume' : actual_price['Volume'].iloc[-1],
                        'OpenInterest': np.nan})
#        df = df.rename(columns = {'index': '<DTYYYYMMDD>'})
#    return df
    
    df['EMA18'] = pd.Series(pd.Series.ewm(df['Close'], span = 18,  min_periods = 18-1).mean()) 
    
    n_fast = 3
    n_slow = 6
    nema = 20
    df['MACD_3_6'], df['MACDSign20'], df['MACDDiff3620'] =  compute_MACD(df, n_fast, n_slow, nema)
       
        
    n_fast = 50
    n_slow = 100
    nema = 9 
    df['MACD_50_100'], df['MACDSign9'], df['MACDDiff501009'] = compute_MACD(df, n_fast, n_slow, nema)
    

    # TREND TREND LONG
    df['TT_LONG']= ((df['Close'] > df['EMA18']) 
    & (df['MACD_3_6'] < df['MACDSign20']) & (df['MACD_3_6'] <0) & (df['MACDSign20'] > 0) 
    & (df['MACD_50_100'] > df['MACDSign9']) & (df['MACD_50_100'] > 0))
    # TREND TREND SHORT
    df['TT_SHORT']= ((df['Close'] < df['EMA18']) 
    & (df['MACD_3_6'] > df['MACDSign20']) & (df['MACD_3_6'] > 0) & (df['MACDSign20'] < 0) 
    & (df['MACD_50_100'] < df['MACDSign9']) & (df['MACD_50_100'] < 0))
   
    # TREND COUNTER TREND LONG
    df['TCT_LONG']= ((df['Close'] > df['EMA18']) 
    & (df['MACD_3_6'] < df['MACDSign20']) & (df['MACD_3_6'] <0) & (df['MACDSign20'] > 0) 
    & (df['MACD_50_100'] > df['MACDSign9']) & (df['MACD_50_100'] < 0)) 
    
    # TREND COUNTER TREND SHORT
    df['TCT_SHORT']= ((df['Close'] < df['EMA18']) 
    & (df['MACD_3_6'] > df['MACDSign20']) & (df['MACD_3_6'] > 0) & (df['MACDSign20'] < 0) 
    & (df['MACD_50_100'] < df['MACDSign9']) & (df['MACD_50_100'] > 0))
   
    
    
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
    
    
    
    df['1TT_SHORT'] = ((df['Low']> df['Low'].shift(1)) 
    & (df['Close'] < df['EMA18']) & (df['Close'].shift(1) > df['EMA18'].shift(1)))
    
    df['2TT_SHORT'] = ((df['Low'] > df['Low'].shift(1)) 
    & (df['Close'] < df['EMA18']) & (df['Open'] > df['EMA18'])
    & (df['Close'].shift(1) > df['EMA18'].shift(1)) & (df['Open'].shift(1) < df['EMA18'].shift(1)))
    
    df['B_SHORT'] = ((df['Low'] > df['Low'].shift(1)) 
    & (df['Close'] < df['EMA18']) & (df['Open'] < df['EMA18']) & (df['High'] > df['EMA18'])
    & (df['Close'].shift(1) < df['EMA18'].shift(1)) & (df['Open'].shift(1) < df['EMA18'].shift(1))
    & (df['High'].shift(1) > df['EMA18'].shift(1)))
     # MOMENTUM
     
    n_fast = 12
    n_slow = 26
    nema = 9
    MACD, MACDsign,_ = compute_MACD(df, n_fast, n_slow, nema)
    df['MACD_UP'] = (MACD > MACDsign)
    df['MACD_DOWN'] = (MACD < MACDsign)
    
    
    hm_days = 10
    for i in range(1,hm_days+1):
        if (df['LTT'].iloc[-i] | df['LCTT'].iloc[-i]):
            print(" Time for slingshot trading ", str(i), " days before ", df.iloc[-i].name ,  ticker)
        if (df['LTT_A'].iloc[-i] | df['LCTT_A'].iloc[-i]):
            print(" Advanced slingshot trading ", str(i), " days before ", df.iloc[-i].name ,  ticker)
        
    return df



def compute_MACD(df, n_fast, n_slow, nema = 9):  
    EMAfast = pd.Series(pd.Series.ewm(df['Close'], span = n_fast, min_periods = n_fast - 1).mean())  
    EMAslow = pd.Series(pd.Series.ewm(df['Close'], span = n_slow, min_periods = n_slow - 1).mean())  
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(pd.Series.ewm(MACD, span = nema, min_periods = nema-1).mean(), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))  
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))  
    return MACD, MACDsign, MACDdiff

