# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 08:55:26 2018

@author: sonng
"""

import numpy as np
import pandas as pd
import talib
from finance_util import symbol_to_path
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from collections import Counter
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import (BaggingRegressor, RandomForestRegressor, AdaBoostRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

###################################### Machine learning ###################################
def process_data_for_labels(df, ticker):
    hm_days = 7    
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    
    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
        
    df.fillna(0, inplace=True)
    return tickers, df


 
def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.03
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0




def extract_featuresets(df, ticker):
    tickers, df = process_data_for_labels(df, ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold, 
                                               df['{}_1d'.format(ticker)],                                             
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],                                             
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)] ))


    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    
    return X,y,df



def analysis_stock(tickers, df, start, end):
    for ticker in tickers:
        X, y, df = extract_featuresets(df, ticker)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        
        #clf = neighbors.KNeighborsClassifier()
        clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                                    ('knn',neighbors.KNeighborsClassifier()),
                                    ('rfor',RandomForestClassifier())])
        
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        print('accuracy:',confidence)
        predictions = clf.predict(X_test)
        print('predicted class counts:', Counter(predictions))
        if (confidence > 0.7):
            print(' Recommend invesment for next 5-7 days:', ticker)
            print('Predictions for next 5-7 days: ', clf.predict(X[-1:]))
            

###################################### Machine learning ###################################
    


def compute_indicator_bb(df_price, window = 20):
    rolling_mean = df_price['Close'].rolling(window = window, center = False).mean()
    rolling_std  = df_price['Close'].rolling(window = window, center = False).std()
    bb_value = (df_price['Close'] - rolling_mean)/(2 * rolling_std)
    return bb_value

def compute_indicator_volatility(df_price, timeperiod=5):
    '''
    Calculate volatility for the previous [timeperiod] days
    '''    
    daily_rets = df_price['Close'].pct_change(periods = timeperiod)
    daily_rets.iloc[0] = 0 
    sddr = daily_rets.rolling(window = timeperiod, center = False).std()
    return sddr



def compute_indicator_stoch(df):
    '''
    Calculate stoch
    Input: df[['close', 'high', 'low']]
    '''
    df['k'], df['d'] = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values, 
      fastk_period=14, slowk_period=1, slowd_period=5)
    return df['d']

def price_predictions(ticker, start, end, forecast_out):
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
    
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    bbwindow = 18
    vlwindow = 6
    mmtum = 5
    df['BB_Value'] = compute_indicator_bb(df, window = bbwindow)
    df['Volatility'] = compute_indicator_volatility(df, timeperiod = vlwindow)
    df['Momentum'] = talib.MOM(df['Close'].values, timeperiod = mmtum)
    df['OBV'] = talib.OBV(df['Close'].values, df['Volume'].values.astype(np.float64))
    df['MACD'],_,_ = talib.MACD(df['Close'].values, fastperiod=12, slowperiod= 26, signalperiod=9)
    _,df['STOCH']  = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values, fastk_period=14, slowk_period=1, slowd_period=5)
    df['MFI'] = talib.MFI(df['High'].values, df['Low'].values,df['Close'].values, df['Volume'].values.astype(np.float64), timeperiod=14)
#    df['EMA3'] = pd.Series(pd.Series.ewm(df['Close'], span = 3, min_periods = 3-1).mean()) 
#    df['EMA6'] = pd.Series(pd.Series.ewm(df['Close'], span = 6, min_periods = 6-1).mean()) 
#    df['EMA18'] = pd.Series(pd.Series.ewm(df['Close'], span = 18,  min_periods = 18-1).mean())
    df['PDI'] = talib.PLUS_DI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    df['NDI'] = talib.MINUS_DI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume','BB_Value', 
                        'Volatility', 'Momentum', 'MACD', 'STOCH', 'MFI', 'OBV']]
    
    #df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume','BB_Value']]
    df.fillna(method ="ffill", inplace = True)
    df.fillna(method ="backfill", inplace = True)
    
   
    forecast_col = 'Close'
    
    #inplace : boolean, default False
    # If True, fill in place. Note: this will modify any other views on this object,
    # (e.g. a no-copy slice for a column in a DataFrame).
    # Du bao 1% cua du lieu
       # Copy du lieu tu cot Adj. Close vao cot moi
    # Lenh Shift
    df['Target'] = df[forecast_col].shift(-forecast_out)
    # Lenh Drop loai bo label
    #axis : int or axis name: column
    # Whether to drop labels from the index (0 / ‘index’) or columns (1 / ‘columns’).
    X = np.array(df.drop(['Target'], 1))
    y_true = df[forecast_col][-forecast_out:]
    # Preprocessing Input Data
    X = preprocessing.scale(X)
    
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler()
    #X = scaler.fit_transform(X)
    
    
    # Tach gia tri X va X_lately ra khoi chuoi
    X_lately = X[-forecast_out:]
    
    X = X[:-forecast_out]
    # Loai bo cac gia tri NA
    # df.dropna(inplace=True)
    # Target la vector y lay tu cot label
    y = np.array(df['Target'].dropna())
    
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    #from sklearn.preprocessing import MinMaxScaler
    #from sklearn.preprocessing import StandardScaler
    #scaler = MinMaxScaler()
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    #X_lately = scaler.transform(X_lately)
    
   
    n_neighbors = 5
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
    knn.fit(X_train, y_train)
    print('Train score KNN: ', knn.score(X_train, y_train), 'Test score KNN : ', knn.score(X_test, y_test))
    forecast_set = knn.predict(X_lately)
    print('Price for next {} days'.format(forecast_out), forecast_set)
   
   
    bagging = BaggingRegressor(
            DecisionTreeRegressor(), 
            n_estimators=50,        
            random_state=50
        )
    bagging.fit(X_train, y_train)
    print('Train score BAG: ', bagging.score(X_train, y_train), 'Test score BAG : ', bagging.score(X_test, y_test))
    forecast_set = bagging.predict(X_lately)
    print('Price for next {} days'.format(forecast_out), forecast_set)
   
    rf = RandomForestRegressor(
            n_estimators=50,   
            random_state=50
        )
    rf.fit(X_train, y_train)
    print('Train score RF: ', rf.score(X_train, y_train), 'Test score RF : ', rf.score(X_test, y_test))
    forecast_set = rf.predict(X_lately)
    print('Price for next {} days'.format(forecast_out), forecast_set)

   
    adaboost = AdaBoostRegressor(neighbors.KNeighborsRegressor(n_neighbors=5),
                              n_estimators=30, random_state=0)
    
    #adaboost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
    #                          n_estimators=30, random_state=0)
    adaboost.fit(X_train, y_train)
    print('Train score Ada: ', adaboost.score(X_train, y_train), 'Test score Ada : ', adaboost.score(X_test, y_test))
    forecast_set = adaboost.predict(X_lately)
    print('Price for next {} days'.format(forecast_out), forecast_set)
 
    