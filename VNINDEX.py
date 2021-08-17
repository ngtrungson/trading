# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:35:57 2017

@author: sonng
"""
# import datetime
from datetime import datetime
from finance_util import get_info_stock_cp68_mobile, get_data, get_RSI, fill_missing_values, optimize_portfolio, compute_portfolio, plot_normalized_data, \
                         get_data_from_cophieu68_openwebsite, get_data_from_SSI_website, analysis_alpha_beta
from strategy import process_data, momentum_strategy,  hung_canslim
# from plot_strategy import plot_hedgefund_trading, plot_ninja_trading, plot_trading_weekly,plot_shortselling_trading, plot_canslim_trading
# from machine_learning import price_predictions, ML_strategy
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import sys
import time
import os
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def my_portfolio(start = "2018-7-10" , end = "2018-10-16"):
    dates = pd.date_range(start, end)
    symbols = ['PHC','MBB','GEX', 'MBS', 'NDN', 'PNJ', 'STB', 'PVD', 'VRC', 'VIX']
#    symbols = ['PHC','MBB','GEX', 'MBS']
    df_data = get_data(symbols, dates, benchmark = '^VNINDEX')  # get data for each symbol
    fill_missing_values(df_data)
    plot_normalized_data(df_data, title= " Daily porfolio value with VNINDEX ", xlabel="Date", ylabel= " Normalized price ")


    

def portfolio_management():
    df = pd.DataFrame()
    tickers = ['HVN','VRE','DCM', 'KSB']
    # chu y xu ly cac CP nhu PVS (co kha nang thoat hang), ACB, MBS, NVB(ngam lau dai doi thoi),  (HAR, DVN, VIX): sieu lo
    buy_price = [25, 25.75, 8.74, 25.55]
    shares_number = [2000, 2000, 6000, 2000]
    
    low_candle = [24.1, 25.15, 8.4, 24.3]
    
    df['Ticker'] = tickers
    df = df.set_index('Ticker')    
    df['Buy'] = buy_price
    df['Cut_loss'] = df['Buy']*0.97
    df['Target'] = df['Buy']*1.05
    df['Shares'] = shares_number
    df['Values'] = df['Buy']*df['Shares']
    df['Low'] = low_candle
    df['MinPort'] = df['Low']*df['Shares']
    df['MaxLoss'] = df['MinPort'] - df['Buy']*df['Shares']
    #df['']
    actual_price = []
    for ticker in tickers:        
        print(ticker)
        try:
            df_temp = get_info_stock_cp68_mobile(ticker)
            actual_price.append(df_temp['Close'].iloc[-1]) 
    #        print(df_temp['Close'].iloc[-1])
    #        print(df['Target'][ticker])
            if (df_temp['Close'].iloc[-1] >= df['Target'][ticker]):
                print(' SELL TARGET signal : actual price:' , df_temp['Close'].iloc[-1], 'target:', df['Target'][ticker])
            else:
                if (df_temp['Close'].iloc[-1] <= df['Cut_loss'][ticker]) &  (df_temp['Close'].iloc[-1] <= df['Low'][ticker]):
                    print(' CUT_LOSS signal : actual price:' , df_temp['Close'].iloc[-1], 'cut loss: ', min(df['Cut_loss'][ticker], df['Low'][ticker]))
                else:
                    print(' Continue HOLDING : actual price:', df_temp['Close'].iloc[-1], ' actual/buy ratio: ' , round(df_temp['Close'].iloc[-1]/df['Buy'][ticker],3))
        except Exception as e:
            print(" Error in symbol : ", ticker) 
            print(e)
            pass
    
    df['Current'] = actual_price
    df['Port_val'] = df['Current']*df['Shares']
    df['Diff_val'] = df['Port_val'] - df['Values']
    portfolio_diff =  df['Diff_val'].sum(axis = 0)
    
    print(' Porfolio value :', df['Diff_val'].values, ' Sum: ', portfolio_diff)
    print(' Max loss:', df['MaxLoss'].sum(axis=0))
#    return df
    




def getliststocks(typestock = "^VNINDEX"):
    benchmark = ["^VNINDEX", "^HASTC", "^UPCOM","VNINDEX","UPINDEX","HNXINDEX"]
    futures = ["VN30F1M", "VN30F2M", "VN30F1Q", "VN30F2Q"]
    
    nganhang = ['ACB','CTG','VPB','VCB','NVB', 'LPB', 'VIB', 'BID','HDB', 'EIB', 'MBB', 'SHB', 'STB']
    
    thuysan = ['VHC', 'ANV']
    daukhi = ['PVS','PVD','PVB','PLX', 'BSR', 'POW','TDG','GAS', 'PLC']
    batdongsan =['HAR', 'HLD', 'DXG', 'NVL', 'KDH', 'CEO', 'VIC','NDN','PDR','VPI', 'VRE','ASM','EVG','NBB' ]
    chungkhoan = ['HCM', 'SSI', 'VND', 'TVB','TVS', 'BVS','MBS','FTS', 'HCM', 'VIX', 'ART','SHS', 'VCI']
    baohiem = ['BVH', 'BMI']
    xaydung = ['CTD', 'HBC', 'PHC','DXG']
    duocpham = ['DVN', 'DHG']
    hangkhong = ['HVN','VJC']
    thep = ['HSG', 'HPG', 'NKG']
    cntt = ['MWG', 'FPT']
    nhua = ['BMP','AAA']
    vatlieuxd = ['VCS']
    caosu = ['PHR', 'DRC','GVR']
    anuong = ['VNM', 'SAB']
    
    stocks2019 =['SZC', 'VGI','GVR','CTR','VTP']
    
# Danh muc co phieu khong co quy mua, han che va khong nen mua
# VIX MBS VRC HTN CMX SZC IDC ACL FIR HAX  PMG L14 AMV
# TVC VPG DGC DVN SHI PHC
# NVB TCH VPI CVT VNG 
# (GVR TDM MSH ACV BWE NTC VEA CTR ANV)
    
    symbolsVN30 = ['BID','BVH','CTD', 'CTG', 'DPM', 'EIB','FPT', 'GAS', 
                   'HDB','HPG', 'MBB', 'MSN', 'MWG', 'NVL', 'PLX','PNJ','POW',
                   'REE', 'ROS', 'SAB', 'SBT', 'SSI', 'STB', 'TCB', 'VCB', 'VHM',
                   'VIC', 'VJC', 'VNM', 'VPB','VRE']
    
    
    symbolsHNX = ['NDN','PVS','VCG','VCS', 'TNG','SHB','SHS', 'PLC','NTP' ,'VND']
    
    symbolsVNI = [ 'ANV',  "BWE",  'CMG', "AGG", "HTN", "TIP",
                   "BID", "BMI", "BMP", "BVH",  "CTD", "CSV", "CTG", 'D2D',
               "DHG",  "DPM",  "DRC", "DXG", 'DGW', 'DBC',
                "FCN",  'FMC', "FPT", "GAS", "GMD",  
                  "HT1",   "LPB", "HSG", "DVP", "TPB","TCL", "TV2",
                "HDG", "HCM", "HPG", 'LHG', 'HDC',
                "IJC",  "KBC",  "KDH",
               "MBB", "MSN", "MWG",  "NLG",  "NVL",
                "PVT","PVD","PHR","PDR", "PNJ",  "PC1",   "PLX",
                "PPC",  "REE",
                "SJS","STB", "SSI", "SBT", 
                "VNM", "VHC", "VIC", "VCB", "VSC", "VJC", 
                   'GEX', "VIB", 'HAH', 'SMC','HAH','ITD','OCB','FTS','PTB',
                'TCM',  'AAA',  'VGC',
                'VPB','VRE',  "HDB",  "ACB",
                'NTL', 'AST', 'VHM',  'TCB', 
                'DHC', 'TDM', 'DCM', 'LCG', "VIX",
                   'SZL', 'GVR', 'GIL', 'BFC', 'SZC', 
                'IMP', 'MSH', 'POW','TCH','VCI','DIG','KSB','FRT','CRE','PET','DGC']
    
    # 'SMC','HAH','ITD','OCB','FTS','PTB'
    
    symbolsUPCOM = ['QNS',  'ACV','VGI','CTR','VTP','VEA'] 
   
#    symbolsHNX = ['TNG', 'NVB',  'L14',  
#                  'ACB',  'CEO', 'DBC',  'MBS', 'NDN', 'PVI', 'PVB',
#                  'PVS',  'VCG','VCS',  'VIX', 'TVC', 
#                  'VPI', 'AMV', 'DGC']
#    
#    symbolsVNI = [ 'BFC','STK','CII','PHC','APC', 'ANV',  "BWE", 'HTN', 'C32', 'ACL', 'LCG',
#                   "BID", "BMI", "BMP", "BVH",  'CTI', "CTD", "CSV", "CTG", 'CMX','D2D',
#               "DHG",  "DPM",  "DRC", "DXG", 'DGW',
#                "FCN",  'FMC', "FPT", "GAS", "GMD", "GTN", 
#                'HAX',  "HNG",  "HT1",   'DPR',
#                "HDG", "HCM", "HPG", "HBC", 'LHG', 'HDC',
#                "IJC",  "KBC", "KSB",  "KDH",
#               "MBB", "MSN", "MWG",  "NLG", "NT2", "NVL",
#                "PVT","PVD","PHR","PDR","PTB", "PNJ",  "PC1",   "PLX",
#                "PPC",  "REE",  
#                'SHI',"SAM","SJD","SJS","STB", "SSI", "SBT", "SAB", 'PMG',
#                "VNM", "VHC", "VIC", "VCB", "VSC", "VJC", 
#                 'PAN','TCH', 'TDH',  'GEX', 
#                'TCM',  'AAA', 'VRC',  'HVN', 'VGC',
#                'EIB','VPB','VRE','ROS',"VND", "HDB",  "CVT",'VNG',
#                'NTL', 'AST','HAH', 'VHM', 'VPG',  'TPB', 'TCB',
#                'HPX','FIR','CRE','NAF', 'DHC', 'MSH','TDM', 'SZC']
#    
#    symbolsUPCOM = ['QNS',  'ACV',   "DVN",  'VGI','GVR','CTR','VTP',
#                    'VGT', 'VIB', 'POW',  'MPC', 'VEA', 'GEG', 'NTC', 'IDC']
    
    if typestock == "ALL":
        symbols = benchmark + symbolsVNI + symbolsHNX + symbolsUPCOM 
    if typestock == "^VNINDEX":
        symbols = symbolsVNI + [typestock]
    if typestock == "^HASTC":
        symbols = symbolsHNX + [typestock]
    if typestock == "^UPCOM":
        symbols = symbolsUPCOM + [typestock]
    if typestock == "TICKER":
        symbols = symbolsVNI + symbolsHNX + symbolsUPCOM + benchmark
    if typestock == "BENCHMARK":
        symbols = benchmark
    if typestock == "VN30":
        symbols = symbolsVN30
#    symbols =  high_cpm
    symbols = pd.unique(symbols).tolist()
    
    symbols = sorted(symbols)
    
    
    
    return symbols
    
    
    

def get_stocks_highcpm(download = True, source = "ssi"):

    data = pd.read_csv('fundemental_stocks_all.csv', parse_dates=True, index_col=0)
    df = data.query("MeanVol_10W > 100000")
    df = df.query("FVQ > 0")
    df = df.query("CPM > 1.4")
    df = df.query("EPS > 0")
    tickers  = df.index
    
    if download:
        if source == "cp68":
            get_data_from_cophieu68_openwebsite(tickers)
        else:
           get_data_from_SSI_website(tickers) 
    
    
    return tickers
    
    
def analysis_trading(tickers, start, end, update = False, nbdays = 15, source = "cp68", trade = 'Long'):
    
    if tickers == None:
        tickers = getliststocks(typestock = "TICKER")
        
    if tickers == 'VN30':
        tickers = getliststocks(typestock = "VN30")
        
#    data = pd.read_csv('fundemental_stocks_all.csv', parse_dates=True, index_col=0)
#    data['Diff_Price'] = data['Close'] - data['EPS']*data['PE']/1000
#    data['EPS_Price'] = data['EPS']/data['Close']/1000
#    df = data.query("MeanVol_10W > 80000")
#    df = data.query("MeanVol_13W > 80000")
#    df = df.query("EPS > 1000")
#    df = df.query("ROE > 15")
#       
#    canslim_symbol = df.index.tolist()
#    
#    tickers = canslim_symbol
    # result = pd.DataFrame([['Ticker', 'Advise']])
    result = pd.DataFrame(columns =['Ticker', 'Advise','PCT', 'Close'])
    result = result.set_index('Ticker')
    for ticker in tickers:
#        print(" Analysing ..." , ticker)
        try:
#            ninja_trading(ticker, start, end, realtime = update, source = source)
#            hedgefund_trading(ticker, start, end, realtime = update, source = source)
#            hung_canslim(ticker, start, end, realtime = update, source = source, ndays = 5, typetrade = 'MarkM_tickers')#           
             res = hung_canslim(ticker, start, end, realtime = update, source = source, ndays = nbdays, typetrade = trade)
             
             if len(res) > 1:
                 # result = result.append([res])
                 result.loc[res[0]] = [res[1], 100*res[2], res[3]]
#            hung_canslim(ticker, start, end, realtime = update, source = source, ndays = 3, typetrade = 'Short')
#            mean_reversion(ticker, start, end, realtime = update, source = source)
#            bollinger_bands(ticker, start, end, realtime = update, source = source)
#            short_selling(ticker, start, end, realtime = update, source = source, ndays = 2, typetrade = 'Short')
        except Exception as e:
            print (e)
            print("Error in reading symbol: ", ticker)
            pass
    return result

def canslim_strategy(ticker, start, end, update = False, source = "cp68"):               
    df = momentum_strategy(ticker, start, end, realtime = update, source = source)#
    df = df.reset_index()
    
    df.loc[:,'day'] = df['Date'].values
    # df['day'] = df['day'].map(mdates.date2num)
    
    buy = df[df['Buy'] == 1]
    sell = df[df['Sell'] == -1]
#    df['Date'] = df['Date'].map(mdates.date2num)
    
    plt.plot(sell['day'], sell['Close'].values, "ro")
    plt.plot(buy['day'], buy['Close'].values, "go")
    plt.plot(df['day'], df['Close'].values)
    ax = plt.gca()    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.legend(['sell', 'buy', 'close'], loc='upper right')
    plt.xlabel("Date")
    plt.ylabel("1K VND price")
    plt.title("Canslim trading for {} ".format(ticker))
    plt.grid(True)
    plt.show()   

    return df         
    
   
    
    
def analysis_market(tickers, start, end, update = False, market = "^VNINDEX", source = "cp68"):
    
    if tickers == None:
        tickers = getliststocks(typestock = market)
        
    if update:
        end_date = datetime.datetime.today()
        dates = pd.date_range(start, end_date)  # date range as index
        df_data = pd.DataFrame(index=dates)
        df_volume = pd.DataFrame(index=dates)
    else:
        dates = pd.date_range(start, end)  # date range as index
        df_data = pd.DataFrame(index=dates)
        df_volume = pd.DataFrame(index=dates)
        
    for ticker in tickers:
#        print(" Analysing ..." , ticker)
#        try:
#           
            df = process_data(ticker = ticker, start = start, end = end, realtime = update, source = source)
            
           
#            print(df_temp.head())
            
            df_data = df_data.join(df['Close'])
            df_data = df_data.rename(columns={'Close': ticker})
            
            
            df_volume = df_volume.join(df['Volume'])
            df_volume = df_volume.rename(columns={'Volume': ticker})
            
#        except Exception as e:
#            print (e)
#            print("Error in reading symbol: ", ticker)
#            pass
    
    fill_missing_values(df_data) 
    fill_missing_values(df_volume) 
    
    marketVNI = df_data[tickers].pct_change() 
    advances = marketVNI[marketVNI > 0] 
    declines = marketVNI[marketVNI <= 0] 
    dec = advances.isnull().sum(axis=1)
    adv = declines.isnull().sum(axis=1)
    
    
    df_market = pd.DataFrame(index = marketVNI.index)
    
#    df_market[market+'Volume'] = df_volume[market]
#    df_market[market+'PCT_Volume'] = df_volume[market].pct_change() *100
#    df_market[market+'PCT_Index'] = df_data[market].pct_change() *100
#    df_market[market+'Adv_Dec'] = adv - dec
#    df_market[market+'Dec/Adv'] = dec/adv
#    strength = pd.Series(index = marketVNI.index)
#    strength[(df_market[market+'Adv_Dec']> 0) & (df_market[market+'PCT_Index'] > 0)] = 1
#    strength[(df_market[market+'Adv_Dec']< 0) & (df_market[market+'PCT_Index'] < 0)] = -1
#    strength[(df_market[market+'Adv_Dec']< 0) & (df_market[market+'PCT_Index'] > 0)] = 0
#    df_market[market+'Strength'] = strength 
        
    return df_market

def analysis_all_market(tickers, start, end, update = False, source = "cp68"):
    
    hsx_market = analysis_market(tickers = None, start = start, end = end, update = update, market = "^VNINDEX", source = source)
#    hnx_market = analysis_market(tickers = None, start = start, end = end, update = update, market = "^HASTC", source = source)
#    upcom_market = analysis_market(tickers = None, start = start, end = end, update = update, market = "^UPCOM", source = source)
#    
    df_market = pd.DataFrame(index=hsx_market.index)
    df_market = df_market.join(hsx_market)
#    df_market = df_market.join(hnx_market)
#    df_market = df_market.join(upcom_market)    
    
    return df_market


def get_csv_data(source = "cp68"):

    symbols = getliststocks(typestock = "ALL")
     
    if source == "cp68":
        get_data_from_cophieu68_openwebsite(symbols)
    else:
       get_data_from_SSI_website(symbols) 
    return symbols
 

            
# def predict_stocks(tickers, start, end):
#     for ticker in tickers:
#         print('Prediction of ticker .................' , ticker)
#         price_predictions(ticker, start, end, forecast_out = 5)
#         print(' End of prediction ticker ...................', ticker)

def analysis_stocks(start_date, end_date, realtime = True, source ='cp68'):
    
    hsx_res, hsx_data, hsx_market = passive_strategy(start_date = start_date, end_date = end_date, market = "^VNINDEX", realtime = realtime, source = source)
    hnx_res, hnx_data, hnx_market = passive_strategy(start_date = start_date, end_date = end_date, market = "^HASTC", realtime = realtime, source = source)
    upcom_res, upcom_data, upcom_market = passive_strategy(start_date = start_date, end_date = end_date, market = "^UPCOM", realtime = realtime, source = source)
    
    
    frames = [hsx_res, hnx_res, upcom_res]
#    frames = [hnx_res]
    df_result  = pd.concat(frames)
    # df_rsi = pd.concat([hsx_rsi, hnx_rsi, upcom_rsi])
    df_market = pd.DataFrame(index=hsx_market.index)
    df_market = df_market.join(hsx_market)
    df_market = df_market.join(hnx_market)
    df_market = df_market.join(upcom_market)
    
    return df_result, df_market

def analysis_VN30(start_date, end_date):
    
    symbolsVN30 = getliststocks(typestock = "VN30")
    hsxvn30_res, hsxvn30_data, hsxvn30_market = passive_strategy(start_date = start_date, end_date = end_date, market = "^VNINDEX", symbols = symbolsVN30)
    
    return hsxvn30_res

#Relative Strength Index  




def passive_strategy(start_date, end_date, market = "^VNINDEX", symbols = None, realtime = False, source = 'cp68'):

    if symbols == None:
        symbols = getliststocks(typestock = market)
        
    if realtime:
        end_date = datetime.datetime.today()
        
    dates = pd.date_range(start_date, end_date)  # date range as index
    df_data = get_data(symbols, dates, benchmark = market, realtime = realtime, source = source)  # get data for each symbol
    # Fill missing values
    fill_missing_values(df_data)
    
    df_volume = get_data(symbols, dates, benchmark = market, colname = '<Volume>', realtime = realtime, source = source)  # get data for each symbol
    df_high = get_data(symbols, dates, benchmark = market, colname = '<High>', realtime = realtime, source = source)
    df_low = get_data(symbols, dates, benchmark = market, colname = '<Low>', realtime = realtime, source = source)
    df_rsi = get_RSI(symbols, df_data)
#    covariance = numpy.cov(asset , SPY)[0][1]  
#    variance = numpy.var(asset)
#    
#    beta = covariance / variance 
    df_volume = df_volume.fillna(0)
    df_value = (df_volume*df_data).fillna(0)
    valueM30 = df_value.rolling(window =30).mean()
    volumeM30 = df_volume.rolling(window =30).mean()
    
    vol_mean = pd.Series(df_volume.mean(),name = 'Volume')
    # max_high = pd.Series(df_high.max(), name = 'MaxHigh')
    # min_low = pd.Series(df_low.min(), name = 'MinLow')
    # cpm = pd.Series(max_high/min_low, name = 'CPM')
    value_mean = pd.Series(df_value.mean(), name = 'ValueMean')
    
    
    

    
    # Assess the portfolio
    
    # allocations, cr, adr, sddr, sr  = optimize_portfolio(sd = start_date, ed = end_date,
    #     syms = symbols,  benchmark = market, gen_plot = False)

    #  # Print statistics
    # print ("Start Date:", start_date)
    # print ("End Date:", end_date)
    # print ("Symbols:", symbols)
    # print ("Optimal allocations:", allocations)
    # print ("Sharpe Ratio:", sr)
    # print ("Volatility (stdev of daily returns):", sddr)
    # print ("Average Daily Return:", adr)
    # print ("Cumulative Return:", cr)
    
    # investment = 50000000
    df_result = pd.DataFrame(index = symbols)    
    # df_result['Opt allocs'] = allocations
    # df_result['Cash'] = allocations * investment
    df_result['Ticker'] = symbols
    df_result['Close'] = df_data[symbols].iloc[-1,:].values
    df_result['PCT_C'] = 100*(df_data[symbols].iloc[-1,:].values - df_data[symbols].iloc[0,:].values)/df_data[symbols].iloc[0,:].values
    df_result['Volume'] = df_volume[symbols].iloc[-1,:].values
    # df_result['VolMean'] = vol_mean[symbols]
    # df_result['VolMA30'] = volumeM30[symbols].iloc[-1,:].values
    df_result['Value'] = df_result['Close'] * df_result['Volume']   
    # df_result['ValMean'] = value_mean[symbols]    
    # df_result['ValMA30'] = valueM30[symbols].iloc[-1,:].values
    #    df_result['MaxH'] = max_high
#    df_result['MinL'] = min_low
    # df_result['CPM'] = cpm[symbols]
    # df_result['Shares'] = round(df_result['Cash']/df_result['Close'].values/1000,0)    
    df_result ['Volatility'] = df_data[symbols].pct_change().std() 
    alpha_beta = analysis_alpha_beta(df_data, symbols, market)
    df_result['Alpha'] = alpha_beta['Alpha']
    df_result['Beta'] = alpha_beta['Beta']
    df_result ['PCT_3D'] = df_data[symbols].pct_change().iloc[-4,:].values*100
    df_result ['PCT_2D'] = df_data[symbols].pct_change().iloc[-3,:].values*100
    df_result ['PCT_1D'] = df_data[symbols].pct_change().iloc[-2,:].values*100
    df_result ['PCT_0D'] = df_data[symbols].pct_change().iloc[-1,:].values*100
    df_result['PCT_Sum4D'] = df_result ['PCT_3D'] + df_result ['PCT_2D'] + df_result ['PCT_1D'] + df_result ['PCT_0D']
    
    df_result['Vol_ratio'] = df_volume[symbols].iloc[-1,:].values/volumeM30[symbols].iloc[-1,:].values
   
    df_result['Vrx0D'] = df_result['Vol_ratio']*df_result ['PCT_0D']
    
    
    
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
    rsi_change = df_rsi[symbols].pct_change()
    df_result['RSI_1D'] = rsi_change.iloc[-1,:].values*100
    df_result['RSI_2D'] = rsi_change.iloc[-2,:].values*100
    
    marketVNI = df_data[symbols].pct_change() 
    advances = marketVNI[marketVNI > 0] 
    declines = marketVNI[marketVNI <= 0] 
    dec = advances.isnull().sum(axis=1)
    adv = declines.isnull().sum(axis=1)
    
    df_market = pd.DataFrame(index = marketVNI.index)
    df_market[market+'Volume'] = df_volume[market]
    df_market[market+'PCT_Volume'] = df_volume[market].pct_change() *100
    df_market[market+'PCT_Index'] = df_data[market].pct_change() *100
#    df_market['Adv'] = adv
#    df_market['Dec'] = dec
    df_market[market+'Adv_Dec'] = adv - dec
    df_market[market+'Dec/Adv'] = dec/adv
    strength = pd.Series(index = marketVNI.index, dtype="float64")
    strength[(df_market[market+'Adv_Dec']> 0) & (df_market[market+'PCT_Index'] > 0)] = 1
    strength[(df_market[market+'Adv_Dec']< 0) & (df_market[market+'PCT_Index'] < 0)] = -1
    strength[(df_market[market+'Adv_Dec']< 0) & (df_market[market+'PCT_Index'] > 0)] = 0
    df_market[market+'Strength'] = strength 
#    np.where((df_data[symbols].pct_change() > 0), 1, -1)
    
    
    
    return df_result, df_data, df_market


def active_strategy(start_date, end_date, update = False, source = "cp68", market = "^VNINDEX"):

    symbols = getliststocks(typestock = market)
    
    for ticker in symbols:
        try:
#            ninja_trading(ticker, start, end, realtime = update, source = source)
#            hedgefund_trading(ticker, start, end, realtime = update, source = source)
            hung_canslim(ticker, start = start_date, end = end_date, realtime = update, source = source, market = market, ndays = 2, typetrade = 'Long')
#            mean_reversion(ticker, start, end, realtime = update, source = source)
#            bollinger_bands(ticker, start, end, realtime = update, source = source)
#            short_selling(ticker, start, end, realtime = update, source = source)
        except Exception as e:
            print (e)
            print("Error in reading symbol: ", ticker)
            pass

def rebalancing_porfolio(symbols = None, bench = '^VNINDEX'):

   
    start0 = "2018-1-2"
    end0 = "2019-1-2"
    allocations, cr, adr, sddr, sr  = optimize_portfolio(sd = start0, ed = end0,
            syms = symbols,  benchmark = bench, gen_plot = True)
    print ("Optimize start Date:", start0)
    print ("Optimize end Date:", end0) 
    print ("Optimize volatility (stdev of daily returns):", sddr)
    print ("Optimize average Daily Return:", adr)
    print ("Optimize cumulative Return:", cr)
    print(" -----------------------------------------------------")
    start_date_list = ["2019-1-3", "2019-7-3"]
    end_date_list = ["2019-7-2",  "2020-4-1"]
    for start, end in zip(start_date_list, end_date_list):    
        
        cr, adr, sddr, sr  = compute_portfolio(sd = start, ed = end,
            syms = symbols, allocs = allocations, benchmark = bench, gen_plot = True)
        print ("Start Date:", start)
        print ("End Date:", end) 
        print ("Volatility (stdev of daily returns):", sddr)
        print ("Average Daily Return:", adr)
        print ("Cumulative Return:", cr)  
        print(" -----------------------------------------------------")
        allocations, cr, adr, sddr, sr  = optimize_portfolio(sd = start, ed = end,
            syms = symbols,  benchmark = bench, gen_plot = False)
        print ("Optimize volatility (stdev of daily returns):", sddr)
        print ("Optimize average Daily Return:", adr)
        print ("Optimize cumulative Return:", cr)
        print(" -----------------------------------------------------")
        
    
    
    
    
    # Out of sample testing optimisation algorithm
    
    end_date = "2021-8-6"
    start_date = "2020-4-2"
    
    cr, adr, sddr, sr  = compute_portfolio(sd = start_date, ed = end_date,
            syms = symbols, allocs = allocations, benchmark = bench, gen_plot = True)
    print("....................... Out of sample performance .................")
    print ("Start Date:", start_date)
    print ("End Date:", end_date) 
    print ("Volatility (stdev of daily returns):", sddr)
    print ("Average Daily Return:", adr)
    print ("Cumulative Return:", cr)  
    # Assess the portfolio
    investment = 600E6
    df_result = pd.DataFrame(index = symbols)    
    df_result['Opt allocs'] = allocations
    df_result['Cash'] = allocations * investment

    dates = pd.date_range(start_date, end_date)  # date range as index
    df_data = get_data(symbols, dates, benchmark = bench)  # get data for each symbol
    
   
    df_high = get_data(symbols, dates, benchmark = bench, colname = '<High>')
    df_low = get_data(symbols, dates, benchmark = bench, colname = '<Low>')
    
    max_high = pd.Series(df_high.max(), name = 'MaxHigh')
    min_low = pd.Series(df_low.min(), name = 'MinLow')
    cpm = pd.Series(max_high/min_low, name = 'CPM')
    volatility = df_data[symbols].pct_change().std()  
    
    # Fill missing values
            
    df_result['Close'] = df_data[symbols].iloc[-1,:].values    
    df_result['CPM'] = cpm[symbols]
    df_result['Shares'] = round(df_result['Cash']/df_result['Close'].values/1000,0)
    df_result ['Volatility'] = volatility
    
    alpha_beta = analysis_alpha_beta(df_data, symbols, market = bench)
    df_result['Alpha'] = alpha_beta['Alpha']
    df_result['Beta'] = alpha_beta['Beta']
    
    relative_strength = 40*df_data[symbols].pct_change(periods = 63).fillna(0) \
                     + 20*df_data[symbols].pct_change(periods = 126).fillna(0) \
                     + 20*df_data[symbols].pct_change(periods = 189).fillna(0) \
                     + 20*df_data[symbols].pct_change(periods = 252).fillna(0)    
    
    df_result ['RSW'] = relative_strength.iloc[-1,:].values
   
    return df_result
    
# from IPython.display import clear_output

    
if __name__ == "__main__":
    # import sys
    # old_stdout = sys.stdout
    # sys.stdout=open("logging.txt","w")
    # GTVT, Logistics: GMD, VSC, PVT, HVN, VJC
    # Dau khi: PVD, PVS, GAS, PLX
    # Che tao, trang suc: PNJ
    # BDS VRE, VHM, VIC, NLG, PDR, NVL
    # Ban le: FRT, MWG, DGW, AST
    # Thep: HPG, HSG
    # Duoc: DMC, DHG, IMP, DGC, CSV
    # VLXD: KSB, VCS
    
    # symbols = getliststocks(typestock = "^VNINDEX")
    
#   
#    
    # symbols = get_csv_data(source = "ssi")
#    symbols = get_csv_data()
#    symbols = get_stocks_highcpm(download = False, source ="cp68")
    
#    symbols =  ['FTS', 'PVI', 'VNE']

#    analysis_trading(symbols, start = "2017-3-1" , end = "2018-4-11", update = False, source = "cp68")

    # rebalancing_porfolio(symbols, bench = '^VNINDEX')
    
    # VNI_result, VNI_data, _  = passive_strategy(start_date = "2019-4-6" , end_date = "2021-8-6", market= "^VNINDEX")
    

###    plot_canslim_trading(ticker, canslim)

#    RSWlist= ['CVN', 'TTB', 'NDN', 'HLD', 'CEO',  'ACB', 'MBS', 'PHC', 'PGS', 'PVB', 
#              'MBB', 'CTG', 'DHC',   'HCM', 'HPG', 'VCI',
#               'BVH', 'TCH', 'PMG',  'VJC', 'GEX', 'MSN',
#              'DGW',    'PNJ',  'PAN', 'GAS', 'DXG', 'IDI', 'VIC', 'ANV',
#              'MSR', 'MCH', 'TVB', 'TBD']

    ticker = ['CTR','VGI','BWE','TDM']
    end_date = "2021-8-17"
    start_date = "2019-4-6"
    ticker = 'DGC'
    # canslim = hung_canslim(ticker, start_date, end_date, realtime = False,  source ="cp68", ndays = 1, typetrade = 'EarlyBreakout') 
    watchlist =['VCB','PVD','KDC','HDG','NT2','FRT','HPG','QNS']
    
    # canslim_strategy(ticker = 'PNJ', start = start_date , end = end_date, update = False,  source ="cp68")
    # agent, history, df_val, test_result, total_rewards, total_losses = auto_trading(ticker='HDG', start="2006-1-19", end= end_date, validation_size = 10, update = False)
    # plot_result(df_val, history, title= "Auto trading " + agent.model_name)
    # print('Final profits: ', test_result)
    
    
    # CHON CO PHIEU BAT DAY
    # analysis_trading(tickers = None, start = start_date, end = end_date, update = False, nbdays = 1, source ="cp68", trade = 'Bottom')
    
    
    # CHON CO PHIEU SIDEWAY
    # analysis_trading(tickers = None, start = start_date , end = end_date, update = False, nbdays = 1, source ="cp68", trade = 'SidewayBreakout')
    
    #CHON CO PHIEU CO EARLY BREAKOUT KHOI NEN GIA
    # analysis_trading(tickers = None, start = start_date , end = end_date, update = True, nbdays = 1, source ="cp68", trade = 'EarlySignal')
    t0 = time.time()
    trade_type = ['EarlySignal','Bottom','SidewayBreakout']
    idx = 0 # EarlySignal
    realtime = True
    datasource = "cp68"
    t1 = 9*60 + 20   
    t2 = 11*60 + 30   
    t3 = 13*60 + 0  
    t4 = 14*60 + 45
    trading = not True
    while trading:  
        # clear_output(wait=True)
        trade_time = datetime.now()
        t = trade_time.hour*60 + trade_time.minute
        if (t >= t1 and t <= t2) or (t >= t3 and t <= t4) and realtime:
            os.system('cls')
            print('TRADING SYSTEM SIGNAL...............',time.asctime(time.localtime(time.time())))
            res = analysis_trading(tickers = None, start = start_date , end = end_date, update = realtime, nbdays = 1, source =datasource, trade = trade_type[idx])
            print("WAIT FOR 4 MINUTES ............................",time.asctime(time.localtime(time.time())))
            print(res.to_string())
            time.sleep(240.0 - ((time.time() - t0) % 240.0))
        elif t < t1 and realtime:
            waittime = t1 - t
            print("WAIT FOR {} MINUTES FROM NOW {} OR COME BACK LATER...".format(waittime,trade_time))
            time.sleep(waittime*60 - ((time.time() - t0) % (waittime*60)))
        elif t2 < t and t < t3 and realtime:
            waittime = t3 - t
            print("WAIT FOR {} MINUTES ............................".format(waittime))
            time.sleep(waittime*60 - ((time.time() - t0) % (waittime*60)))
        elif t > t4 and realtime:
             print('STOCK MARKET CLOSED. SEE YOU NEXT DAY OR USING OFFLINE MODE!')
             print('LAST-MINUTE TRADING SYSTEM SIGNAL...............',time.asctime(time.localtime(time.time())))
             res = analysis_trading(tickers = None, start = start_date , end = end_date, update = realtime, nbdays = 1, source =datasource, trade = trade_type[idx])
             print(res.to_string())     
             break
        else:            
            os.system('cls')
            print('OFF-LINE TRADING SIGNAL ............!')
            print('TRADING SYSTEM SIGNAL...............',time.asctime(time.localtime(time.time())))
            res = analysis_trading(tickers = None, start = start_date , end = end_date, update = realtime, nbdays = 1, source =datasource, trade = trade_type[idx])
            print(res.to_string())            
            break
        
    
    #CHON CO PHIEU CO DIEM MUA BUNG NO KHOI LUONG
    # analysis_trading(tickers = None, start = start_date , end = end_date, update = False, nbdays = 3, source ="cp68", trade = 'LongShortTrend')
    
###    
    
###    
#    my_stock = ['HDC', 'PHR', 'VRE','PVS','PVB','PPC','NTL']
    # analysis_trading(tickers = my_stock, start = start_date , end = end_date, update = False,  source ="cp68", trade = 'Short')
    
#    benchVNI = ["^VNINDEX"]
    # market = analysis_all_market(tickers = benchVNI, start = "2017-1-2" , end = "2018-3-14", update = True,  source ="cp68")
##   
#    
#    
#    my_portfolio()
    # portfolio_management()
    # stock_all, market_all = analysis_stocks(start_date = start_date, end_date = end_date, realtime = False, source = 'cp68')
    
#    hsx_res, hsx_data, hsx_market = passive_strategy(start_date = start_date, end_date = end_date, market = "^VNINDEX")
#    stockVN30 = analysis_VN30(start_date = start_date, end_date = end_date)
#    

    # sys.stdout = old_stdout
    
    # tickers = getliststocks(typestock = "TICKER")
    # textfile = open("WL.txt", "w")
    # for ticker in tickers:
    #     textfile.write(ticker + "\n")
    # textfile.close()
    