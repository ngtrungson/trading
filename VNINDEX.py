# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:35:57 2017

@author: sonng
"""
# import datetime
from datetime import date, datetime
from strategy import hung_canslim, get_RSI, get_data, fill_missing_values
import pandas as pd
import numpy as np
import sys
import time
import os
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import webbrowser
import holidays

def get_data_from_cophieu68_openwebsite(tickers):       
#    https://www.cophieu68.vn/export/excelfull.php?id=^hastc
    file_url = 'https://www.cophieu68.vn/export/excelfull.php?id='
    for ticker in tickers:
       webbrowser.open(file_url+ticker)


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
    
    
    symbolsHNX = ['NDN','PVS','VCG','VCS', 'TNG','SHS', 'PLC','NTP','IDC','IDV' ]
    
    symbolsVNI = [ 'ANV',  "BWE",  'CMG', "AGG", "HTN", "TIP",
                   "BID", "BMI", "BMP", "BVH",  "CTD", "CSV", "CTG", 'D2D',
               "DHG",  "DPM",  "DRC", "DXG", 'DGW', 'DBC',
                "FCN",  'FMC', "FPT", "GAS", "GMD",  
                  "HT1",   "LPB", "HSG", "DVP", "TPB","TCL", "TV2",
                "HDG", "HCM", "HPG", 'LHG', 'HDC',
                "IJC",  "KBC",  "KDH", 'CII', 
               "MBB", "MSN", "MWG",  "NLG",  "NVL",
                "PVT","PVD","PHR","PDR", "PNJ",  "PC1",   "PLX",
                "PPC",  "REE", "NKG", 'ILB', 'DHA',
                "SJS","STB", "SSI", "SBT", 
                "VNM", "VHC", "VIC", "VCB", "VSC", "VJC", 
                   'GEX', "VIB", 'HAH', 'SMC','HAH','ITD','OCB','FTS','PTB',
                'TCM',  'AAA',  'VGC', 'DPG', 'BCM', 'KHG', 'SCR','ELC',
                'VPB','VRE',  "HDB",  "ACB", 'BCG' ,'VND', 'SKG',
                'NTL', 'AST', 'VHM',  'TCB', 'ITA',
                'DHC', 'TDM', 'DCM', 'LCG', "VIX",
                   'SZL', 'GVR', 'GIL', 'BFC', 'SZC', 'SHB', 'HHV',
                'IMP', 'MSH', 'POW','TCH','VCI','DIG','KSB','FRT','CRE','PET','DGC']
    
    symbolsUPCOM = ['QNS',  'ACV','VGI','CTR','VTP','VEA','VGT','SNZ','C4G','G36','PXL']    

    
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
    
    
def analysis_trading(tickers, start, end, update = False, nbdays = 15, source = "cp68", trade = 'Long'):    
    if tickers == None:
        tickers = getliststocks(typestock = "TICKER")        
    if tickers == 'VN30':
        tickers = getliststocks(typestock = "VN30")

    result = pd.DataFrame(columns =['Ticker', 'Advise','PCT', 'Close'])
    result = result.set_index('Ticker')
    for ticker in tickers:        
        try:
            res = hung_canslim(ticker, start, end, realtime = update, source = source, ndays = nbdays, typetrade = trade)             
            if len(res) > 1:
                result.loc[res[0]] = [res[1], 100*res[2], res[3]]
        except Exception as e:
            print (e)
            print("Error in reading symbol: ", ticker)
            pass
    return result


def get_csv_data(source = "cp68"):
    symbols = getliststocks(typestock = "ALL")     
    if source == "cp68":
        get_data_from_cophieu68_openwebsite(symbols)
    else:
       get_data_from_cophieu68_openwebsite(symbols)
    return symbols 

def export_watchlist():
    tickers = getliststocks(typestock = "TICKER")
    textfile = open("WL.txt", "w")
    for ticker in tickers:
        textfile.write(ticker + "\n")
    textfile.close()



def passive_strategy(start_date, end_date, market = "^VNINDEX", symbols = None, realtime = False, source = 'cp68'):

    if symbols == None:
        symbols = getliststocks(typestock = market)
        
    if realtime:
        end_date = datetime.today()
        
    dates = pd.date_range(start_date, end_date)  # date range as index
    df_data = get_data(symbols, dates, benchmark = market, realtime = realtime, source = source)  # get data for each symbol
    # Fill missing values
    fill_missing_values(df_data)
    
    df_volume = get_data(symbols, dates, benchmark = market, colname = '<Volume>', realtime = realtime, source = source)  # get data for each symbol
    df_high = get_data(symbols, dates, benchmark = market, colname = '<HighFixed>', realtime = realtime, source = source)
    df_low = get_data(symbols, dates, benchmark = market, colname = '<LowFixed>', realtime = realtime, source = source)
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
    # alpha_beta = analysis_alpha_beta(df_data, symbols, market)
    # df_result['Alpha'] = alpha_beta['Alpha']
    # df_result['Beta'] = alpha_beta['Beta']
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
    
if __name__ == "__main__":
    #import sys
    #orig_stdout = sys.stdout
    #sys.stdout = open("logging.txt","w")
    
    # export_watchlist()
#   
    symbols = None     
    # symbols = get_csv_data(source = "cp68")   
     
    end_date = "2023-7-6"
    start_date = "2021-4-6"
    t0 = time.time()
    trade_type = ['EarlySignal','Bottom','SidewayBreakout']
    idx = 0 # EarlySignal
    realtime = True
    datasource = "ssi"
    t1 = 9*60 + 20   
    t2 = 11*60 + 30   
    t3 = 13*60 + 0  
    t4 = 14*60 + 45
    trading = True if (symbols == None or datasource == "ssi") else False    
       
    nlastdays = 1
    today = date.today().strftime('%Y-%m-%d')
    vn_holidays = holidays.country_holidays('VN')
    while trading:  
        # clear_output(wait=True)
        trade_time = datetime.now()
        t = trade_time.hour*60 + trade_time.minute        
        if (t >= t1 and t <= t2) or (t >= t3 and t <= t4) and realtime:
            os.system('cls')
            print('TRADING SYSTEM SIGNAL...............',time.asctime(time.localtime(time.time())))
            res = analysis_trading(tickers = None, start = start_date , end = end_date, update = realtime, nbdays = nlastdays, source =datasource, trade = trade_type[idx])
            print("WAIT FOR 4 MINUTES ............................",time.asctime(time.localtime(time.time())))
            print(res.to_string())
            if not np.is_busday(today) or today in vn_holidays:
                trading = False
                break
            time.sleep(240.0 - ((time.time() - t0) % 240.0))
        elif t < t1 and realtime:
            waittime = t1 - t
            print("WAIT FOR {} MINUTES FROM NOW {} OR COME BACK LATER...".format(waittime,trade_time))
            res = analysis_trading(tickers = None, start = start_date , end = end_date, update = False, nbdays = nlastdays, source =datasource, trade = trade_type[idx])
            print(res.to_string())
            if not np.is_busday(today) or today in vn_holidays:
                trading = False
                break
            
            time.sleep(waittime*60 - ((time.time() - t0) % (waittime*60)))
        elif t2 < t and t < t3 and realtime:
            print('TRADING SYSTEM SIGNAL DURING BREAK...............',time.asctime(time.localtime(time.time())))
            res = analysis_trading(tickers = None, start = start_date , end = end_date, update = realtime, nbdays = nlastdays, source =datasource, trade = trade_type[idx])
            print(res.to_string())   
            if not np.is_busday(today) or today in vn_holidays:
                trading = False
                break
            waittime = t3 - t
            print("WAIT FOR {} MINUTES ............................".format(waittime))   
            time.sleep(waittime*60 - ((time.time() - t0) % (waittime*60)))
        elif t > t4 and realtime:
             print('STOCK MARKET CLOSED. SEE YOU NEXT DAY OR USING OFFLINE MODE!')
             print('LAST-MINUTE TRADING SYSTEM SIGNAL...............',time.asctime(time.localtime(time.time())))
             res = analysis_trading(tickers = None, start = start_date , end = end_date, update = realtime, nbdays = nlastdays, source =datasource, trade = trade_type[idx])
             print(res.to_string())     
             trading = False
             break
        else:            
            os.system('cls')
            print('OFF-LINE TRADING SIGNAL ............!')
            print('TRADING SYSTEM SIGNAL...............',time.asctime(time.localtime(time.time())))
            res = analysis_trading(tickers = None, start = start_date , end = end_date, update = realtime, nbdays = nlastdays, source =datasource, trade = trade_type[idx])
            print(res.to_string())  
            trading = False
            break
        
    stock_all, market_all = analysis_stocks(start_date = start_date, end_date = end_date, realtime = False, source = 'cp68')
    
    
    
