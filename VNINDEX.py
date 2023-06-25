# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:35:57 2017

@author: sonng
"""
# import datetime
from datetime import datetime
from strategy import hung_canslim
import pandas as pd
import sys
import time
import os
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import webbrowser

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

    
if __name__ == "__main__":
    #import sys
    #orig_stdout = sys.stdout
    #sys.stdout = open("logging.txt","w")
    
    # export_watchlist()
#   
    symbols = None     
    # symbols = get_csv_data(source = "cp68")    
    end_date = "2023-6-23"
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
    # trading = False
    nlastdays = 1
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
            time.sleep(240.0 - ((time.time() - t0) % 240.0))
        elif t < t1 and realtime:
            waittime = t1 - t
            print("WAIT FOR {} MINUTES FROM NOW {} OR COME BACK LATER...".format(waittime,trade_time))
            res = analysis_trading(tickers = None, start = start_date , end = end_date, update = False, nbdays = nlastdays, source =datasource, trade = trade_type[idx])
            print(res.to_string())
            time.sleep(waittime*60 - ((time.time() - t0) % (waittime*60)))
        elif t2 < t and t < t3 and realtime:
            print('TRADING SYSTEM SIGNAL DURING BREAK...............',time.asctime(time.localtime(time.time())))
            res = analysis_trading(tickers = None, start = start_date , end = end_date, update = realtime, nbdays = nlastdays, source =datasource, trade = trade_type[idx])
            print(res.to_string())   
            waittime = t3 - t
            print("WAIT FOR {} MINUTES ............................".format(waittime))
            time.sleep(waittime*60 - ((time.time() - t0) % (waittime*60)))
        elif t > t4 and realtime:
             print('STOCK MARKET CLOSED. SEE YOU NEXT DAY OR USING OFFLINE MODE!')
             print('LAST-MINUTE TRADING SYSTEM SIGNAL...............',time.asctime(time.localtime(time.time())))
             res = analysis_trading(tickers = None, start = start_date , end = end_date, update = realtime, nbdays = nlastdays, source =datasource, trade = trade_type[idx])
             print(res.to_string())     
             break
        else:            
            os.system('cls')
            print('OFF-LINE TRADING SIGNAL ............!')
            print('TRADING SYSTEM SIGNAL...............',time.asctime(time.localtime(time.time())))
            res = analysis_trading(tickers = None, start = start_date , end = end_date, update = realtime, nbdays = nlastdays, source =datasource, trade = trade_type[idx])
            print(res.to_string())            
            break
    
    
