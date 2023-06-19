# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:44:36 2018

@author: sonng
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# import urllib3
import bs4 as bs
import pickle
import requests
import webbrowser
import datetime as dt
import scipy.optimize as spo
# from statsmodels import regression
# import statsmodels.api as sm

from alpha_vantage.timeseries import TimeSeries
from pandas_datareader import data as pdr
import talib
import yfinance as yf
import datetime
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
def get_symbols_rts():
    tickers = ['MMM','AA', 'BABA','AMZN', 'AAPL', 'T', 'AXP','ALB', 'BB', 'BAC',
                   'BA','CAT', 'CSCO', 'C', 'KO', 'CL', 'DIS', 'DB', 'DBX', 'EBAY',
                   'FB','FSLR','GE','GM', 'GOOG', 'GS', 'GPRO', 'HOG', 'IBM', 'INTC',
                   'JPM', 'LN','LMT', 'MTCH', 'MA', 'MCD', 'MSFT','NFLX', 'NKE',
                   'NOK', 'NVDA', 'PYPL', 'PEP', 'PFE', 'RACE', 'SNE', 'SBUX',
                   'SNAP', 'SPOT', 'TSLA', 'TWTR', 'UBS', 'V', 'WMT', 'YNDX','AUY', 'ZTO']
    return tickers

def get_symbols_us():
    tickers = ['MMM', 'ABT', 'ABBV', 'ACN', 'ATVI', 'AYI', 'ADBE', 'AMD', 'AAP', 'AES', 
               'AET', 'AMG', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALXN', 
               'ALGN', 'ALLE', 'AGN', 'ADS', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 
               'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 
               'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'ANDV', 'ANSS', 'ANTM', 
               'AON', 'AOS', 'APA', 'AIV', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ARNC', 
               'AJG', 'AIZ', 'T', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'BHGE', 'BLL', 
               'BAC', 'BK', 'BAX', 'BBT', 'BDX', 'BRK.B', 'BBY', 'BIIB', 'BLK', 
               'HRB', 'BA', 'BWA', 'BXP', 'BSX', 'BHF', 'BMY', 'AVGO', 'BF.B', 
               'CHRW', 'CA', 'COG', 'CDNS', 'CPB', 'COF', 'CAH', 'CBOE', 'KMX', 
               'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNC', 'CNP', 'CTL', 'CERN', 
               'CF', 'SCHW', 'CHTR', 'CHK', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 
               'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 
               'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'CXO', 'COP', 
               'ED', 'STZ', 'COO', 'GLW', 'COST', 'COTY', 'CCI', 'CSRA', 'CSX', 
               'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 
               'DVN', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D', 
               'DOV', 'DWDP', 'DPS', 'DTE', 'DRE', 'DUK', 'DXC', 'ETFC', 'EMN', 
               'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ETR', 'EVHC', 
               'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'RE', 
               'EXC', 'EXPE', 'EXPD', 'ESRX', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST',
               'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FISV', 'FLIR', 'FLS', 'FLR', 
               'FMC', 'FL', 'F', 'FTV', 'FBHS', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT',
               'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GS', 'GT', 
               'GWW', 'HAL', 'HBI', 'HOG', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HP', 
               'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 
               'HPQ', 'HUM', 'HBAN', 'HII', 'IDXX', 'INFO', 'ITW', 'ILMN', 'IR', 'INTC',
               'ICE', 'IBM', 'INCY', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IQV', 
               'IRM', 'JEC', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 
               'KEY', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KHC', 'KR', 'LB', 'LLL', 
               'LH', 'LRCX', 'LEG', 'LEN', 'LUK', 'LLY', 'LNC', 'LKQ', 'LMT', 'L', 
               'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 
               'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 
               'MGM', 'KORS', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'TAP', 'MDLZ', 'MON',
               'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP', 
               'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI',
               'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'ORLY',
               'OXY', 'OMC', 'OKE', 'ORCL', 'PCAR', 'PKG', 'PH', 'PDCO', 'PAYX', 'PYPL', 
               'PNR', 'PBCT', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 
               'PXD', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCLN', 'PFG', 'PG', 'PGR', 
               'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX',
               'RRC', 'RJF', 'RTN', 'O', 'RHT', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 
               'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RCL', 'CRM', 'SBAC', 'SCG', 'SLB',
               'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIG', 'SPG', 'SWKS', 'SLG', 'SNA', 
               'SO', 'LUV', 'SPGI', 'SWK', 'SBUX', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 
               'SYF', 'SNPS', 'SYY', 'TROW', 'TPR', 'TGT', 'TEL', 'FTI', 'TXN', 'TXT', 
               'TMO', 'TIF', 'TWX', 'TJX', 'TMK', 'TSS', 'TSCO', 'TDG', 'TRV', 'TRIP', 
               'FOXA', 'FOX', 'TSN', 'UDR', 'ULTA', 'USB', 'UAA', 'UA', 'UNP', 'UAL', 
               'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'VFC', 'VLO', 'VAR', 'VTR', 
               'VRSN', 'VRSK', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA', 
               'DIS', 'WM', 'WAT', 'WEC', 'WFC', 'HCN', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 
               'WMB', 'WLTW', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YUM', 
               'ZBH', 'ZION', 'ZTS']
    return tickers


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers

def get_data_from_web(tickers, start, end, source = 'yahoo', redownload = False):
    
    if not os.path.exists(source):
        os.makedirs(source)
        
    yf.pdr_override() # <== that's all it takes :-)
    apiKey = '9ODDY4H8J5P847TA'
    filepath = source + '/{}.csv'
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!        
       if (redownload | (not os.path.exists(filepath.format(ticker)))):
           try:
#            df = web.DataReader(ticker, source, start, end)
                
                if (source == 'yahoo'):                    
                    df = pdr.get_data_yahoo(ticker, start=start, end=end,  as_panel = False)                   
                    df.to_csv(filepath.format(ticker))
                if (source == 'alpha'):                    
                    ts = TimeSeries(key=apiKey, output_format='pandas')
                    df, _ = ts.get_daily(symbol=ticker, outputsize='full')                    
                    df.to_csv(filepath.format(ticker))
                    
           except Exception as e:
                print(str(e))
       else:
            print('Already have {}'.format(ticker))


# Create Pandas data frame for backtrader
def get_data_ticker_us(symbol, start, end): 
    # columns order for backtrader type
    columnsOrder=["Open","High","Low","Close", "Volume", "OpenInterest"]
    # if symbol is list type
    if  isinstance(symbol, list):
        symbol = ''.join(symbol)
    # obtain data from csv file    
    dataframe = pd.read_csv(symbol_to_path(symbol,base_dir ='data'), parse_dates=True, index_col=0) 
    # change the index by new index
    dataframe = dataframe.reindex(columns = columnsOrder)   
    # change date index to increasing order
    dataframe = dataframe.sort_index()
    # might be wrong and contain NA values ????    
    df = dataframe.loc[start:end]
    return df


    
def get_data_us(symbols, dates, benchmark = 'SPY', colname = 'Close'):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df_final = pd.DataFrame(index=dates)
    if (benchmark not in symbols) and isinstance(benchmark, str):  # add SPY for reference, if absent
        symbols = [benchmark] + symbols
        
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol, base_dir ="yahoo"), index_col='Date',
                parse_dates=True, usecols=['Date', colname], na_values=['nan'])
        df_temp = df_temp.rename(columns={colname: symbol})
        df_final = df_final.join(df_temp)
        if symbol == benchmark:  # drop dates SPY did not trade
            df_final = df_final.dropna(subset=[benchmark])
            
#    fill_missing_values(df_final)
    
    return df_final



# Create Pandas data frame for backtrader
def get_data_trading(symbol, start, end): 
    # columns order for backtrader type
    columnsOrder=["Open","High","Low","Close", "Volume", "OpenInterest"]
    # if symbol is list type
    if  isinstance(symbol, list):
        symbol = ''.join(symbol)
    # obtain data from csv file    
    file_path = symbol_to_path(symbol)
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
    # might be wrong and contain NA values ????    
    df = df.loc[start:end]   
    
    return df

################################# Get data from website ################################################

def save_and_analyse_vnindex_tickers():
    resp = requests.get('https://www.cophieu68.vn/export.php')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'width': '100%','cellpadding' : '4', 'border' : '0'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    tickers.remove('ALL DATA (HOSE & HNX)')
#    tickers.remove('000001.SS')
#    tickers.remove('^XAYDUNG')
#    tickers.remove('VNINDEX')
#    tickers.remove('HNX')
#    tickers.remove('^VNINDEX')
#    tickers.remove('^VNINDEX2')
#    tickers.remove('AGD')
#    tickers.remove('AGC')
#    tickers.remove('ALP')
    with open("vnindextickers.pickle","wb") as f:
        pickle.dump(tickers,f)
    
        
    data = fundemental_analysis(tickers)
    data.to_csv('fundemental_stocks_all_0706.csv')
    
    
    return tickers

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def fundemental_analysis(tickers):
    df = pd.DataFrame()
    for ticker in tickers:        
        print(ticker)
        try:
            df_temp = get_info_stock(ticker)
            if (len(df) == 0):
                df = pd.DataFrame(df_temp, index = [0])   
            else:       
                df.loc[len(df)] = df_temp
        except Exception as e:
            print(" Error in symbol : ", ticker) 
            print(e)
            pass
    df = df.set_index('Ticker')
    return df
    

def get_info_stock_bsc(ticker):
    url = 'https://www.bsc.com.vn/Companies/Overview/{}'.format(ticker)   
    resp = requests.get(url, verify =False)
    # print(resp.text)
    soup = bs.BeautifulSoup(resp.text, 'lxml')  
   
        # print(line)
    # print(soup)
    tables = soup.find_all('table')
    data = tableDataText(tables[0])
    line2 = data[2][1].replace('(','').replace(')','').replace('/',' ').replace(',','.').split()
    line3 = data[3][1].replace(',','.').replace('-','').split()
    line4 = data[4][1].replace('.','')
    volume = float(line4)
    low = float(line3[0])
    high = float(line3[1])
    close = float(line2[0])
    close1D = close - float(line2[1]) 

    
    df = {'Ticker':ticker, ## Getting Only The Stock Name, not 'json'
                        'Close': close,
                        'Close_1D' : close1D,
                          'Open' : close1D,
                          'Low' : low,
                          'High': high, 
                          'Volume': volume,                       
                          }
  
    return df  

def get_info_stock_ssi(ticker):
    url = 'https://ivt.ssi.com.vn/CorporateSnapshot.aspx?ticket={}'.format(ticker)   
    resp = requests.get(url, verify =False)
    # print(resp.text)
    soup = bs.BeautifulSoup(resp.text, 'lxml')     
    
        # print(line)
    # print(soup)
    tables = soup.find_all('table')
    data = tableDataText(tables[4])
    # print(data)
    line2 = data[1][0].replace(')','').replace('(',' ').replace(',','.').split()
    # line3 = data[3].replace(',','.').split()
    # line1 = data[1][2].replace(',','.')
    volume = float(data[3][0])*1000
    low = float(data[3][2].replace(',','.'))
    high = float(data[1][2].replace(',','.'))
    open0D = float(data[1][1].replace(',','.'))
    close = float(data[3][1].replace(',','.'))
    close1D = close - float(line2[0]) 

    
    df = {'Ticker':ticker, ## Getting Only The Stock Name, not 'json'
                        'Close': close,
                        'Close_1D' : close1D,
                          'Open' : open0D,
                          'Low' : low,
                          'High': high, 
                          'Volume': volume,                       
                          }
  
    return df  


def get_info_stock_vssc(ticker):
    url = 'http://ra.vcsc.com.vn/?lang=vi-VN&ticker={}'.format(ticker)  
    resp = requests.get(url)
    # print(resp.text)
    soup = bs.BeautifulSoup(resp.text, 'lxml')     
   
    value_number = []
    data = soup.find('div', {'class':'home-block2'}).stripped_strings
    for line in data:
        line = line.replace('.','').replace(',','')
        if isfloat(line): 
            value_number.append(float(line)) 
    
    close1D = value_number[0]/1000.0 
    open0D = value_number[1]/1000.0 
    high = value_number[2]/1000.0 
    low = value_number[3]/1000.0     
    close = value_number[4]/1000.0 
    volume = value_number[5]
    
    df = {'Ticker':ticker, ## Getting Only The Stock Name, not 'json'
                        'Close': close,
                        'Close_1D' : close1D,
                          'Open' : open0D,
                          'Low' : low,
                          'High': high, 
                          'Volume': volume,                       
                          }
  
    return df  


def get_info_stock(ticker):
    url = 'https://www.cophieu68.vn/snapshot.php?id={}'.format(ticker)   
    
    # http = urllib3.PoolManager()
    # r = http.request('GET', url)
    # soup = bs.BeautifulSoup(r.data, 'lxml')
        
    resp = requests.get(url, verify =False)
    # print(resp.text)
    soup = bs.BeautifulSoup(resp.text, 'lxml') 

    value_number = []
    stockexchange = 'HSX'
    
    
    for line in soup.find('div', {'class':'listHeader'}).stripped_strings:
        line = line.replace(',','').replace('%','').replace(':','').replace(ticker,'').replace(' ','')
        line = line.replace('triá»\x87u','').replace('ngÃ\xa0n','').replace('(','').replace(')','')       
        if isfloat(line): 
            value_number.append(float(line)) 
         
        if ((line == 'HSX')| (line == 'HNX') | (line == 'UPCOM') | (line == 'HOSE')):
            stockexchange = line
        else:
            stockexchange = 'BM'
#    print(stockexchange)
    for line in soup.find('div', {'id':'snapshot_trading'}).stripped_strings:
        line = line.replace(',','').replace('%','')
        line = line.replace('triá»\x87u','').replace('ngÃ\xa0n','')        
        if isfloat(line): 
            value_number.append(float(line)) 
#        print(line) 
#    print(value_number)   
    
    df = {'Ticker':ticker, ## Getting Only The Stock Name, not 'json'
                        'Close': value_number[0],
                        'Close_1D' : value_number[1],
                         'Open' : value_number[2],
                         'Low' : value_number[3],
                         'High': value_number[4], 
                         'Volume': value_number[5],
                         'MeanVol_13W' : value_number[6], 
                         'MeanVol_10D' : value_number[7],
                         'High52W' : value_number[8], 
                         'Low52W' : value_number[9], 
                         'EPS' : value_number[10]*1E3, 
                         'PE' : value_number[11],
                         'Market capital' : value_number[12]*1E9, 
                         'Float' : value_number[13]*1E6, 
                         'BookValue' : value_number[14], 
                         'ROE' : value_number[15], 
                         'Beta' : value_number[16], 
                         'EPS_52W' : value_number[17],
                         'CPM': value_number[8]/value_number[9],
                         'FVQ': value_number[13]/value_number[6]*1E6,
                         'Exchange': stockexchange}
  
    return df 

def tableDataText(table):       
    rows = []
    trs = table.find_all('tr')
    headerow = [td.get_text(strip=True) for td in trs[0].find_all('th')] # header row
    if headerow: # if there is a header row include first
        rows.append(headerow)
        trs = trs[1:]
    for tr in trs: # for every table row
        rows.append([td.get_text(strip=True) for td in tr.find_all('td')]) # data row
    return rows    
    
def get_info_stock_cp68_mobile(ticker):
    
    url = 'https://m.cophieu68.vn/snapshot.php?id={}&s_search=Go'.format(ticker)
    resp = requests.get(url, verify =False)
    soup = bs.BeautifulSoup(resp.text, 'lxml') 
    # df = pd.DataFrame(columns = ['Ticker',
    #                          'Close',
    #                          'Close_1D',
    #                          'Open',
    #                          'Low',
    #                          'High', 
    #                          'Volume'])
   
        # print(line)
   
    # for line in soup.find(id="stockname_change").stripped_strings:
    #     line = line.replace(' ','').replace('\xa0\xa0\r','').replace('\n',' ').split()
    #     if isfloat(line[0]): 
    #         value_number.append(float(line[0])) 
    
    # hard coding for Close, Close_1D, Open, High, Low, Volume   
    value_number = []
    for line in soup.find(id="stockname_close").stripped_strings:
        line =  line.replace(',','')
        if isfloat(line): 
            value_number.append(float(line))
        
    tables = soup.find_all('table')
    
    data = tableDataText(tables[2])
    
    value_number.append(float(data[0][1].replace(',',''))) # close
    value_number.append(float(data[1][1].replace(',',''))) # close-1D
    value_number.append(float(data[2][1].replace(',',''))) # high
    value_number.append(float(data[3][1].replace(',',''))) # low
    value_number.append(float(data[4][1].replace(',',''))) # volume
    df = {'Ticker':ticker, ## Getting Only The Stock Name, not 'json'
            'Close': value_number[0],
            'Close_1D' : value_number[1],
            'Open' : value_number[2],
            'Low' : value_number[4],
            'High': value_number[3], 
            'Volume': value_number[5],  }
    # df = pd.DataFrame(data, index=[0])
    # df = df.append({'Ticker':ticker, ## Getting Only The Stock Name, not 'json'
    #                     'Close': value_number[0],
    #                     'Close_1D' : value_number[1],
    #                      'Open' : value_number[2],
    #                      'Low' : value_number[4],
    #                      'High': value_number[3], 
    #                      'Volume': value_number[5],                       
    #                      }, ignore_index = True)
  
    return df  
 
def get_data_from_cophieu68_openwebsite(tickers):       
#    https://www.cophieu68.vn/export/excelfull.php?id=^hastc
    file_url = 'https://www.cophieu68.vn/export/excelfull.php?id='
    for ticker in tickers:
       webbrowser.open(file_url+ticker)
       

def get_data_from_SSI_website(tickers):       
    file_url = 'http://ivt.ssi.com.vn/Handlers/DownloadHandler.ashx?Download=1&Ticker='
    for ticker in tickers:
       webbrowser.open(file_url+ticker)
   
    
def symbol_to_path(symbol, base_dir="cp68"):
    """Return CSV file path given ticker symbol."""
    if base_dir == "cp68":
        fileformat = "excel_{}.csv"        
    if base_dir == "ssi":       
        fileformat = "Historical_Price_{}.csv"
    if (base_dir == "yahoo") | (base_dir == "alpha"):      
        fileformat = "{}.csv"
    if (base_dir == "amibroker"):      
        fileformat = "{}.csv"
        
    return os.path.join(base_dir, fileformat.format(str(symbol)))


#def get_data(symbols, dates, addVNINDEX=True, colname = '<CloseFixed>'):
#    """Read stock data (adjusted close) for given symbols from CSV files."""
#    df_final = pd.DataFrame(index=dates)
#    if addVNINDEX and 'VNINDEX' not in symbols:  # add SPY for reference, if absent
#        symbols = ['VNINDEX'] + symbols
#        
#    for symbol in symbols:
#        file_path = symbol_to_path(symbol)
#        df_temp = pd.read_csv(file_path, parse_dates=True, index_col="<DTYYYYMMDD>",
#            usecols=["<DTYYYYMMDD>", colname], na_values=["nan"])
#        df_temp = df_temp.rename(columns={"<DTYYYYMMDD>": "Date", colname: symbol})
#        df_final = df_final.join(df_temp)
#        if symbol == "VNINDEX":  # drop dates SPY did not trade
#            df_final = df_final.dropna(subset=["VNINDEX"])
#
#    return df_final

def get_data(symbols, dates, benchmark = '^VNINDEX', colname = '<CloseFixed>', realtime = False, source ='cp68'):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df_final = pd.DataFrame(index=dates)
    if (benchmark not in symbols) and isinstance(benchmark, str):  # add SPY for reference, if absent
        symbols = [benchmark] + symbols
        
    for symbol in symbols:
        file_path = symbol_to_path(symbol, base_dir = source)
        if source == 'cp68':
            df_temp = pd.read_csv(file_path, parse_dates=True, index_col="<DTYYYYMMDD>",
            usecols=["<DTYYYYMMDD>", colname], na_values=["nan"])
            df_temp = df_temp.rename(columns={"<DTYYYYMMDD>": "Date", colname: symbol})  
        if source == 'yahoo':
            df_temp = pd.read_csv(file_path, index_col='Date',
                parse_dates=True, usecols=['Date', colname], na_values=['nan'])
            df_temp = df_temp.rename(columns={colname: symbol})
          
        df_final = df_final.join(df_temp)
        if symbol == benchmark:  # drop dates SPY did not trade
            df_final = df_final.dropna(subset=[benchmark])
            
#    fill_missing_values(df_final)
    
        
    if (realtime & ((source == 'cp68') | (source == 'ssi'))):
        today_data = []
        for symbol in symbols:
            actual_price = get_info_stock_cp68_mobile(symbol)
            # actual_price = get_info_stock_bsc(ticker)
            today = datetime.datetime.today()
            next_date = today
            if colname == '<Volume>':
                today_data.append(actual_price['Volume'])
                # df_temp.loc[next_date] = ({symbol : actual_price['Volume'].iloc[-1]})
            elif colname == '<High>':
                today_data.append(actual_price['High'])
                # df_temp.loc[next_date] = ({symbol : actual_price['High'].iloc[-1]})
            elif colname == '<Low>':
                today_data.append(actual_price['Low'])
                # df_temp.loc[next_date] = ({symbol : actual_price['Low'].iloc[-1]})
            else:
                today_data.append(actual_price['Close'])
                # df_temp.loc[next_date] = ({symbol : actual_price['Close'].iloc[-1]})
            # print(df_temp.loc[next_date])  
        df_final.loc[next_date] = today_data
    
    return df_final



def get_RSI(symbols, df):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df_final = pd.DataFrame(index=df.index)   
    for symbol in symbols:
        df_final[symbol] = talib.RSI(df[symbol].values, timeperiod = 14) 
    fill_missing_values(df_final)    
    return df_final

# def linreg(x,y):
#     # We add a constant so that we can also fit an intercept (alpha) to the model
#     # This just adds a column of 1s to our data
#     x = sm.add_constant(x)
#     model = regression.linear_model.OLS(y,x).fit()
#     # Remove the constant now that we're done
#     x = x[:, 1]
#     return model.params[0], model.params[1]

#alpha, beta = linreg(X,Y)

# def analysis_alpha_beta(df_data, symbols, market = '^VNINDEX'):
#     df_result = pd.DataFrame(columns= ['Ticker', 'Alpha', 'Beta'])
    
#     for ticker in symbols:       
#         alpha, beta = compute_alpha_beta(df = df_data, symbol = ticker, index = market)    
#         df_result =  df_result.append({'Ticker':ticker, 'Alpha': alpha, 'Beta': beta},ignore_index = True)
   
#     df_result = df_result.set_index('Ticker')
#     return df_result
        
    

# def compute_alpha_beta(df, symbol = 'BVH', index = '^VNINDEX'):
# #    covariance = np.cov(df[symbol] , df[index])[0][1] 
# #    variance = np.var(df[index])
# #    beta = covariance / variance 
#     r_a = df[symbol].pct_change()[1:]   
#     r_b = df[index].pct_change()[1:]
#     X = r_b.values # Get just the values, ignore the timestamps
#     Y = r_a.values
#     alpha, beta = linreg(X,Y)
    
#     return alpha, beta

def compute_daily_returns(df):
    """Compute and return the daily return values."""    
    # Note: Returned DataFrame must have the same number of rows
    daily_returns = df.pct_change()
#    daily_returns[1:] = (df[1:]/df[:-1].values)-1
    daily_returns.iloc[0,:]=0
    return daily_returns

def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    ##########################################################
    df_data.fillna(method ="ffill", inplace = True)
    df_data.fillna(method ="backfill", inplace = True)
    pass  
    
    
def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    plt.show()
    
    
def get_portfolio_volume_mean(prices):
    #  Compute daily portfolio value given stock prices, allocations and starting value
    vol_mean = prices.mean(axis = 1)      
    return vol_mean

def get_portfolio_value(prices, allocs, start_val = 1.0):
    #  Compute daily portfolio value given stock prices, allocations and starting value
    normed = prices/prices.iloc[0,:].values    
    alloced = normed * allocs
    pos_vals = alloced*start_val
    port_val = pos_vals.sum(axis=1)    
    return port_val

def get_portfolio_stats(port_val, daily_rf = 0.0, samples_per_year = 252):
    # Calculate statistics on daily portfolio value, given daily risk-free rate and data sampling frequency.
    daily_rets = port_val.pct_change()
    daily_rets.iloc[0] = 0 
    cr = (port_val[-1]/port_val[0]) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr =   np.sqrt(samples_per_year) * np.mean(daily_rets - daily_rf) / np.std(daily_rets)
    return cr, adr, sddr, sr
################################################ PLOTING ###############################################
def plot_normalized_data(df, title="Normalized stock prices", xlabel="Date", ylabel="Normalized price"):
    import matplotlib.pyplot as plt
    """Plot stock prices with a custom title and meaningful axis labels."""
    df = df/df.iloc[0,:].values    
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    plt.show()

def plot_bollinger_bands(maindf, symbol, window = 20, nb_std = 2):
     # Compute Bollinger Bands
    rm = maindf[symbol].rolling(window=window,center=False).mean()   
    rstd = maindf[symbol].rolling(window=window,center=False).std()
    upper_band = rm + nb_std*rstd
    lower_band = rm - nb_std*rstd  
    ax = maindf[symbol].plot(title="Bollinger Bands", label = symbol)
    rm.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='Upper band', ax=ax)
    lower_band.plot(label='Lower band', ax=ax)
    ax.legend([symbol, "Rolling mean", "Upper band", "Lower band"]);
    ax.grid(True)    


################################################ PLOTING ###############################################

# Find the optimal allocations for a given stock portfolio
def find_optimal_allocations(prices):
    guess = 1.0/prices.shape[1]
    function_guess = [guess] * prices.shape[1]
    bnds = [[0,1] for _ in prices.columns]
    cons = ({ 'type': 'eq', 'fun': lambda function_guess: 1.0 - np.sum(function_guess) })
    result = spo.minimize(error_optimal_allocations, function_guess, args = (prices,), method='SLSQP', bounds = bnds, constraints = cons, options={'disp':True})
    allocs = result.x
    return allocs

"""A helper function for the above function to minimize over"""
def error_optimal_allocations(allocs, prices):
    port_val = get_portfolio_value(prices, allocs, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)
    error = sharpe_ratio * -1
    return error
# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], benchmark = '^VNINDEX', country = 'VN', gen_plot=False):
    
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    if country=='VN':
        prices_all = get_data(syms, dates, benchmark = benchmark)  # automatically adds SPY
    else:
        prices_all = get_data_us(syms, dates, benchmark = benchmark)
    
    fill_missing_values(prices_all)
    
    prices = prices_all[syms]  # only portfolio symbols
    prices_benchmark = prices_all[benchmark]  # only VNINDEX, for comparison later
    
    
    
    
#    return prices, prices_benchmark

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    allocs = np.asarray([0.2, 0.2, 0.3, 0.3, 0.0]) # add code here to find the allocations
    cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats
    
    allocs = find_optimal_allocations(prices) 
    port_val = get_portfolio_value(prices, allocs)
    cr, adr, sddr, sr = get_portfolio_stats(port_val)   

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_benchmark], keys=['Portfolio', benchmark], axis=1)
        plot_normalized_data(df_temp, title= " Daily porfolio value and benchmark ", xlabel="Date", ylabel= " Normalized price ")
        pass

    
    return allocs, cr, adr, sddr, sr



def compute_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], allocs = [0.2, 0.2, 0.3, 0.3, 0.0], benchmark = '^VNINDEX', gen_plot=False):
    
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates, benchmark = benchmark)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_benchmark = prices_all[benchmark]  # only VNINDEX, for comparison later
        
    port_val = get_portfolio_value(prices, allocs)
    cr, adr, sddr, sr = get_portfolio_stats(port_val)   

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_benchmark], keys=['Portfolio', benchmark], axis=1)
        plot_normalized_data(df_temp, title= " Daily porfolio value and benchmark ", xlabel="Date", ylabel= " Normalized price ")
        pass

    
    return cr, adr, sddr, sr



def optimize_porfolio_markowitz(maindf, symbols):
        
    data = maindf[symbols]
    #convert daily stock prices into daily returns
    returns = data.pct_change()
     
    #calculate mean daily return and covariance of daily returns
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
     
    #set number of runs of random portfolio weights
    num_portfolios = 25000
     
    #set up array to hold results
    #We have increased the size of the array to hold the weight values for each stock
    results = np.zeros((4+len(symbols)-1,num_portfolios))
     
    for i in range(num_portfolios):
        #select random weights for portfolio holdings
        weights = np.array(np.random.random(len(symbols)))
        #rebalance weights to sum to 1
        weights /= np.sum(weights)
     
        #calculate portfolio return and volatility
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)
     
        #store results in results array
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std_dev
        #store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        results[2,i] = results[0,i] / results[1,i]
        #iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results[j+3,i] = weights[j]
     
    #convert results array to Pandas DataFrame
#    print(" results Shape 0 : ", results.shape[0], " results Shape 1 : ", results.shape[1])
    results_frame = pd.DataFrame(results.T,columns=['ret','stdev','sharpe'] + symbols)
#    print(" results_frame Shape 0 : ", results_frame.shape[0], " results_frame Shape 1 : ", results_frame.shape[1])
    #locate position of portfolio with highest Sharpe Ratio
    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
    #locate positon of portfolio with minimum standard deviation
    min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
    
 
    #create scatter plot coloured by Sharpe Ratio
#    fig = plt.figure()
#    plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
#    plt.xlabel('Volatility')
#    plt.ylabel('Returns')
#    plt.colorbar()
#    #plot red star to highlight position of portfolio with highest Sharpe Ratio
#    plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=1000)
#    #plot green star to highlight position of minimum variance portfolio
#    plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=1000)
    return results_frame, max_sharpe_port, min_vol_port 

if __name__ == "__main__":
     
     neg_beta = ['AMD', 'ATG', 'DAH', 'DST', 'FRT', 'HMC', 'ITC', 'MCG', 'MST', 'NSH',
       'PGD', 'PLC', 'SPP', 'TCH', 'TV2', 'VTO'] 
    
#     symbolsHNX = [ 'ALV',  'TNG', 'BVS',  'NVB',   "VE9", 
#                  'ACB', 'BCC', 'CVN', 'CEO', 'DBC',  'DST', 'HUT', 'SD9', 'HLD', 'HHG', 'NSH', 'DS3',
#                  'LAS',  'MBS', 'NDN', 'PGS', 'PVC', 'PVI',   'PHC', 'PVE', 'PVG', 'PVB',
#                  'PVS', 'SHB', 'SHS', 'TTB','VC3', 'VCG','VCS', 'VGC','VMC','VIX', 'TVC', 'SPP',
#                 'VKC', 'VPI', 'NBC', 'VGS']
#    
#     symbolsVNI = ['APG', 'APC', 'ANV', "ASM", "BSI", "BWE", 'CEE',
#                  'BCG', "BFC", "BID", "BMI", "BMP", "BVH",  'CTS', 'CTI', "CII", "CTD", "CAV", "CMG", "CSM", "CSV", "CTG", 'CHP', 'C47', 
#               "DCM","DHG", "DIG",  "DPM","DPR", "DRH",  "DQC", "DRC", "DXG", 'DGW', 'DHA', 'DHC', 'DAH',
#               "ELC", "EVE", 'EVG', "FCN","FIT","FLC", 'FMC', 'FTS', "FPT", "GAS", "GMD", "GTN", 
#                'HAX', "HAG", "HHS", "HNG",  "HT1",  'HAR', 'HII', 'HCD',
#               "HSG", "HDG", "HCM", "HPG", "HBC", 'LDG', 'LCG', 'LGL', 'LHG', 'HDC',
#               'IDI', "IJC",  'ITD', "KBC", "KSB",  "KDH", "KDC", 
#               "MBB", "MSN", "MWG", "NKG", "NLG", "NT2", "NVL", "NBB", 'NAF',
#                "PVT","PVD","PHR","PGI","PDR","PTB", "PNJ",  "PC1",   "PLX", "PXS",
#                "PPC", "PAC", 'QBS', "QCG", "REE",  "SJF",
#                'SHI',"SAM","SJD","SJS","STB","STG","SKG",  "SSI", "SBT", "SAB", 
#                "VSH","VNM", "VHC", "VIC", "VCB", "VSC", "VJC", "VNS" , 'TVS', 'VDS', 'TNI','TLH',
#                'LSS',  'PME', 'PAN','TCH', 'TDH',  'GEX','VCI', 'VIS',
#                'TDC','TCM', 'VNE', 'SHN', 'AAA','SCR',  'TDG', 'VRC',  'SRC', 'TLD', 'PMG',
#                'EIB','VPB','VRE','ROS',"VND", "HDB",  "C32","CVT",'VPH','VNG','VIP',
#                'NTL','PET','VPD','VTO','SHA','DCL', 'GIL', 'TEG', 'AST','DAG', 'HAH']
#    
#     symbolsUPCOM = ['TBD', 'LPB', 'QNS',   'ART',  'ACV',  "SWC", "NTC","DVN", 'HVN', 'HPI','IDC',  'MSR', 
#                    'VGT','TVN','TVB','TIS','VIB','DRI', 'POW', 'BSR','MCH']
     
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
     
     # 'SMC','HAH','ITD','OCB','FTS','PTB'
     
     symbolsUPCOM = ['QNS',  'ACV','VGI','CTR','VTP','VEA','VGT','SNZ','C4G','G36','PXL'] 
    
     
     symbols = symbolsVNI + symbolsHNX +  symbolsUPCOM
     
     symbols = pd.unique(symbols).tolist()
     
#     df_temp = get_info_stock('VNM')
# 
     # data = fundemental_analysis(symbols)
#    
#    data.to_csv('fundemental_stocksVN.csv')
    
     tickers = save_and_analyse_vnindex_tickers()
    
     data = pd.read_csv('fundemental_stocks_all_0706.csv', parse_dates=True, index_col=0)
      # data['Diff_Price'] = data['Close'] - data['EPS']*data['PE']/1000
      # data['EPS_Price'] = data['EPS']/data['Close']/1000
     data['Value'] = data['Close']* data['MeanVol_10D']
     df = data.query("MeanVol_13W > 50000")
     df = df.query("MeanVol_10D> 50000")
     df = df.query("Value > 3000000")
###     df = df.query("FVQ > 0")
###     df = df.query("CPM > 1.4")
     df = df.query("EPS >= 1500")
###     df = df.query("EPS_52W >= 0")
     df = df.query("ROE >= 10")
#     df = df.query("Close > 4")
#     df = df.query("Beta < 0")
#     df = df.query("Beta > 0")
#     df = df.query("Diff_Price < 0")
#     df.to_csv('investment_stock3.csv')
#     print(df.index)
     
     listA = symbols
     listB = df.index.tolist()
     common = list(set(listA) & set(listB))
     listC = list(set(listB).difference(set(listA)))
     df2 = data.loc[symbols]
##     
#     end_date = "2018-5-29"
#     start_date = "2017-1-2"
     
     
     # ticker = 'FPT'
     # url = 'https://www.bsc.com.vn/Companies/Overview/{}'.format(ticker) 
     # url2 = 'http://ra.vcsc.com.vn/?lang=vi-VN&ticker={}'.format(ticker)
     # url3 = 'http://data.vdsc.com.vn/vi/Stock/{}'.format(ticker)
     # df1 = get_info_stock_cp68_mobile("FPT")
     # print("done 1")
     # df2 = get_info_stock_ssi("FPT")
     # print("done 2")
     # url4 ='https://ivt.ssi.com.vn/CorporateSnapshot.aspx?ticket=vcs'
     
     # # dfs = pd.read_html(url4)
     # print("done 3")
#     data = yf.download("SPY", start="2017-01-2", end="2018-05-29")
#     get_data_from_web(tickers = ['MSFT'], start = start_date, end = end_date, source ='yahoo')
#     yf.pdr_override()
#     ticker = 'ZTO'
#     df = pdr.get_data_yahoo(ticker, start = start_date, end = end_date) 
     
     
     
     
     
     

