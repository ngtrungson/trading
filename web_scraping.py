# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:44:36 2018

@author: sonng
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import bs4 as bs
import pickle
import requests
import webbrowser
import datetime as dt
import urllib3



################################# Get data from website ################################################

def save_and_analyse_vnindex_tickers():
    resp = requests.get('http://www.cophieu68.vn/export.php')
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
    data.to_csv('fundemental_stocks_all_0608.csv')
    
    
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
            df = df.append(df_temp)
        except Exception as e:
            print(" Error in symbol : ", ticker) 
            print(e)
            pass
    df = df.set_index('Ticker')
    return df
    

def get_info_stock(ticker):
    url = 'http://www.cophieu68.vn/snapshot.php?id=' + ticker   
    
#    cookies = {'uid': 'sonngtrung@gmail.com', 'pass': '29011985',
#               '__cfduid': '6Lc7GnoUAAAAAHZYpAVQPW-Vr9Q-5c7BbeMni_H8'}
#    try:
#        page = requests.get(url, cookies=cookies, timeout=100).content
#        soup = bs. BeautifulSoup(page, 'lxml')
#        print(soup)
#    except Exception as e:
#            print(" Error ..... : ", e)             
#            pass
        
    resp = requests.get(url)
    print(resp.text)
    soup = bs.BeautifulSoup(resp.text, 'lxml') 
    df = pd.DataFrame(columns = ['Ticker',
                             'Close',
                             'Close_1D',
                             'Open',
                             'Low',
                             'High', 
                             'Volume',
                             'MeanVol_13W', 
                             'MeanVol_10D',
                             'High52W', 
                             'Low52W', 
                             'EPS', 
                             'PE',
                             'Market capital', 
                             'Float', 
                             'BookValue', 
                             'Beta', 
                             'ROE', 
                             'EPS_52W',
                             'CPM', 'FVQ','Exchange'])
   
   

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
#    link href="http://www.cophieu68.vn/css/screen.css?date=20180212
#    for line in soup.find(l)

    
    
    df = df.append({'Ticker':ticker, ## Getting Only The Stock Name, not 'json'
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
                         'Exchange': stockexchange}, ignore_index = True)
  
    return df  
    
def get_info_stock2(ticker):
    url = 'http://www.cophieu68.vn/snapshot.php?id=' + ticker   
    
    http = urllib3.PoolManager()
    r = http.request('GET', url)
    soup = bs.BeautifulSoup(r.data, 'lxml')
#    print(soup)
#    print (soup.title)
#    print (soup.title.text)
        
#    resp = requests.get(url)
#    print(resp.text)
#    soup = bs.BeautifulSoup(resp.text, 'lxml') 
#    print(soup)
    df = pd.DataFrame(columns = ['Ticker',
                             'Close',
                             'Close_1D',
                             'Open',
                             'Low',
                             'High', 
                             'Volume',
                             'MeanVol_13W', 
                             'MeanVol_10D',
                             'High52W', 
                             'Low52W', 
                             'EPS', 
                             'PE',
                             'Market capital', 
                             'Float', 
                             'BookValue', 
                             'Beta', 
                             'ROE', 
                             'EPS_52W',
                             'CPM', 'FVQ','Exchange'])
   
   

    value_number = []
    stockexchange = 'HSX'
   
    for line in soup.find('div', {'class':'listHeader'}).stripped_strings:
        line = line.replace(',','').replace('%','').replace(':','').replace(ticker,'').replace(' ','')
        line = line.replace('triá»\x87u','').replace('ngÃ\xa0n','').replace('(','').replace(')','')   
#        print(line)
        if isfloat(line): 
            value_number.append(float(line)) 
        
        if ((line == 'HSX')| (line == 'HNX') | (line == 'UPCOM') | (line == 'HOSE')):
            stockexchange = line
            print(line)
        else:
            stockexchange = 'BM'
            print(line)
        
    for line in soup.find('div', {'id':'snapshot_trading'}).stripped_strings:
        line = line.replace(',','').replace('%','')
        line = line.replace('triá»\x87u','').replace('ngÃ\xa0n','')   
#        print(line)
        if isfloat(line): 
            value_number.append(float(line)) 
        


    
    
    df = df.append({'Ticker':ticker, ## Getting Only The Stock Name, not 'json'
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
                         'Exchange': stockexchange}, ignore_index = True)
  
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

def get_data(symbols, dates, benchmark = '^VNINDEX', colname = '<CloseFixed>'):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df_final = pd.DataFrame(index=dates)
    if (benchmark not in symbols) and isinstance(benchmark, str):  # add SPY for reference, if absent
        symbols = [benchmark] + symbols
        
    for symbol in symbols:
        file_path = symbol_to_path(symbol)
        df_temp = pd.read_csv(file_path, parse_dates=True, index_col="<DTYYYYMMDD>",
            usecols=["<DTYYYYMMDD>", colname], na_values=["nan"])
        df_temp = df_temp.rename(columns={"<DTYYYYMMDD>": "Date", colname: symbol})
        df_final = df_final.join(df_temp)
        if symbol == benchmark:  # drop dates SPY did not trade
            df_final = df_final.dropna(subset=[benchmark])
            
#    fill_missing_values(df_final)
    
    return df_final



if __name__ == "__main__":
     
    df = get_info_stock2('VNM')
     
     
     
     
     
     

