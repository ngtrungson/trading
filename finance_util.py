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
import scipy.optimize as spo


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
    data.to_csv('fundemental_stocks_all.csv')
    
    
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
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text, 'lxml') 
    df = pd.DataFrame(columns = ['Ticker',
                             'Close',
                             'Close_1D',
                             'Open',
                             'Low',
                             'High', 
                             'Volume',
                             'MeanVol_13W', 
                             'MeanVol_10W',
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
                             'CPM'])
   
   

    value_number = []
    
    for line in soup.find('div', {'class':'listHeader'}).stripped_strings:
        line = line.replace(',','').replace('%','')
        line = line.replace('triá»\x87u','').replace('ngÃ\xa0n','')         
        if isfloat(line): 
            value_number.append(float(line)) 
    for line in soup.find('div', {'id':'snapshot_trading'}).stripped_strings:
        line = line.replace(',','').replace('%','')
        line = line.replace('triá»\x87u','').replace('ngÃ\xa0n','')        
        if isfloat(line): 
            value_number.append(float(line)) 
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
                         'MeanVol_10W' : value_number[7],
                         'High52W' : value_number[8], 
                         'Low52W' : value_number[9], 
                         'EPS' : value_number[10], 
                         'PE' : value_number[11],
                         'Market capital' : value_number[12], 
                         'Float' : value_number[13], 
                         'BookValue' : value_number[14], 
                         'ROE' : value_number[15], 
                         'Beta' : value_number[16], 
                         'EPS_52W' : value_number[17],
                         'CPM': value_number[8]/value_number[9],
                         'FVQ': value_number[13]/value_number[6]*1E6}, ignore_index = True)
  
    return df  
    

def get_data_from_cophieu68_openwebsite(tickers):       
    file_url = 'http://www.cophieu68.vn/export/excelfull.php?id='
    for ticker in tickers:
       webbrowser.open(file_url+ticker)
       

def get_data_from_SSI_website(tickers):       
    file_url = 'http://ivt.ssi.com.vn/Handlers/DownloadHandler.ashx?Download=1&Ticker='
    for ticker in tickers:
       webbrowser.open(file_url+ticker)
   
    
def symbol_to_path(symbol, base_dir="cp68"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "excel_{}.csv".format(str(symbol)))

def symbol_to_path_ssi(symbol, base_dir="ssi"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "Historical_Price_{}.csv".format(str(symbol)))

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

    return df_final


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
    syms=['GOOG','AAPL','GLD','XOM'], benchmark = '^VNINDEX', gen_plot=False):
    
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates, benchmark = benchmark)  # automatically adds SPY
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
#    symbolsHNX = ['TNG', 'BVS', 'PVX', "KDM", "ASA", "HKB", "HVA", "KLF", "VE9", 
#                  'ACB', 'BCC', 'CEO', 'DBC', 'DCS', 'HHG', 'HUT',
#                  'LAS',  'MBS', 'NDN', 'PGS', 'PVC', 'PVI',
#                  'PVS', 'S99','SHB', 'SHS', 'VC3', 'VCG','VCS', 'VGC']
#    
#    symbolsVNI = [ "ASM", "BFC", "BID", "BMI", "BMP", "BVH",
#              "CII", "CTD", "CAV", "CMG", "CSM", "CSV", "CTG",  
#           "DCM","DHG", "DIG", "DLG", "DPM","DPR", "DRH",  "DQC", "DRC", "DXG", 
#           "ELC", "EVE","FCN","FIT","FLC","FPT", "GAS", "GMD", "GTN", 
#           "HAG", "HHS", "HNG", "HQC", "HT1", "HVG",
#           "HSG", "HDG", "HCM", "HPG", "HBC", 
#           "IJC", "ITA", "KBC", "KSB",  "KDH", "KDC", 
#           "MBB", "MSN", "MWG", "NKG", "NLG", "NT2", "NVL", "NBB",
#            "PVT","PVD","PHR","PGI","PDR","PTB", "PNJ",  "PC1",   "PLX", "PPC", "PAC",
#            "QCG", "REE",  "SAM","SJD","SJS","STB","STG","SKG",  "SSI", "SBT", "SAB", 
#                "VSH","VNM", "VHC", "VIC", "VCB", "VSC", "VJC", "VNS" ,
#                'ITC','LSS','VOS', 'OGC', 'PME', 'PAN','TCH', 'GEX','VCI',
#                'TDC','TCM', 'VNE','KSA', 'SHN', 'AAA','SCR', 'AGR',
#                'EIB','BHN','VPB','VRE','ROS',"VND", "HDB","NVT","VHG", "SMC", "C32","CVT"]
#    
#    symbolsUPCOM = ["SBS", "SWC", "NTC","DVN"]
#    
#    symbols = symbolsVNI + symbolsHNX +  symbolsUPCOM
# 
#    data = fundemental_analysis(symbols)
#    
#    data.to_csv('fundemental_stocksVN.csv')
    
#    tickers = save_and_analyse_vnindex_tickers()
    
     data = pd.read_csv('fundemental_stocks_all.csv', parse_dates=True, index_col=0)
     data['Diff_Price'] = data['Close'] - data['EPS']*data['PE']/1000
     data['EPS_Price'] = data['EPS']/data['Close']/1000
     df = data.query("MeanVol_10W > 100000")
     df = df.query("FVQ > 0")
     df = df.query("CPM > 1.4")
     df = df.query("EPS > 0")
     df = df.query("Diff_Price < 0")
#     df.to_csv('investment_stock3.csv')
#     print(df.index)
     
     
     

