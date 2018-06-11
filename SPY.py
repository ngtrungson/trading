# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:35:57 2017

@author: sonng
"""
import pandas as pd
from finance_util import get_data_from_web, get_data_us, fill_missing_values, optimize_portfolio, compute_portfolio, analysis_alpha_beta
from strategy import ninja_trading, hedgefund_trading, bollinger_bands, short_selling, canslim_usstock, mean_reversion
from plot_strategy import plot_hedgefund_trading, plot_ninja_trading, plot_trading_weekly,plot_shortselling_trading, plot_canslim_trading
from machine_learning import price_predictions, ML_strategy, analysis_stock




def getliststocks(typestock = "RTS"):
    #IXIC = NASDAQ
    #NYA = 	NYSE COMPOSITE (DJ)
    if typestock == "Index":
        symbols = ['^IXIC', '^DJI', '^GSPC','^NYA']
    
    if typestock == "RTS":
        symbols = ['MMM','AA', 'BABA','AMZN', 'AAPL', 'T', 'AXP','ALB', 'BB', 'BAC',
                   'BA','CAT', 'CSCO', 'C', 'KO', 'CL', 'DIS', 'DBX', 'EBAY',
                   'FB','FSLR','GE','GM', 'GOOG', 'GS', 'GPRO', 'HOG', 'IBM', 'INTC',
                   'JPM', 'LN','LMT', 'MTCH', 'MA', 'MCD', 'MSFT','NFLX', 'NKE',
                   'NOK', 'NVDA', 'PYPL', 'PEP', 'PFE', 'RACE', 'SNE', 'SBUX',
                   'SNAP', 'SPOT', 'TSLA', 'TWTR', 'UBS', 'V', 'WMT', 'YNDX','AUY', 'ZTO']
    if typestock == "RTS_IND":
        symbols = ['MMM','AA', 'BABA','AMZN', 'AAPL', 'T', 'AXP','ALB', 'BB', 'BAC',
                   'BA','CAT', 'CSCO', 'C', 'KO', 'CL', 'DIS', 'DBX', 'EBAY',
                   'FB','FSLR','GE','GM', 'GOOG', 'GS', 'GPRO', 'HOG', 'IBM', 'INTC',
                   'JPM', 'LN','LMT', 'MTCH', 'MA', 'MCD', 'MSFT','NFLX', 'NKE',
                   'NOK', 'NVDA', 'PYPL', 'PEP', 'PFE', 'RACE', 'SNE', 'SBUX',
                   'SNAP', 'SPOT', 'TSLA', 'TWTR', 'UBS', 'V', 'WMT', 'YNDX','AUY', 'ZTO'] + ['^IXIC', '^DJI', '^GSPC','^NYA']
        
    
    if typestock == "^IXIC":
        symbols = ['AMZN', 'AAPL', 'BB', 'CSCO', 'DIS', 'DBX', 'EBAY',
                   'FB','FSLR','GOOG', 'GPRO', 'INTC',
                    'MTCH',  'MSFT','NFLX',  'NVDA', 'PYPL', 'SBUX',
                   'TSLA', 'YNDX']
    
    if typestock == "^NYA":
        symbols = ['MMM','AA', 'BABA', 'T', 'AXP','ALB', 'BAC',
                   'BA','CAT', 'C', 'KO', 'CL', 
                   'GE','GM', 'GS', 'HOG', 'IBM', 
                   'JPM', 'LN','LMT', 'MA', 'MCD', 'NKE',
                   'NOK', 'PEP', 'PFE', 'RACE', 'SNE', 
                   'SNAP', 'SPOT', 'TWTR', 'UBS', 'V', 'WMT', 'AUY', 'ZTO']
        
    if typestock == "ALL":
        symbols = ['MMM', 'ABT', 'ABBV', 'ACN', 'ATVI', 'AYI', 'ADBE', 'AMD', 'AAP', 'AES', 
               'AET', 'AMG', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALXN', 
               'ALGN', 'ALLE', 'AGN', 'ADS', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 
               'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 
               'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'ANDV', 'ANSS', 'ANTM', 
               'AON', 'AOS', 'APA', 'AIV', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ARNC', 
               'AJG', 'AIZ', 'T', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'BHGE', 'BLL', 
               'BAC', 'BK', 'BAX', 'BBT', 'BDX', 'BBY', 'BIIB', 'BLK', 
               'HRB', 'BA', 'BWA', 'BXP', 'BSX', 'BHF', 'BMY', 'AVGO',  
               'CHRW', 'CA', 'COG', 'CDNS', 'CPB', 'COF', 'CAH', 'CBOE', 'KMX', 
               'CCL', 'CAT', 'CBS', 'CELG', 'CNC', 'CNP', 'CTL', 'CERN', 
               'CF', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 
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
               'FMC', 'FL', 'F', 'FTV', 'FBHS', 'BEN', 'FCX',  'GRMN', 'IT',
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
               'PXD', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PFG', 'PG', 'PGR', 
               'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 
               'RRC', 'RJF', 'RTN', 'O', 'RHT', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 
               'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RCL', 'CRM', 'SBAC', 'SCG', 'SLB',
               'STX', 'SEE', 'SRE', 'SHW', 'SIG', 'SPG', 'SWKS', 'SLG', 'SNA', 
               'SO', 'LUV', 'SPGI', 'SWK', 'SBUX', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 
               'SYF', 'SNPS', 'SYY', 'TROW', 'TPR', 'TGT', 'TEL', 'FTI', 'TXN', 'TXT', 
               'TMO', 'TIF', 'TWX', 'TJX', 'TMK', 'TSS', 'TSCO', 'TDG', 'TRV', 'TRIP', 
               'FOXA', 'FOX', 'TSN', 'UDR', 'ULTA', 'USB', 'UAA', 'UA', 'UNP', 'UAL', 
               'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'VFC', 'VLO', 'VAR', 'VTR', 
               'VRSN', 'VRSK', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA', 
               'DIS', 'WM', 'WAT', 'WEC', 'WFC', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 
               'WMB', 'WLTW', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YUM', 
               'ZBH', 'ZION', 'ZTS']
#    symbols =  high_cpm
#    symbols = symbols 
    symbols = pd.unique(symbols).tolist()
    symbols = sorted(symbols)
    
    return symbols

    
    
def analysis_trading(tickers, start, end, update = False, source = "yahoo"):
    
    
   
    for ticker in tickers:
#        print(" Analysing ..." , ticker)
        try:
#            ninja_trading(ticker, start, end, realtime = update, source = source)
#            hedgefund_trading(ticker, start, end, realtime = update, source = source)
            canslim_usstock(ticker, start, end, realtime = update, source = source)
#            mean_reversion(ticker, start, end, realtime = update, source = source)
#            bollinger_bands(ticker, start, end, realtime = update, source = source)
#            short_selling(ticker, start, end, realtime = update, source = source)
        except Exception as e:
            print (e)
            print("Error in reading symbol: ", ticker)
            pass
   
 
def passive_strategy(start_date, end_date, market = "SPY"):

    symbols = getliststocks(typestock = market)
    
    dates = pd.date_range(start_date, end_date)  # date range as index
    df_data = get_data_us(symbols, dates, benchmark = market)  # get data for each symbol
    
    df_volume = get_data_us(symbols, dates, benchmark = None, colname = 'Volume')  # get data for each symbol
    df_high = get_data_us(symbols, dates, benchmark = None, colname = 'High')
    df_low = get_data_us(symbols, dates, benchmark = None, colname = 'Low')
    
   
    vol_mean = pd.Series(df_volume.mean(),name = 'Volume')
    max_high = pd.Series(df_high.max(), name = 'MaxHigh')
    min_low = pd.Series(df_low.min(), name = 'MinLow')
    cpm = pd.Series(max_high/min_low, name = 'CPM')
    df_value = df_volume*df_data
    value_mean = pd.Series(df_value.mean(), name = 'Value')
    # Fill missing values
    fill_missing_values(df_data)

    
    # Assess the portfolio
    
    allocations, cr, adr, sddr, sr  = optimize_portfolio(sd = start_date, ed = end_date,
        syms = symbols,  benchmark = market, country = 'US', gen_plot = True)

     # Print statistics
    print ("Start Date:", start_date)
    print ("End Date:", end_date)
    print ("Symbols:", symbols)
    print ("Optimal allocations:", allocations)
    print ("Sharpe Ratio:", sr)
    print ("Volatility (stdev of daily returns):", sddr)
    print ("Average Daily Return:", adr)
    print ("Cumulative Return:", cr)
    
    investment = 50000000
    df_result = pd.DataFrame(index = symbols)    
    df_result['Opt allocs'] = allocations
    df_result['Cash'] = allocations * investment
    df_result['Volume'] = vol_mean
    df_result['Close'] = df_data[symbols].iloc[-1,:].values
    df_result['Value'] = value_mean
    #    df_result['MaxH'] = max_high
#    df_result['MinL'] = min_low
    df_result['CPM'] = cpm
    df_result['Shares'] = round(df_result['Cash']/df_result['Close'].values/1000,0)
    df_result ['Volatility'] = df_data[symbols].pct_change().std() 
    
    alpha_beta = analysis_alpha_beta(df_data, symbols, market)
    df_result['Alpha'] = alpha_beta['Alpha']
    df_result['Beta'] = alpha_beta['Beta']
    
    relative_strength = 40*df_data[symbols].pct_change(periods = 63).fillna(0) \
                     + 20*df_data[symbols].pct_change(periods = 126).fillna(0) \
                     + 20*df_data[symbols].pct_change(periods = 189).fillna(0) \
                     + 20*df_data[symbols].pct_change(periods = 252).fillna(0) 
             
            
    
    df_result ['RSW'] = relative_strength.iloc[-1,:].values

    return df_result, df_data


def analysis_stocks_RTS(start_date, end_date, margin= 25, investment_size = 200):

    
    df_result = pd.read_csv('dataRTS.csv', index_col='Ticker')

    df_result = df_result.sort_index()
    symbols = df_result.index
    
    dates = pd.date_range(start_date, end_date)  # date range as index
    df_data = get_data_us(symbols, dates, benchmark = None)  # get data for each symbol
    
    # Fill missing values
    fill_missing_values(df_data)
#    return prices
#    df_result['Commision'] = prices['Commision']
    
    df_result['Close'] = df_data[symbols].iloc[-1,:].values
    df_result['MinNbStock'] = df_result['Lot']* 0.01
    df_result['PriceStockMarg'] = df_result['MinNbStock'] * df_result['Close']/margin    
    
    df_result['NbMaxVol'] = round(investment_size/df_result['PriceStockMarg'], 0)
    df_result['MaxVolLot'] = round(investment_size/df_result['PriceStockMarg']*0.01, 2)
    
    df_result['CommMin'] = (df_result['Spread']- df_result['Buy'] - df_result['Sell'])/df_result['Lot']* df_result['MinNbStock']   
    df_result['Comm_Trade'] = (df_result['Spread']- df_result['Buy'] - df_result['Sell'])/df_result['Lot']*df_result['NbMaxVol']
       
    df_result['Comm_Price'] = round(df_result['CommMin']/df_result['MinNbStock']/df_result['Close'].values*100, 3)
    df_result['Comm_Invest'] = round(df_result['Comm_Trade']/investment_size*100, 3)
  
    
    relative_strength = 40*df_data[symbols].pct_change(periods = 63).fillna(0) \
                     + 20*df_data[symbols].pct_change(periods = 126).fillna(0) \
                     + 20*df_data[symbols].pct_change(periods = 189).fillna(0) \
                     + 20*df_data[symbols].pct_change(periods = 252).fillna(0) 
     
    df_result ['RSW'] = relative_strength.iloc[-1,:].values

    return df_result, df_data

def analysis_single_stock(ticker, bid, ask, lot = 100, over_night = 5, investment = 200, margin = 25, stoploss = 0.03, takeprofit = 0.08):
    spread = (ask - bid)*100
    close = ask
    nb_stock_min = lot * 0.01
    
    price_stock_mg = nb_stock_min* close/margin 
    print(ticker, " price min_volume with margin", price_stock_mg)
    nb_min_vol = round(investment/price_stock_mg, 0)
    commision = (spread + over_night)/lot*nb_min_vol
    print(ticker, " commision min_volume with margin overnight", commision, " x 0.01")
    print(ticker, " min_volume with $", investment, ":", nb_min_vol, " x 0.01")
    print(ticker, " commision ratio with invesment with margin:", round(commision/investment*100, 2))
    

    
if __name__ == "__main__":
    
#    analysis_single_stock(ticker = 'TSLA', 
#                          bid = 317.21, 
#                          ask = 318.21, 
#                          lot = 50, 
#                          over_night = 4.36, 
#                          investment = 200, 
#                          margin = 10)
#
    end_date = "2018-6-9"
    start_date = "2015-1-1"
    
    symbols = getliststocks(typestock = "RTS")

#    get_data_from_web(tickers = symbols, start = start_date, end = end_date, source ='yahoo', redownload = True)
    stock_res, stock_data = analysis_stocks_RTS(start_date = start_date, end_date = end_date)
#    analysis_stock(symbols, stock_data, start_date, end_date)

#    stock_alloc, stock_data = passive_strategy(start_date = start_date, end_date = end_date, market = "^IXIC")
#    analysis_trading(symbols, start = start_date , end = end_date, update = False, source = "yahoo")
#    ticker = 'NVDA'    
#    shortselling = short_selling(ticker, start_date, end_date, realtime = False, source ="yahoo")    
#    plot_hedgefund_trading(ticker, hedgefund)
    
   