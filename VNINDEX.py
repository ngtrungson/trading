# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:35:57 2017

@author: sonng
"""
import pandas as pd
from finance_util import get_data, fill_missing_values, optimize_portfolio, compute_portfolio, \
                         get_data_from_cophieu68_openwebsite, get_data_from_SSI_website
from strategy import ninja_trading, hedgefund_trading, bollinger_bands, short_selling
from plot_strategy import plot_hedgefund_trading, plot_ninja_trading, plot_trading_weekly,plot_shortselling_trading
from machine_learning import price_predictions, ML_strategy


def getliststocks(typestock = "^VNINDEX"):
    benchmark = ["^VNINDEX", "^HASTC", "^UPCOM"]
    symbolsHNX = ['APS', 'ALV', 'C69', 'TNG', 'BVS', 'PVX', "KDM", "ASA", "HKB", "HVA", 'NVB', "KLF", 'KVC', "VE9", 
                  'ACB', 'BCC', 'CVN', 'CEO', 'DBC', 'DCS', 'DST','HHG', 'HUT', 'SD9', 'HLD', 'NSH', 'DPS','DS3',
                  'LAS',  'MBS', 'NDN', 'PGS', 'PVC', 'PVI',  'MST', 'PHC', 'PVE', 'PVG', 'PVB',
                  'PVS', 'S99','SHB', 'SHS', 'TTB','VC3', 'VCG','VCS', 'VGC','VMC','VIX', 'TVC',  'TIG', 'SPP',
                  'VIG','VKC']
    
    symbolsVNI = [ 'AMD', 'ATG', 'ASP', 'APG', 'APC', 'ANV', "ASM", "BSI", "BWE", 
                  'BCG', "BFC", "BID", "BMI", "BMP", "BVH", 'CDO',  'CTS', 'CTI', "CII", "CTD", "CAV", "CMG", "CSM", "CSV", "CTG", 'CCL', 'CHP', 'C47', 
               "DCM","DHG", "DIG", "DLG", "DPM","DPR", "DRH",  "DQC", "DRC", "DXG", 'DGW', 'DHA', 'DHC', 'DAH',
               'DHM', 
               "ELC", "EVE", 'EVG', "FCN","FIT","FLC", 'FMC', 'FTS', "FPT", "GAS", "GMD", "GTN", 
                'HAX', "HAG", "HHS", "HNG", "HQC", "HT1", "HVG", 'HAI', 'HAR', 'HID', 'HII', 'HTT',
               "HSG", "HDG", "HCM", "HPG", "HBC", 'LDG', 'LCG', 'LGL', 'LHG', 'HDC',
               'IDI', "IJC", "ITA", "KBC", "KSB",  "KDH", "KDC", 'KSH',
               "MBB", "MSN", "MWG", "NKG", "NLG", "NT2", "NVL", "NBB",
                "PVT","PVD","PHR","PGI","PDR","PTB", "PNJ",  "PC1",   "PLX", "PXS",
                "PPC", "PAC", 'QBS', "QCG", "REE",  
                'SHI',"SAM","SJD","SJS","STB","STG","SKG",  "SSI", "SBT", "SAB", 
                "VSH","VNM", "VHC", "VIC", "VCB", "VSC", "VJC", "VNS" , 'TVS', 'VDS', 'TNI',
                'ITC','LSS','VOS', 'OGC', 'PME', 'PAN','TCH', 'TDH', 'TNT', 'TTF','GEX','VCI', 'VIS',
                'TDC','TCM', 'VNE','KSA', 'SHN', 'AAA','SCR', 'AGR', 'TSC', 'TDG', 'VRC', 'JVC', 'SRC',
                'EIB','BHN','VPB','VRE','ROS',"VND", "HDB", "NVT","VHG", "SMC", "C32","CVT",'VPH','VNG','VIP']
    
    symbolsUPCOM = ['TOP', 'TBD', 'LPB', 'QNS', 'RCC', 'ATB', 'ART',  'ACV', "SBS", "SWC", "NTC","DVN", 'HVN', 'HPI','IDC',  'MSR', 'PXL', 'VGT','TVN','TVB','TIS','VIB']
    
    if typestock == "ALL":
        symbols = benchmark + symbolsVNI + symbolsHNX + symbolsUPCOM 
    if typestock == "^VNINDEX":
        symbols = symbolsVNI
    if typestock == "^HASTC":
        symbols = symbolsHNX
    if typestock == "^UPCOM":
        symbols = symbolsUPCOM
    if typestock == "TICKER":
        symbols = symbolsVNI + symbolsHNX + symbolsUPCOM 
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
    
    
def analysis_trading(tickers, start, end, update = False, source = "cp68"):
    
    if tickers == None:
        tickers = getliststocks(typestock = "TICKER")
        
    for ticker in tickers:
#        print(" Analysing ..." , ticker)
        try:
            ninja_trading(ticker, start, end, realtime = update, source = source)
#            hedgefund_trading(ticker, start, end, realtime = update, source = source)
#            bollinger_bands(ticker, start, end, realtime = update, source = source)
#            short_selling(ticker, start, end, realtime = update, source = source)
        except Exception as e:
            print (e)
            print("Error in reading symbol: ", ticker)
            pass
               
def get_csv_data(source = "cp68"):

    symbols = getliststocks(typestock = "ALL")
     
    if source == "cp68":
        get_data_from_cophieu68_openwebsite(symbols)
    else:
       get_data_from_SSI_website(symbols) 
    return symbols
 

            
def predict_stocks(tickers, start, end):
    for ticker in tickers:
        print('Prediction of ticker .................' , ticker)
        price_predictions(ticker, start, end, forecast_out = 5)
        print(' End of prediction ticker ...................', ticker)

def passive_strategy(start_date, end_date, market = "^VNINDEX"):

    symbols = getliststocks(typestock = market)
    
    dates = pd.date_range(start_date, end_date)  # date range as index
    df_data = get_data(symbols, dates, benchmark = market)  # get data for each symbol
    
    df_volume = get_data(symbols, dates, benchmark = None, colname = '<Volume>')  # get data for each symbol
    df_high = get_data(symbols, dates, benchmark = None, colname = '<High>')
    df_low = get_data(symbols, dates, benchmark = None, colname = '<Low>')
    
    
    vol_mean = pd.Series(df_volume.mean(),name = 'Volume')
    max_high = pd.Series(df_high.max(), name = 'MaxHigh')
    min_low = pd.Series(df_low.min(), name = 'MinLow')
    cpm = pd.Series(max_high/min_low, name = 'CPM')
    # Fill missing values
    fill_missing_values(df_data)

    
    # Assess the portfolio
    
    allocations, cr, adr, sddr, sr  = optimize_portfolio(sd = start_date, ed = end_date,
        syms = symbols,  benchmark = market, gen_plot = True)

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
    #    df_result['MaxH'] = max_high
#    df_result['MinL'] = min_low
    df_result['CPM'] = cpm
    

    return df_result, df_data


def rebalancing_porfolio(symbols = None, bench = '^VNINDEX'):

   
    start0 = "2015-1-2"
    end0 = "2017-1-2"
    allocations, cr, adr, sddr, sr  = optimize_portfolio(sd = start0, ed = end0,
            syms = symbols,  benchmark = bench, gen_plot = True)
    print ("Optimize start Date:", start0)
    print ("Optimize end Date:", end0) 
    print ("Optimize volatility (stdev of daily returns):", sddr)
    print ("Optimize average Daily Return:", adr)
    print ("Optimize cumulative Return:", cr)
    print(" -----------------------------------------------------")
    start_date_list = ["2017-1-3", "2017-7-3"]
    end_date_list = ["2017-7-2",  "2018-4-1"]
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
    
    end_date = "2018-4-26"
    start_date = "2018-4-2"
    
    cr, adr, sddr, sr  = compute_portfolio(sd = start_date, ed = end_date,
            syms = symbols, allocs = allocations, benchmark = bench, gen_plot = True)
    print("....................... Out of sample performance .................")
    print ("Start Date:", start_date)
    print ("End Date:", end_date) 
    print ("Volatility (stdev of daily returns):", sddr)
    print ("Average Daily Return:", adr)
    print ("Cumulative Return:", cr)  
    # Assess the portfolio
    investment = 60000000
    df_result = pd.DataFrame(index = symbols)    
    df_result['Opt allocs'] = allocations
    df_result['Cash'] = allocations * investment

    dates = pd.date_range(start_date, end_date)  # date range as index
    df_data = get_data(symbols, dates, benchmark = bench)  # get data for each symbol
    
   
    df_high = get_data(symbols, dates, benchmark = None, colname = '<High>')
    df_low = get_data(symbols, dates, benchmark = None, colname = '<Low>')
    
    max_high = pd.Series(df_high.max(), name = 'MaxHigh')
    min_low = pd.Series(df_low.min(), name = 'MinLow')
    cpm = pd.Series(max_high/min_low, name = 'CPM')
    volatility = df_data[symbols].pct_change().std()  
    # Fill missing values
            
    df_result['Close'] = df_data[symbols].iloc[-1,:].values    
    df_result['CPM'] = cpm
    df_result['Shares'] = round(df_result['Cash']/df_result['Close'].values/1000,0)
    df_result ['Volatility'] = volatility

    return df_result
    

    
if __name__ == "__main__":
#    symbols = get_csv_data(source = "cp68")
#    symbols = get_csv_data()
#    symbols = get_stocks_highcpm(download = False, source ="cp68")
    
#    symbols =  ['FTS', 'PVI', 'VNE']

#    analysis_trading(symbols, start = "2017-3-1" , end = "2018-4-11", update = False, source = "cp68")


    
#    VNI_result, VNI_data  = passive_strategy(start_date = "2017-3-26" , end_date = "2018-4-24", market= "^VNINDEX")
    

#    ticker = 'VGC'    
#
#    end_date = "2018-4-5"
#    start_date = "2016-4-5"
###    bollingerbands = bollinger_bands(ticker, start_date, end_date, realtime = False, source = "cp68")
##    
#    hedgefund = hedgefund_trading(ticker, start_date, end_date, realtime = False, source ="cp68")    
#    plot_hedgefund_trading(ticker, hedgefund)
###    
###    shortsell = short_selling(ticker, start_date, end_date, realtime = False, source ="ssi")    
###    plot_shortselling_trading(ticker, shortsell)
###    
###
###    
#    ninja = ninja_trading(ticker, start_date, end_date, realtime = False,  source ="cp68")    
#    plot_ninja_trading(ticker, ninja)
    
#    plot_trading_weekly(ticker, hedgefund)
#    
#    investment_stocks = ['CII', 'HPG', 'NBB', 'STB', 'PAN', 'VND' ]
    

#    analysis_trading(tickers = None, start = "2017-3-26" , end = "2018-4-26", update = False,  source ="cp68")
    
    
    symbolsVNI = getliststocks(typestock = "^VNINDEX")
#    symbolsHNX = getliststocks(typestock = "^HASTC")
    ALLOC_opt = rebalancing_porfolio(symbols = symbolsVNI, bench = '^VNINDEX')
    
#    investing = ['NVB', 'MBS', 'FPT', 'TVN', 'VIX']
#    predict_stocks(investing, start ="2010-3-18", end = "2018-4-13")
   
#    ML_strategy('ACB', start ="2008-1-2", end = "2018-4-26")
#    tickers = pd.Series(symbols)
    