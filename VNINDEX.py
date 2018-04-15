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
from machine_learning import price_predictions


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
    
def analysis_stocks(start, end, update = False, source = "ssi"):
    
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
    
    
    
    symbols = symbolsVNI + symbolsHNX + symbolsUPCOM 
 
   
#    symbols = symbolsVNI 
    
    tickers  = pd.unique(symbols).tolist()
    tickers = sorted(tickers)
    
    for ticker in tickers:
#        print("Analysing ...", ticker)

#        ninja_trading(ticker, start, end, realtime = update, source = source)
        hedgefund_trading(ticker, start, end, realtime = update, source = source)
#        bollinger_bands(ticker, start, end, realtime = update, source = source)
#        short_selling(ticker, start, end, realtime = update, source = source)

    
    
def analysis_trading(tickers, start, end, update = False, source = "cp68"):
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
    
   
           
#    symbols = symbolsVNI + symbolsHNX + symbolsUPCOM + symbolother + high_cpm
    symbols = benchmark + symbolsVNI + symbolsHNX + symbolsUPCOM 
#    symbols =  high_cpm
    symbols = pd.unique(symbols).tolist()
     
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

def test_runVNINDEX():

    
#    symbolsVNI = [ "BIC", "BMI", "BMP", "BVH","CII", "CTD", "CAV", "CTG", 
#               "DHG", "DVP",  "DQC", "DRC", "DXG", "FPT",
#               "HSG", "HDG", "HCM", "HPG", "HBC", "IMP",  "GAS", "GMD", 
#               "KDH", "KDC", "LIX", "MBB", "MSN", "MWG", 
#                "NKG", "NLG", "NT2", "NTP", "NVL", "PTB", "PAN", "PNJ",  "PC1",   "PLX", "PPC", "PAC",
#                   "REE",   "RAL",  "SKG",  "SSI", "SBT", "SAB", "TLG", "TMS", "TRA",
#                    "VNM", "VHC", "VIC", "VCB", "VSC", "VCF", "VJC", "VNS" ]
    
#    symbolsVNI = [ "ASM", "BCI", "BFC", "BHS" ,"BIC", "BID", "BMI", "BMP", "BVH",
#                  "CII", "CTD", "CAV", "CMG","CNG", "CSM", "CSV", "CTG",  
#               "DCM","DHG", "DIG", "DLG","DMC", "DPM","DPR", "DRH", "DVP",  "DQC", "DRC", "DXG", 
#               "ELC", "EVE","FCN","FIT","FLC","FPT", "GAS", "GMD", "GTN", 
#               "HAG", "HAH", "HHS", "HNG", "HQC", "HT1","HTL","HVG",
#               "HSG", "HDG", "HCM", "HPG", "HBC", 
#               "IJC","IMP", "ITA", "KBC", "KSB",  "KDH", "KDC", 
#               "LIX", "MBB", "MSN", "MWG", 
#                "NKG", "NLG", "NT2", "NTP", "NVL", "NCT","NBB","NNC","NSC",
#                "PVT","PVD","PHR","PGI","PGD","PDR","PTB", "PAN", "PNJ",  "PC1",   "PLX", "PPC", "PAC",
#                "QCG", "REE",   "RAL",
#                "SAM","SHP","SJD","SJS","STB","STG","STK","SKG",  "SSI", "SBT", "SAB", "TLG", "TMS", "TRA",
#                    "VFG","VSH","VNM", "VHC", "VIC", "VCB", "VSC", "VCF", "VJC", "VNS" ]
    
    
#    
#    symbolsVNI = [ "ASM", "BFC", "BID", "BMI", "BMP", "BVH",
#                  "CII", "CTD", "CAV", "CMG", "CSM", "CSV", "CTG",  
#               "DCM","DHG", "DIG", "DLG", "DPM","DPR", "DRH",  "DQC", "DRC", "DXG", 
#               "ELC", "EVE","FCN","FIT","FLC","FPT", "GAS", "GMD", "GTN", 
#               "HAG", "HHS", "HNG", "HQC", "HT1", "HVG",
#               "HSG", "HDG", "HCM", "HPG", "HBC", 
#               "IJC", "ITA", "KBC", "KSB",  "KDH", "KDC", 
#               "MBB", "MSN", "MWG", 
#                "NKG", "NLG", "NT2", "NVL", "NBB",
#                "PVT","PVD","PHR","PGI","PDR","PTB", "PNJ",  "PC1",   "PLX", "PPC", "PAC",
#                "QCG", "REE",  
#                "SAM","SJD","SJS","STB","STG","SKG",  "SSI", "SBT", "SAB", 
#                    "VSH","VNM", "VHC", "VIC", "VCB", "VSC", "VJC", "VNS" ]
    
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

#    update = False
#    # Read data
#    if update:
#        get_data_from_cophieu68_openwebsite(symbolsVNI)
    
#    symbols = ["VCG", "VCB", "VSC", "FCN"]  # list of symbols
    end_date = "2018-4-12"
    start_date = "2018-2-1"

    dates = pd.date_range(start_date, end_date)  # date range as index
    df_data = get_data(symbolsVNI, dates, benchmark = '^VNINDEX')  # get data for each symbol
    
    df_volume = get_data(symbolsVNI, dates, benchmark = None, colname = '<Volume>')  # get data for each symbol
    df_high = get_data(symbolsVNI, dates, benchmark = None, colname = '<High>')
    df_low = get_data(symbolsVNI, dates, benchmark = None, colname = '<Low>')
    
    
    vol_mean = pd.Series(df_volume.mean(),name = 'Volume')
    max_high = pd.Series(df_high.max(), name = 'MaxHigh')
    min_low = pd.Series(df_low.min(), name = 'MinLow')
    cpm = pd.Series(max_high/min_low, name = 'CPM')
    # Fill missing values
    fill_missing_values(df_data)

    
    # Assess the portfolio
    
    allocations, cr, adr, sddr, sr  = optimize_portfolio(sd = start_date, ed = end_date,
        syms = symbolsVNI,  benchmark = '^VNINDEX', gen_plot = True)

     # Print statistics
    print ("Start Date:", start_date)
    print ("End Date:", end_date)
    print ("Symbols:", symbolsVNI)
    print ("Optimal allocations:", allocations)
    print ("Sharpe Ratio:", sr)
    print ("Volatility (stdev of daily returns):", sddr)
    print ("Average Daily Return:", adr)
    print ("Cumulative Return:", cr)
    
    investment = 50000000
    df_result = pd.DataFrame(index = symbolsVNI)    
    df_result['Opt allocs'] = allocations
    df_result['Cash'] = allocations * investment
    df_result['Volume'] = vol_mean
    df_result['Close'] = df_data[symbolsVNI].iloc[-1,:].values
    #    df_result['MaxH'] = max_high
#    df_result['MinL'] = min_low
    df_result['CPM'] = cpm
    
#    ticker = 'STB'
##    
#    trading = ninja_trading(ticker, start_date, end_date)
##    
#    plot_ninja_trading(ticker, trading)
#    

    
#    analysis_trading(symbolsVNI, start_date, end_date)
#    investing = ['SHB', 'PVS', 'NDN', 'DVN', 'BMI']
#    analysis_stock(symbolsVNI, df_data, start_date, end_date)
#    predict_stocks(investing, start ="2010-2-5", end = "2018-2-5")
    return df_result, df_data


def rebalancing_porfolio(symbols = None, bench = '^VNINDEX'):

   
   


    start0 = "2015-1-2"
    end0 = "2017-1-2"
    allocations, cr, adr, sddr, sr  = optimize_portfolio(sd = start0, ed = end0,
            syms = symbols,  benchmark = bench, gen_plot = True)
    print ("Start Date:", start0)
    print ("End Date:", end0) 
    print ("Volatility (stdev of daily returns):", sddr)
    print ("Average Daily Return:", adr)
    print ("Cumulative Return:", cr)
    print(" -----------------------------------------------------")
    start_date_list = ["2017-1-3", "2017-7-3",  "2018-1-3"]
    end_date_list = ["2017-7-2",  "2018-1-2", "2018-4-13"]
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
        
        # Assess the portfolio
    investment = 50000000
    df_result = pd.DataFrame(index = symbols)    
    df_result['Opt allocs'] = allocations
    df_result['Cash'] = allocations * investment
    
    
    end_date = "2018-4-13"
    start_date = "2018-2-1"

    dates = pd.date_range(start_date, end_date)  # date range as index
    df_data = get_data(symbols, dates, benchmark = bench)  # get data for each symbol
    
   
    df_high = get_data(symbols, dates, benchmark = None, colname = '<High>')
    df_low = get_data(symbols, dates, benchmark = None, colname = '<Low>')
    
    max_high = pd.Series(df_high.max(), name = 'MaxHigh')
    min_low = pd.Series(df_low.min(), name = 'MinLow')
    cpm = pd.Series(max_high/min_low, name = 'CPM')
    # Fill missing values
    fill_missing_values(df_data)
    
    
    df_result['Close'] = df_data[symbols].iloc[-1,:].values    
    df_result['CPM'] = cpm
    df_result['Shares'] = df_result['Cash']/df_result['Close'].values/1000

    return df_result
    
def test_runUPCOM():

 
    
    symbolsUPCOM = ['TOP', 'TBD', 'LPB', 'QNS', 'RCC', 'ATB', 'ART',  'ACV', "SBS", "SWC", "NTC","DVN", 
                   'HVN', 'HPI','IDC',  'MSR', 'PXL', 'VGT','TVN','TVB','TIS','VIB']
    
    
    
#    update = False
#    # Read data
#    if update:
#        get_data_from_cophieu68_openwebsite(symbolsVNI)
    
#    symbols = ["VCG", "VCB", "VSC", "FCN"]  # list of symbols
    end_date = "2018-4-12"
    start_date = "2018-1-2"

    dates = pd.date_range(start_date, end_date)  # date range as index
    df_data = get_data(symbolsUPCOM, dates, benchmark = '^UPCOM')  # get data for each symbol
    
    df_volume = get_data(symbolsUPCOM, dates, benchmark = None, colname = '<Volume>')  # get data for each symbol
    df_high = get_data(symbolsUPCOM, dates, benchmark = None, colname = '<High>')
    df_low = get_data(symbolsUPCOM, dates, benchmark = None, colname = '<Low>')
    
    
    vol_mean = pd.Series(df_volume.mean(),name = 'Volume')
    max_high = pd.Series(df_high.max(), name = 'MaxHigh')
    min_low = pd.Series(df_low.min(), name = 'MinLow')
    cpm = pd.Series(max_high/min_low, name = 'CPM')
    # Fill missing values
    fill_missing_values(df_data)

    
    # Assess the portfolio
    
    allocations, cr, adr, sddr, sr  = optimize_portfolio(sd = start_date, ed = end_date,
        syms = symbolsUPCOM,  benchmark = '^UPCOM', gen_plot = True)

     # Print statistics
    print ("Start Date:", start_date)
    print ("End Date:", end_date)
    print ("Symbols:", symbolsUPCOM)
    print ("Optimal allocations:", allocations)
    print ("Sharpe Ratio:", sr)
    print ("Volatility (stdev of daily returns):", sddr)
    print ("Average Daily Return:", adr)
    print ("Cumulative Return:", cr)
    
    investment = 5000000
    df_result = pd.DataFrame(index = symbolsUPCOM)    
    df_result['Opt allocs'] = allocations
    df_result['Cash'] = allocations * investment
    df_result['Volume'] = vol_mean
    df_result['Close'] = df_data[symbolsUPCOM].iloc[-1,:].values
    #    df_result['MaxH'] = max_high
#    df_result['MinL'] = min_low
    df_result['CPM'] = cpm
    
#    ticker = 'STB'
##    
#    trading = ninja_trading(ticker, start_date, end_date)
##    
#    plot_ninja_trading(ticker, trading)
#    

    
#    analysis_trading(symbolsVNI, start_date, end_date)
#    investing = ['SHB', 'PVS', 'NDN', 'DVN', 'BMI']
#    analysis_stock(symbolsVNI, df_data, start_date, end_date)
#    predict_stocks(investing, start ="2010-2-5", end = "2018-2-5")
    return df_result, df_data
  
def test_runHNX():
   
#    
#    symbolsHNX = ['ACB', 'BCC', 'BVS', 'CEO', 'DBC', 'DCS', 'DGC', 'HHG', 'HUT',
#                  'IDV', 'LAS', 'LHC','MAS', 'MBS', 'NDN','NTP','PGS', 'PLC','PVC', 'PVI',
#                  'PVS', 'S99','SHB', 'SHS', 'TV2', 'VC3', 'VCG','VCS', 'VGC']
#    
    
      
#    symbolsHNX = ['ACB', 'BCC', 'CEO', 'DBC', 'DCS', 'HHG', 'HUT',
#                  'LAS',  'MBS', 'NDN', 'PGS', 'PVC', 'PVI',
#                  'PVS', 'S99','SHB', 'SHS', 'VC3', 'VCG','VCS', 'VGC']
#    
    symbolsHNX = ['APS', 'ALV', 'C69', 'TNG', 'BVS', 'PVX', "KDM", "ASA", "HKB", "HVA", 'NVB', "KLF", 'KVC', "VE9", 
                  'ACB', 'BCC', 'CVN', 'CEO', 'DBC', 'DCS', 'DST','HHG', 'HUT', 'SD9', 'HLD', 'NSH', 'DPS','DS3',
                  'LAS',  'MBS', 'NDN', 'PGS', 'PVC', 'PVI',  'MST', 'PHC', 'PVE', 'PVG', 'PVB',
                  'PVS', 'S99','SHB', 'SHS', 'TTB','VC3', 'VCG','VCS', 'VGC','VMC','VIX', 'TVC',  'TIG', 'SPP',
                  'VIG','VKC']
    
#    symbols = ["VCG", "VCB", "VSC", "FCN"]  # list of symbols
    end_date = "2018-4-13"
    start_date = "2018-2-1"
    dates = pd.date_range(start_date, end_date)  # date range as index
    df_data = get_data(symbolsHNX, dates, benchmark ='^HASTC')  # get data for each symbol

    df_volume = get_data(symbolsHNX, dates, benchmark = None, colname = '<Volume>')  # get data for each symbol
    df_high = get_data(symbolsHNX, dates, benchmark = None, colname = '<High>')
    df_low = get_data(symbolsHNX, dates, benchmark = None, colname = '<Low>')
    
    
    vol_mean = pd.Series(df_volume.mean(),name = 'Volume')
    max_high = pd.Series(df_high.max(), name = 'MaxHigh')
    min_low = pd.Series(df_low.min(), name = 'MinLow')
    cpm = pd.Series(max_high/min_low, name = 'CPM')
    # Fill missing values
    fill_missing_values(df_data)


    
    # Assess the portfolio
    allocations, cr, adr, sddr, sr  = optimize_portfolio(sd = start_date, ed = end_date,
        syms = symbolsHNX, benchmark = '^HASTC', gen_plot = True)

     # Print statistics
    print ("Start Date:", start_date)
    print ("End Date:", end_date)
    print ("Symbols:", symbolsHNX)
    print ("Optimal allocations:", allocations)
    print ("Sharpe Ratio:", sr)
    print ("Volatility (stdev of daily returns):", sddr)
    print ("Average Daily Return:", adr)
    print ("Cumulative Return:", cr)
    
    investment = 50000000
    df_result = pd.DataFrame(index = symbolsHNX)  
    df_result['Opt allocs'] = allocations
    df_result['Cash'] = allocations * investment
    df_result['Volume'] = vol_mean
    df_result['Close'] = df_data[symbolsHNX].iloc[-1,:].values
#    df_result['MaxH'] = max_high
#    df_result['MinL'] = min_low
    df_result['CPM'] = cpm
    
#    ticker = 'ACB'
#    
#    trading = ninja_trading(ticker, start_date, end_date)
#    plot_ninja_trading(ticker, trading)
    
#    ticker = 'ACB'
#    
#    trading = hedgefund_trading(ticker, start_date, end_date)
#    plot_hedgefund_trading(ticker, trading)
    

    
    return df_result, df_data


    
if __name__ == "__main__":
#    symbols = get_csv_data(source = "cp68")
#    symbols = get_csv_data()
#    symbols = get_stocks_highcpm(download = False, source ="cp68")
    
#    symbols =  ['FTS', 'PVI', 'VNE']

#    analysis_trading(symbols, start = "2017-3-1" , end = "2018-4-11", update = False, source = "cp68")


    
#    VNI_result, VNI_data  = test_runVNINDEX()
#    HNX_result, HNX_data = test_runHNX()
#    UPCOM_result, UPCOM_data = test_runUPCOM()
#    

#    ticker = 'VGC'    
#
#    end_date = "2018-4-5"
#    start_date = "2016-4-5"
###    bollingerbands = bollinger_bands(ticker, start_date, end_date, realtime = False, source = "cp68")
##    
##    hedgefund = hedgefund_trading(ticker, start_date, end_date, realtime = False, source ="cp68")    
##    plot_hedgefund_trading(ticker, hedgefund, realtime = False,  source ="cp68")
###    
###    shortsell = short_selling(ticker, start_date, end_date, realtime = False, source ="ssi")    
###    plot_shortselling_trading(ticker, shortsell, realtime = False,  source ="ssi")
###    
###
###    
#    ninja = ninja_trading(ticker, start_date, end_date, realtime = False,  source ="cp68")    
#    plot_ninja_trading(ticker, ninja, realtime = False,  source ="cp68")
    
#    plot_trading_weekly(ticker, hedgefund, realtime = False, source = "ssi")
#    
#    investment_stocks = ['CII', 'HPG', 'NBB', 'STB', 'PAN', 'VND' ]
    

    analysis_stocks(start = "2017-3-26" , end = "2018-4-13", update = False,  source ="cp68")
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
    
    symbolsHNX = ['APS', 'ALV', 'C69', 'TNG', 'BVS', 'PVX', "KDM", "ASA", "HKB", "HVA", 'NVB', "KLF", 'KVC', "VE9", 
                  'ACB', 'BCC', 'CVN', 'CEO', 'DBC', 'DCS', 'DST','HHG', 'HUT', 'SD9', 'HLD', 'NSH', 'DPS','DS3',
                  'LAS',  'MBS', 'NDN', 'PGS', 'PVC', 'PVI',  'MST', 'PHC', 'PVE', 'PVG', 'PVB',
                  'PVS', 'S99','SHB', 'SHS', 'TTB','VC3', 'VCG','VCS', 'VGC','VMC','VIX', 'TVC',  'TIG', 'SPP',
                  'VIG','VKC']

    symbolsUPCOM = ['TOP', 'TBD', 'LPB', 'QNS', 'RCC', 'ATB', 'ART',  'ACV', "SBS", "SWC", "NTC","DVN", 
                   'HVN', 'HPI','IDC',  'MSR', 'PXL', 'VGT','TVN','TVB','TIS','VIB']

#    ALLOC_opt = rebalancing_porfolio(symbols = symbolsUPCOM, bench = '^UPCOM')
    
#    investing = ['NVB', 'MBS', 'FPT', 'TVN', 'VIX']
#    predict_stocks(investing, start ="2010-3-18", end = "2018-4-13")
    
#    tickers = pd.Series(symbols)
    