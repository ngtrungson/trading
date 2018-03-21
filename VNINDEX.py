# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:35:57 2017

@author: sonng
"""
import pandas as pd
from finance_util import get_data, fill_missing_values, optimize_portfolio, \
                         get_data_from_cophieu68_openwebsite, get_data_from_SSI_website
from strategy import ninja_trading, hedgefund_trading, bollinger_bands
from plot_strategy import plot_hedgefund_trading, plot_ninja_trading, plot_trading_weekly
from machine_learning import price_predictions


def get_stocks_highcpm(download = True, source = "ssi"):

    data = pd.read_csv('fundemental_stocks_all.csv', parse_dates=True, index_col=0)
    df = data.query("MeanVol_10W > 100000")
    df = df.query("FVQ > 0")
    df = df.query("CPM > 1.4")
    
    tickers  = df.index
    
    if download:
        if source == "cp68":
            get_data_from_cophieu68_openwebsite(tickers)
        else:
           get_data_from_SSI_website(tickers) 
    
    
    return tickers
    
def analysis_stocks(start, end, update = False, source = "ssi"):
    
    symbolsHNX = ['APS','ALV', 'TNG','BVS','PVX',"KDM","ASA","HKB","HVA","KLF", "VE9", 
                  'ACB','BCC','CEO','DBC','DCS','HHG','HUT',
                  'LAS', 'MBS', 'NDN', 'PGS', 'PVC', 'PVI',
                  'PVS', 'S99','SHB', 'SHS', 'VC3', 'VCG','VCS', 'VGC']
    symbolsVNI = [ 'BCG','ATG','ASP','APG', 'ANV', 'APC', 
                  "ASM", "BFC", "BID", "BMI", "BMP", "BVH",
                  "CII", "CTD", "CAV", "CMG", "CSM", "CSV", "CTG",  
               "DCM","DHG", "DIG", "DLG", "DPM","DPR", "DRH",  "DQC", "DRC", "DXG", 
               "ELC", "EVE","FCN","FIT","FLC","FPT", "GAS", "GMD", "GTN", 
               "HAG", "HHS", "HNG", "HQC", "HT1", "HVG",
               "HSG", "HDG", "HCM", "HPG", "HBC", 
               "IJC", "ITA", "KBC", "KSB",  "KDH", "KDC", 
               "MBB", "MSN", "MWG", "NKG", "NLG", "NT2", "NVL", "NBB",
                "PVT","PVD","PHR","PGI","PDR","PTB", "PNJ",  "PC1",   "PLX", "PPC", "PAC",
                "QCG", "REE",  "SAM","SJD","SJS","STB","STG","SKG",  "SSI", "SBT", "SAB", 
                    "VSH","VNM", "VHC", "VIC", "VCB", "VSC", "VJC", "VNS" ,
                    'ITC','LSS','VOS', 'OGC', 'PME', 'PAN','TCH', 'GEX','VCI',
                    'TDC','TCM', 'VNE','KSA', 'SHN', 'AAA','SCR', 'AGR',
                    'EIB','BHN','VPB','VRE','ROS',"VND", "HDB","NVT","VHG", "SMC", "C32","CVT"]
    symbolsUPCOM = ["SBS", "SWC", "NTC","DVN", 'HVN', 'IDC']
    
    
    symbolother = ['CCL','CHP', 
                   'CTI','CTS','CVN','DGW','DHA','DHC',
                   'FMC','FTS','HAI','IDI',
                   'KSD','KVC','LCG','LDG','LGL', 'MSR','NS3',
                   'NVB','PFL','PHC',
                   'POM','PV2','PVE','PVG','PVV','PXL','QBS',
                   'SD9','SDI','SFG','SPI',
                   'TDH','TIS','TNT','TTB','TTF',
                   'TVB','TVN','TVS','VDS','VGT','VIB',
                  'VIG','VIP','VIX','VMC','VNG','VPH']
    
    high_cpm = ['ACV', 'ALV', 'AMD', 'ANV', 'APC', 'ART', 'ATB', 'BCC', 'C47', 'C69',
       'CCL', 'CDO', 'CMG', 'CVN', 'DHM', 'DIG', 'DST', 'EVG', 'FIT', 'HAI',
       'HAR', 'HCM', 'HID', 'HII', 'HKB', 'HPG', 'HPI', 'HTT', 'HVA', 'HVN',
       'IDI', 'KDM', 'KHB', 'KLF', 'KSA', 'LDG', 'MBB', 'MBS', 'MSR', 'MST',
       'NHP', 'NS3', 'NTB', 'NVB', 'NVT', 'OCH', 'OGC', 'PDR', 'PIV', 'PND',
       'PPI', 'PVO', 'QBS', 'QCG', 'ROS', 'SBS', 'SDI', 'SHB', 'SHS', 'SPI',
       'TSC', 'TVB', 'VHG', 'VIG', 'VJC', 'VKC', 'VND', 'VOS']
    
    symbols = symbolsVNI + symbolsHNX + symbolsUPCOM + symbolother + high_cpm
 
#    symbols = symbolsVNI 
    
    tickers  = pd.unique(symbols).tolist()
    tickers = sorted(tickers)
    
    for ticker in tickers:
#        print("Analysing ...", ticker)

        ninja_trading(ticker, start, end, realtime = update, source = source)
        hedgefund_trading(ticker, start, end, realtime = update, source = source)
        bollinger_bands(ticker, start, end, realtime = update, source = source)

    
    
def analysis_trading(tickers, start, end, update = False, source = "cp68"):
    for ticker in tickers:
#        print(" Analysing ..." , ticker)
        try:
            ninja_trading(ticker, start, end, realtime = update, source = source)
            hedgefund_trading(ticker, start, end, realtime = update, source = source)
            bollinger_bands(ticker, start, end, realtime = update, source = source)
        except Exception as e:
            print (e)
            print("Error in reading symbol: ", ticker)
            pass
               
def get_csv_data(source = "cp68"):
    benchmark = ["^VNINDEX", "^HASTC", "^UPCOM"]
    symbolsHNX = ['TNG', 'BVS', 'PVX', "KDM", "ASA", "HKB", "HVA", "KLF", "VE9", 
                  'ACB', 'BCC', 'CEO', 'DBC', 'DCS', 'HHG', 'HUT',
                  'LAS',  'MBS', 'NDN', 'PGS', 'PVC', 'PVI',
                  'PVS', 'S99','SHB', 'SHS', 'VC3', 'VCG','VCS', 'VGC']
    symbolsVNI = [ "ASM", "BFC", "BID", "BMI", "BMP", "BVH",
                  "CII", "CTD", "CAV", "CMG", "CSM", "CSV", "CTG",  
               "DCM","DHG", "DIG", "DLG", "DPM","DPR", "DRH",  "DQC", "DRC", "DXG", 
               "ELC", "EVE","FCN","FIT","FLC","FPT", "GAS", "GMD", "GTN", 
               "HAG", "HHS", "HNG", "HQC", "HT1", "HVG",
               "HSG", "HDG", "HCM", "HPG", "HBC", 
               "IJC", "ITA", "KBC", "KSB",  "KDH", "KDC", 
               "MBB", "MSN", "MWG", "NKG", "NLG", "NT2", "NVL", "NBB",
                "PVT","PVD","PHR","PGI","PDR","PTB", "PNJ",  "PC1",   "PLX", "PPC", "PAC",
                "QCG", "REE",  "SAM","SJD","SJS","STB","STG","SKG",  "SSI", "SBT", "SAB", 
                    "VSH","VNM", "VHC", "VIC", "VCB", "VSC", "VJC", "VNS" ,
                    'ITC','LSS','VOS', 'OGC', 'PME', 'PAN','TCH', 'GEX','VCI',
                    'TDC','TCM', 'VNE','KSA', 'SHN', 'AAA','SCR', 'AGR',
                    'EIB','BHN','VPB','VRE','ROS',"VND", "HDB","NVT","VHG", "SMC", "C32","CVT"]
    symbolsUPCOM = ["SBS", "SWC", "NTC","DVN", 'HVN', 'IDC']
    
    symbolother = ['ALV','ANV','APC','APG','APS','ASP','ATG',
                   'BCG','CCL','CHP',
                   'CTI','CTS','CVN','DGW','DHA','DHC',
                   'FMC','FTS','HAI','IDI',
                   'KSD','KVC','LCG','LDG','LGL', 'MSR','NS3',
                   'NVB','PFL','PHC',
                   'POM','PV2','PVE','PVG','PVV','PXL','QBS',
                   'SD9','SDI','SFG','SPI',
                   'TDH','TIS','TNT','TTB','TTF',
                   'TVB','TVN','TVS','VDS','VGT','VIB',
                  'VIG','VIP','VIX','VMC','VNG','VPH']
    
    high_cpm = ['ACV', 'ALV', 'AMD', 'ANV', 'APC', 'ART', 'ATB', 'BCC', 'C47', 'C69',
       'CCL', 'CDO', 'CMG', 'CVN', 'DHM', 'DIG', 'DST', 'EVG', 'FIT', 'HAI',
       'HAR', 'HCM', 'HID', 'HII', 'HKB', 'HPG', 'HPI', 'HTT', 'HVA', 'HVN',
       'IDI', 'KDM', 'KHB', 'KLF', 'KSA', 'LDG', 'MBB', 'MBS', 'MSR', 'MST',
       'NHP', 'NS3', 'NTB', 'NVB', 'NVT', 'OCH', 'OGC', 'PDR', 'PIV', 'PND',
       'PPI', 'PVO', 'QBS', 'QCG', 'ROS', 'SBS', 'SDI', 'SHB', 'SHS', 'SPI',
       'TSC', 'TVB', 'VHG', 'VIG', 'VJC', 'VKC', 'VND', 'VOS']
    
#    symbols = symbolsVNI + symbolsHNX + symbolsUPCOM + symbolother + high_cpm
    symbols = benchmark + symbolsVNI + symbolsHNX + symbolsUPCOM + symbolother + high_cpm
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
    
    symbolsVNI = [ "ASM", "BFC", "BID", "BMI", "BMP", "BVH",
                  "CII", "CTD", "CAV", "CMG", "CSM", "CSV", "CTG",  
               "DCM","DHG", "DIG", "DLG", "DPM","DPR", "DRH",  "DQC", "DRC", "DXG", 
               "ELC", "EVE","FCN","FIT","FLC","FPT", "GAS", "GMD", "GTN", 
               "HAG", "HHS", "HNG", "HQC", "HT1", "HVG",
               "HSG", "HDG", "HCM", "HPG", "HBC", 
               "IJC", "ITA", "KBC", "KSB",  "KDH", "KDC", 
               "MBB", "MSN", "MWG", "NKG", "NLG", "NT2", "NVL", "NBB",
                "PVT","PVD","PHR","PGI","PDR","PTB", "PNJ",  "PC1",   "PLX", "PPC", "PAC",
                "QCG", "REE",  "SAM","SJD","SJS","STB","STG","SKG",  "SSI", "SBT", "SAB", 
                    "VSH","VNM", "VHC", "VIC", "VCB", "VSC", "VJC", "VNS" ,
                    'ITC','LSS','VOS', 'OGC', 'PME', 'PAN','TCH', 'GEX','VCI',
                    'TDC','TCM', 'VNE','KSA', 'SHN', 'AAA','SCR', 'AGR',
                    'EIB','BHN','VPB','VRE','ROS',"VND", "HDB","NVT","VHG", "SMC", "C32","CVT"]

#    update = False
#    # Read data
#    if update:
#        get_data_from_cophieu68_openwebsite(symbolsVNI)
    
#    symbols = ["VCG", "VCB", "VSC", "FCN"]  # list of symbols
    end_date = "2018-3-16"
    start_date = "2018-1-2"

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
    
    investment = 5000000
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

  
def test_run_HNX():
   
#    
#    symbolsHNX = ['ACB', 'BCC', 'BVS', 'CEO', 'DBC', 'DCS', 'DGC', 'HHG', 'HUT',
#                  'IDV', 'LAS', 'LHC','MAS', 'MBS', 'NDN','NTP','PGS', 'PLC','PVC', 'PVI',
#                  'PVS', 'S99','SHB', 'SHS', 'TV2', 'VC3', 'VCG','VCS', 'VGC']
#    
    
      
#    symbolsHNX = ['ACB', 'BCC', 'CEO', 'DBC', 'DCS', 'HHG', 'HUT',
#                  'LAS',  'MBS', 'NDN', 'PGS', 'PVC', 'PVI',
#                  'PVS', 'S99','SHB', 'SHS', 'VC3', 'VCG','VCS', 'VGC']
#    
    symbolsHNX = ['TNG', 'BVS', 'PVX', "KDM", "ASA", "HKB", "HVA", "KLF", "VE9", 
                  'ACB', 'BCC', 'CEO', 'DBC', 'DCS', 'HHG', 'HUT',
                  'LAS',  'MBS', 'NDN', 'PGS', 'PVC', 'PVI',
                  'PVS', 'S99','SHB', 'SHS', 'VC3', 'VCG','VCS', 'VGC']
    
#    symbols = ["VCG", "VCB", "VSC", "FCN"]  # list of symbols
    end_date = "2018-3-16"
    start_date = "2018-1-2"
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
    df_result['Allocs'] = allocations
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
    symbols = get_stocks_highcpm(download = False, source ="ssi")
    
#    symbols = symbols + ['EIB', 'TVN', 'DVN', 'FPT', 'VCB', 'PVS', 'SHB', 'HAR']

    analysis_trading(symbols, start = "2017-3-19" , end = "2018-3-21", update = False, source = "ssi")

    
#    VNI_result, VNI_data  = test_runVNINDEX()
#    HNX_result, HNX_data = test_run_HNX()
    

#    ticker = 'HAR'    
#
#    end_date = "2018-3-16"
#    start_date = "2017-3-1"
#    hedgefund = hedgefund_trading(ticker, start_date, end_date, realtime = False, source ="ssi")    
#    plot_hedgefund_trading(ticker, hedgefund, realtime = False,  source ="ssi")
##
##    
#    ninja = ninja_trading(ticker, start_date, end_date, realtime = False,  source ="ssi")    
#    plot_ninja_trading(ticker, ninja, realtime = False,  source ="ssi")
##    
#    plot_trading_weekly(ticker, hedgefund, realtime = False, source = "ssi")
#    
#    investment_stocks = ['CII', 'HPG', 'NBB', 'STB', 'PAN', 'VND' ]
    

#    analysis_stocks(start = "2017-3-14" , end = "2018-3-16", update = False,  source ="ssi")

    
#    investing = ['HAR', 'SHB', 'TVN', 'PVS', 'DVN', 'VIX']
#    predict_stocks(investing, start ="2010-3-18", end = "2018-3-18")
    
    tickers = pd.Series(symbols)
    