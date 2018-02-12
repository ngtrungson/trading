# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 08:50:50 2018

@author: sonng
"""
import matplotlib.pyplot as plt
import pandas as pd
from finance_util import get_data
################################# Orders and market simulation ################################################

def createOrderDf(order, symbol = 'SHB', unit = 200):
    """
    @summary: Takes an order, a filename, and a base directory and writes to a dataframe
    @param order: Series, consists of the values -2, -1, 0, 1, 2 denoting 
                  buying/selling of 200 or 400 shares or just holding
    
    @param symbol: string, the stock to trade
    @param unit: integer, number of stocks to trade per unit in the order Series
    @returns nothing but writes the file on disk
    """
    df_orders = pd.DataFrame(columns=('Symbol', 'Order', 'Shares'))
        
    for ind, val in order.iteritems(): # iterator over the timestamp indices 
                                       # and values in 'order'
        if val == 1:            
#            df_orders.loc[ind] = '{},{},BUY,{}'.format(ind.date(), symbol, unit)
            df_orders.loc[ind] = pd.Series({'Symbol':symbol, 'Order':'BUY', 'Shares':unit})
        elif val == -1:
#            df_orders.loc[ind] = '{},{},SELL,{}'.format(ind.date(), symbol, unit)
            df_orders.loc[ind] = pd.Series({'Symbol':symbol, 'Order':'SELL', 'Shares':unit})
        elif val == 2:
#            df_orders.loc[ind] = '{},{},BUY,{}'.format(ind.date(), symbol, 2 * unit)
            df_orders.loc[ind] = pd.Series({'Symbol':symbol, 'Order':'BUY', 'Shares': 2*unit})
        elif val == -2:
#            df_orders.loc[ind] ='{},{},SELL,{}'.format(ind.date(), symbol, 2 * unit)
            df_orders.loc[ind] = pd.Series({'Symbol':symbol, 'Order':'SELL', 'Shares':2*unit})

    return df_orders
 
def compute_portvals_margin_vn(start, end, orders, start_val = 100000, commission = 0.0015, tax = 0.001, impact = 0.05):
    """Compute daily portfolio value given a dataframe of orders.

    Parameters
    ----------
        start_date: first date to track
        end_date: last date to track
        df:  read orders from dataframe
        start_val: total starting cash available

    Returns
    -------
        portvals: portfolio value for each trading day from start_date to end_date (inclusive)
    """
#    if isinstance(df, pd.DataFrame):
    
    if isinstance(orders, str):
        df = pd.read_csv(orders, index_col='Date', parse_dates=True)
    else:
        df = orders
    
    # Get the first date and last date after reading file
    start_date = df.iloc[0].name
    end_date = df.iloc[-1].name
    print(" Start trading date :" , start_date)
    print(" End trading date :" , end_date)
    
    dates = pd.date_range(start, end)
    if (dates[0] > start_date) or (end_date > dates[-1]):
        raise ValueError(" Error on start date and end date. Must be in larger period as in order file! Check again!")
        
#    symbols = []
#    for i, row in df.iterrows():
#        if row['Symbol'] not in symbols:
#            symbols.append(row['Symbol'])

    symbols = list(df['Symbol'].unique() ) #ndarray to list of symbols in order
    prices_symbol = get_data(symbols, pd.date_range(start, end))

    for symbol in symbols:
        prices_symbol[symbol + ' Shares'] = pd.Series(0, index=prices_symbol.index)
    prices_symbol['Port Val'] = pd.Series(start_val, index=prices_symbol.index)
    prices_symbol['Cash'] = pd.Series(start_val, index=prices_symbol.index)

    for i, row in df.iterrows():
        symbol = row['Symbol']
        if row['Order'] == 'BUY':
            cash_out = (prices_symbol.loc[i, symbol] * (1+impact) * row['Shares'])
            current_cash = prices_symbol.loc[i, 'Cash']
#            print(current_cash)
            if (2*current_cash < cash_out):
                print("Over leverage (margin) 50%. Cannot BUY all shares. ")
                pass
            else:
                if (current_cash < cash_out):
                    print(" Using leverage margin 50%")
                prices_symbol.loc[i:, symbol + ' Shares'] = prices_symbol.loc[i:, symbol + ' Shares'] + row['Shares']
               
                prices_symbol.loc[i:, 'Cash'] -= cash_out
                prices_symbol.loc[i:, 'Cash'] -= commission* cash_out
        if row['Order'] == 'SELL':
            current_shares = prices_symbol.loc[i, symbol + ' Shares']
            if (row['Shares'] > current_shares):
                print(" Not enough shares to SELL!", current_shares, row['Shares'])
                pass
            else:
                prices_symbol.loc[i:, symbol + ' Shares'] = prices_symbol.loc[i:, symbol + ' Shares'] - row['Shares']
                prices_symbol.loc[i:, 'Cash'] += cash_out
                prices_symbol.loc[i:, 'Cash'] -= (commission + tax)* cash_out
    for i, row in prices_symbol.iterrows():
        shares_val = 0
        for symbol in symbols:
            shares_val += prices_symbol.loc[i, symbol + ' Shares'] * row[symbol]
#            print(" Shares values :",  shares_val)
        prices_symbol.loc[i, 'Port Val'] = prices_symbol.loc[i, 'Cash'] + shares_val

    return prices_symbol
    
####============================================================================##