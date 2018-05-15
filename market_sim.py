# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 08:50:50 2018

@author: sonng
"""
import matplotlib.pyplot as plt
import pandas as pd
from finance_util import get_data
import os
import datetime as dt
from finance_util import plot_normalized_data, get_data_trading
import numpy as np
from plot_strategy import plotVline
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
 
def compute_portvals_margin_vn(start, end, orders, start_val = 100000, commission = 0.0015, tax = 0.001, impact = 0.0, market = "^VNINDEX"):
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
    
    prices_symbol = get_data(symbols, pd.date_range(start, end), benchmark = market)

    for symbol in symbols:
        prices_symbol[symbol + ' Shares'] = pd.Series(0, index=prices_symbol.index)
    prices_symbol['Port Val'] = pd.Series(start_val, index=prices_symbol.index)
    prices_symbol['Cash'] = pd.Series(start_val, index=prices_symbol.index)

#    return prices_symbol

    for i, row in df.iterrows():
        symbol = row['Symbol']
        if row['Order'] == 'BUY':
#            print(symbol)
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
def tradingStrategy(signal, holdTime = 21):
    """
    @Summary: Creates an order from a trading signal using 1 of 2 possible strategies
    @param signal: Series, consists of the values -1, 0, 1 denoting sell, hold
                   or buy
    @param holdTime: int, holding period after a transaction
    @returns order: Series, consists of the values -2, -1, 0, 1, 2 denoting 
                  selling/buying of 200 or 400 shares (depending on the 
                  strategy selected by commenting out below), or just holding    """
       
    numDays = signal.shape[0]
    day = 0; 
    order = signal * 0 # initialize a Series of zeros with the same date indices as signal
    currOrder = 0 # current order status, -1 (short), 0 (no position) or 1 (long)
    while day < numDays:
########## +/- 200 shares per transaction with 0, 200, -200 allowed positions
########## order can take values of  0, 1, -1 corresponding to 0, +/-200 shares
        if (currOrder < 1) and (signal[day] == 1): # current order status is
                                    # not long and signal to buy is given
            order[day] = 1 # buy 200
            currOrder += order[day]
            day += holdTime # after buying wait for hold period
        elif (currOrder > -1) and (signal[day] == -1): # current order status
                                    # is not short and signal to sell is given
            order[day] = -1 # sell 200
            currOrder += order[day]
            day += holdTime # after selling wait for hold period
        else:
            day += 1 # END OF +/- 200 TRADING STRATEGY 1

######### +/- 200 or +/- 400 shares per transaction with 0, 200, -200 allowed positions
######### order can take values of  0, 1, -1, 2, -2 corresponding to 0, +/-200, +/-400 shares
#        if signal[day] != 0: # if signal is 1 or -1
#    # if currOrder=0, order=signal. If currOrder = 1 or -1, order is 0, 2 or -2
#            order[day] = signal[day] - currOrder
#            currOrder += order[day]
#            if order[day] == 0: # if no order executed, go to next day
#                day += 1
#            else: # if order = -2, -1, 1, 2
#                day += holdTime # hold time
#        else: # if signal = 0, go to next day
#            day += 1 # END OF +/- 200 or +/- 400 TRADING STRATEGY 2

        if day >= numDays: # if we reach the end of the trading period
                           # redeem all outstanding positions
            if currOrder == 1:
                order[-1] = -1
            elif currOrder == -1:
                order[-1] = 1
    

    return order
    
def bestPossibleStrategy(data, column= 'Close'):
    """
    @Summmary: Evaluate the maximum possible return with a given stock looking 
               into the future and with a restriction of +/- 200 shares per 
               transaction with 0, 200 (represented by 1), -200 (represented 
               by -1) being the only allowed positions (hold, buy 200 and sell 200)
    @param data: Series, contains adj. close prices of the stock with date indices
    @returns order: Series, consists of the values -1, 0, 1, denoting 
                  selling/buying of 200 shares or just holding
    """
    
    
    nextDayReturn = data[column].ix[:-1] / data[column].values[1:] - 1 # calculate today's price
     # relative to tomorrow's price. This is to decide whether to buy/sell today
    nextDayReturn = nextDayReturn.append(data[column][-1:]) # restore the last date/value
                                   # row which was removed by the previous step
    nextDayReturn[-1] = np.nan # The value of the last row is a NaN since we do
                               # not know the next day's price
    dailyOrder = -1 * nextDayReturn.apply(np.sign) # find the sign and invert it
     # In dailyOrder, 1 means buy today, and -1 means sell today
    order = dailyOrder.diff(periods = 1) / 2 # pick out only where 1 changes to
     # -1 and vice-versa while eliminating consecutive 1s and -1s. Division by
     # 2 needed since we are constrained to buying and selling only 200 shares at a time.
    order[0] = dailyOrder[0] # restore the first date/value row removed by the
     # previous differentiation operation

    # on the last day, close any open positions
    if order.sum() == -1:
        order[-1] = 1
    elif order.sum() == 1:
        order[-1] = -1
            
    return order

def getCrossIndicator(data, column = 'Close', crossWindow = 25):
    """
    @Summary: My idea was to trigger buy/sell signals at the minima/maxima of
              a slowly oscillating time-series where it equals and crosses 
              its moving average but with an opposite sign of slope
    @param data: Series, contains prices with date indices
    @crossWindow: int, look-back window for simple moving average
    @returns Series with values ~ 0.5/-0.5 (buy/sell) on days where the 
            crossover occurs at the valleys/peaks of the  slow oscillations. On
            days where the slopes of the time-series and its sma have the same
            sign, the value is set to 0. On days where the slopes have opposite
            sign but a crossover does not occur, values between -1 and 1 are 
            returned
    """
    
    sma = data[column].rolling(window = crossWindow).mean()
    value = 1 / (1 + data[column] / sma) # ranges from 0 to 1,
                                                    # usually around 0.5
    # set sign = 1 (buy) for prices going up in time and sma going down.
    # set sign = -1 (sell) for the opposite situation.
    sign = ( 1 * ( (data[column].diff(periods = crossWindow) > 0) & \
    (sma.diff(periods = crossWindow) < 0) ) | \
                   -1 * ( (data[column].diff(periods = crossWindow) < 0) & \
    (sma.diff(periods = crossWindow) > 0) ) )
    indicator = sign * value
    indicator.name = 'crossIndicator' # set column name
    return indicator

def getMomIndicator(data, column = 'Close', momWindow = 10):
    """
    @Summary: Calculate momentum indicator = ratio of price with respect 
              to price momWindow number of days back - 1
    @param data: Series, contains prices with date indices
    @param momWindow: int, number of days to look back
    @returns Series of the same size as input data
    """
        
    diff = data[column].diff(periods = momWindow) # difference of prices wrt price
                                          # momWindow number of days back
    # divide above difference by price momWindow number of days back
    diff.ix[momWindow:] = diff.ix[momWindow:] / data[column].values[:-momWindow]
    diff.name = 'momIndicator' # set column name
    return diff


def getSmaIndicator(data, column = 'Close', smaWindow = 60):
    """
    @Summary:  wrapper method for getSma to return price/sma indicator
    @param data: Series, contains price with date indices
    @returns indicator: Series, contains price/sma indicator values with
                        date indices
    @returns smaWindow: int, window used in this method
    """
    
    
    smaWindow = 60# 60, optimized value of 60 on manual trading strategy 1
    sma = data[column].rolling(window = smaWindow).mean()
    indicator = data[column] / sma - 1 # indicator is a Series
    indicator.name = 'smaIndicator' # set column name
    return indicator
    
def getBbIndicator(data, column = 'Close', bbWindow = 20):
    """
    @Summary: Calculate Bollinger band indicator
    @param data: Series, contains price with date indices
    @param bbWindow: int, number of days to look back to calculate the moving
                     average and moving standard deviation
    @returns Series of the same size as input data
    """
    MEAN = data[column].rolling(window = bbWindow).mean()
    STD = data[column].rolling(window = bbWindow).std()
    indicator = (data[column] - MEAN) / ( 2 * STD)
    indicator.name = 'bbIndicator' # set column name
    return indicator

def standardize(data):
    """
    @Summary: Normalize by substracting mean and ividing by standard deviation
    @param data: DataFrame, Series, or ndarray
    @returns standardized data
    """
       
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


def testcode():
    
    end_date = "2018-2-21"
    start_date = "2017-2-2"
    dates = pd.date_range(start_date, end_date)
    symbols = ['SHB']
    investment = 1000000
    ticker = 'SHB'
    data = get_data(symbols, dates, benchmark = "^HASTC")
#    data = defineData(start_date, end_date, symbols)
    
#    return data
    plot_normalized_data(data)
     # get AAPL between the in-sample dates set as default
    data = get_data_trading(ticker, start_date, end_date)
    
    holdTime = 21 # in days
    smaWindow = 50
    smaIndicator  = getSmaIndicator(data, smaWindow = smaWindow)
    smaThreshold = 0.012 #0.012 # optimized value on manual trading strategy 1
    # generate a buy signal (1) if price falls significantly below sma
    # generate a sell signal (-1) if prices rises significantly above sma
    smaSignal = 1 * (smaIndicator < -smaThreshold)  +  \
            -1 * (smaIndicator > smaThreshold)
    momWindow = 10
    momIndicator = getMomIndicator(data, momWindow = momWindow)
    momThreshold = 0.06 #0.055 # optimized value on manual trading strategy 1
    # generate a buy/sell signal if momentum is greatly positive/negative
    momSignal = -1 * (momIndicator < -momThreshold)  +  \
            1 * (momIndicator > momThreshold)
            
    bbWindow = 10#48 NOT OPTIMIZED
    bbIndicator = getBbIndicator(data, bbWindow = bbWindow)
    bbThreshold = 0#0.2 NOT OPTIMIZED
    # generate a buy/sell signal if indicator is below/above the lower/upper BB
    # and the indicator is rising/falling significantly
    bbSignal = -1 * ((bbIndicator > 1) & \
                     (standardize(data['Close']).diff(1) < -bbThreshold)) + \
                 1 * ((bbIndicator < -1) & \
                     (standardize(data['Close']).diff(1) > bbThreshold))
    crossWindow = 18             
    crossIndicator = getCrossIndicator(data, crossWindow = crossWindow)
    crossThreshold = 0.08 #0.08 # optimized value on manual trading strategy 1
    # generate a buy/sell signal if indicator is close to 0.5/-0.5
    crossSignal = 1 * ( (crossIndicator - 0.5).abs() < crossThreshold) + \
            -1 * ( (crossIndicator + 0.5).abs() < crossThreshold )

    # Combine individual signals. bbSignal is neglected here since including it
    # with the other signals did not result in label-free trading using strategy 1
    signal = 1 * ( (smaSignal == 1) & (momSignal ==1 ) & (crossSignal == 1) ) \
        + -1 * ( (smaSignal == -1) & (momSignal == -1) & (crossSignal == -1) )
    
    order = tradingStrategy(signal, holdTime)    
    df_orders = createOrderDf(order)  
    prices_symbol = compute_portvals_margin_vn(start_date, end_date, df_orders, start_val = investment, market = "^HASTC")
    
#    return order

    portVals = prices_symbol['Port Val']
#    print('Cumulative return [%]: ', round(cumReturn * 100, 4) )
    
    order_opt = bestPossibleStrategy(data)    
    df_orders_opt = createOrderDf(order_opt)  
    prices_symbol_opt = compute_portvals_margin_vn(start_date, end_date, df_orders_opt, start_val = investment, \
                                         market = "^HASTC")
    


    portVals_opt = prices_symbol_opt['Port Val']
    
    fig = plt.figure(figsize = (10,10))
    ax1 = fig.add_subplot(3,1,1)
#    ax1 = plt.subplot(311)
    plt.plot(data['Close']/data['Close'].iloc[0], label = 'benchmark (market)', color = 'k')
    plt.plot(portVals / portVals[0], label = 'rule-based')
    plt.plot(portVals_opt / portVals_opt[0], label = 'optimal')
#    plt.xticks(rotation=30)
    plotVline(order)
    plt.title('rule-based with sma + mom + crossover indicators')
    lg = plt.legend(loc = 'best')
    lg.draw_frame(False)
    plt.ylabel('normalized')

#    ax2 = plt.subplot(312)
    ax2 = fig.add_subplot(3,1,2)
    plt.plot(smaSignal/2, label = 'sma')
    plt.plot(momSignal/1.3,'.', label = 'mom')
    plt.plot(crossSignal/1.1,'.', label = 'crossover')
    plt.plot(signal, label = 'overall signal')
#    plt.xticks(rotation=30)
    plt.ylabel('indicator signals [a.u.]')
    lg = plt.legend(loc = 'center right')
    lg.draw_frame(False)
    
#    plt.subplot(313)
    ax3 = fig.add_subplot(3,1,3)
    plt.scatter(momIndicator[signal==0], crossIndicator[signal==0], \
                           color = 'k', label = 'hold')
    plt.scatter(momIndicator[signal==1], crossIndicator[signal==1], \
                           color = 'g', label = 'buy')
    plt.scatter(momIndicator[signal==-1], crossIndicator[signal==-1], \
                           color = 'r', label = 'sell')
    lg = plt.legend(loc = 'best')
    lg.draw_frame(True)
    plt.xlabel('Momentum Indicator')
    plt.ylabel('Crossover Indicator')
    
    plt.subplots_adjust(hspace=0)
#    plt.setp([a.get_xticklabels() for a in plt.axes[:-1]], visible=False)
    return portVals_opt
    
if __name__ == '__main__':
#    data = testcode()
    portVals_opt  = testcode()    

#    orders_file = os.path.join("orders", "ordersHNX.csv")
#    investment = 50000
#    end_date = "2018-2-22"
#    start_date = "2017-2-2"
##    start_date = dt.datetime(2017,2,2)
##    end_date = dt.datetime(2018,2,22)
#    prices_symbol = compute_portvals_margin_vn(start = start_date, end = end_date, orders = orders_file, start_val = investment, market = "^HASTC")
#    