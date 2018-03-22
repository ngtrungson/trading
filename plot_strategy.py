# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 08:41:27 2018

@author: sonng
"""

import pandas as pd
import matplotlib.pyplot as plt

import pylab
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

import matplotlib.ticker as mticker



 
def plotVline(order):
    """
    @Summary: Plots vertical lines for buy and sell orders
    @param order: Series, consists of the values -2, -1, 0, 1, 2 denoting 
                  buying/selling of 200 or 400 shares or just holding
    @returns nothing
    """
       
    for date in order.index[order == 1]: # for dates corr. to buy 200 shares
        plt.axvline(date, color = 'g', linewidth=1, linestyle = '--')
        
    for date in order.index[order == -1]: # for dates corr. to sell 200 shares
        plt.axvline(date, color = 'r', linewidth=1, linestyle = '--')
    
    for date in order.index[order == 2]: # for dates corr. to buy 400 shares
        plt.axvline(date, color = 'g', linewidth=1,)
        
    for date in order.index[order == -2]: # for dates corr. to sell 400 shares
        plt.axvline(date, color = 'r', linewidth=1,)

 
def plot_ninja_trading(ticker, df, realtime = False, source ="cp68"):
   
    df_ohlc = df.copy()
    df_ohlc = df.reset_index()
        
    if realtime:
      df_ohlc = df_ohlc.rename(columns = {'index': 'Date'})  
    else:
        if source == "ssi":
            df_ohlc = df_ohlc.rename(columns = {'DATE': 'Date'})
        else:
            df_ohlc = df_ohlc.rename(columns = {'<DTYYYYMMDD>': 'Date'})


#Converting dates column to float values
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)


    
    fig = plt.figure(facecolor='w')
                     

    ax1 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4, facecolor='w')
    candlestick_ohlc(ax1, df_ohlc.values, width=.6, colorup='#53c156', colordown='#ff1717')
          
      
    ax1.plot(df_ohlc['Date'],df_ohlc['EMA3'].values, linewidth=1, label = 'EMA3')
    ax1.plot(df_ohlc['Date'],df_ohlc['EMA6'].values, linewidth=1,label = 'EMA6')
    ax1.plot(df_ohlc['Date'],df_ohlc['EMA18'].values, linewidth=1,label = 'EMA18')
    ax1.plot(df_ohlc['Date'],df_ohlc['EMA50'].values, linewidth=1, label = 'EMA50')
    ax1.plot(df_ohlc['Date'],df_ohlc['EMA100'].values, linewidth=1, label = 'EMA100')
    ax1.plot(df_ohlc['Date'],df_ohlc['EMA200'].values, linewidth=1, label = 'EMA200')
  
#    ax1.grid(True, color ='gray',linestyle = '--')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#    ax1.yaxis.label.set_color("w")
    ax1.spines['bottom'].set_color("#5998ff")
    ax1.spines['top'].set_color("#5998ff")
    ax1.spines['left'].set_color("#5998ff")
    ax1.spines['right'].set_color("#5998ff")
    ax1.tick_params(axis='y')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax1.tick_params(axis='x')
    plt.ylabel('Stock price')

    maLeg = plt.legend(loc=2, ncol=2, prop={'size':9},
               fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = pylab.gca().get_legend().get_texts()
    plt.xticks(rotation=30)
    orders = pd.Series(df_ohlc['Signal'].values, df_ohlc['Date'])
#    order_3_6_18 = pd.Series(df_ohlc['3_6_18_LONG'].values, df_ohlc['Date'])
    plotVline(orders)
#    plotVline(order_3_6_18)
    
#    
    ax0 = plt.subplot2grid((6,4), (0,0), sharex=ax1, rowspan=1, colspan=4, facecolor = 'w')
    
    mfiCol = 'blue'
    posCol = '#386d13'
    negCol = '#8f2020'
    
    ax0.plot(df_ohlc['Date'], df_ohlc['MFI'].values, mfiCol, linewidth=1)
    ax0.axhline(80, color=negCol)
    ax0.axhline(20, color=posCol)
    ax0.fill_between(df_ohlc['Date'], df_ohlc['MFI'].values, 80, where=(df_ohlc['MFI']>=80), facecolor=negCol, edgecolor=negCol, alpha=0.5)
    ax0.fill_between(df_ohlc['Date'], df_ohlc['MFI'].values, 20, where=(df_ohlc['MFI']<=20), facecolor=posCol, edgecolor=posCol, alpha=0.5)
    ax0.set_yticks([20,80])
#    ax0.yaxis.label.set_color("w")
    ax0.spines['bottom'].set_color("#5998ff")
    ax0.spines['top'].set_color("#5998ff")
    ax0.spines['left'].set_color("#5998ff")
    ax0.spines['right'].set_color("#5998ff")
    ax0.tick_params(axis='y')
    ax0.tick_params(axis='x')
    plt.ylabel('MFI')

    ax1v = ax1.twinx()
    ax1v.fill_between(df_ohlc['Date'],df['Volume'].values, facecolor='#00ffe8', alpha=.4)
   # make bar plots and color differently depending on up/down for the day
#    pos = df_ohlc['Open']- df_ohlc['Close'] <0
#    neg = df_ohlc['Open']- df_ohlc['Close'] >0
#    ax1v.bar(df_ohlc['Date'][pos],df_ohlc['Volume'][pos],color='green',width=1,align='center')
#    ax1v.bar(df_ohlc['Date'][neg],df_ohlc['Volume'][neg],color='red',width=1,align='center')
#    
#    #scale the x-axis tight
##    ax1v.set_xlim(min(dates),max(dates))
#    # the y-ticks for the bar were too dense, keep only every third one
#    yticks = ax1v.get_yticks()
#    ax1v.set_yticks(yticks[::3])
    

    
    
    ax1v.axes.yaxis.set_ticklabels([])
    ax1v.grid(False)
    ###Edit this to 3, so it's a bit larger
    ax1v.set_ylim(0, 3*df['Volume'].values.max())
    
    ax1v.spines['bottom'].set_color("#5998ff")
    ax1v.spines['top'].set_color("#5998ff")
    ax1v.spines['left'].set_color("#5998ff")
    ax1v.spines['right'].set_color("#5998ff")
    ax1v.tick_params(axis='x')
    ax1v.tick_params(axis='y')
    
    
    
    
    ax2 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4, facecolor='w')
    fillcolor = '#00ffe8'
#   
#    df['MACD_12_26'] = MACD
#    df['MACDSign9'] = MACDsign
#    df['MACDDiff'] = MACDdiff
    
    ax2.plot(df_ohlc['Date'], df['MACD_12_26'].values, color='green', lw=1)
    ax2.plot(df_ohlc['Date'], df['MACDSign12269'].values, color='red', lw=1)
#    ax2.fill_between(df_ohlc['Date'], df['MACDDiff'].values, 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)
    ax2.axhline(0, color = 'gray', linewidth=1, linestyle = '--')
#    ax2.grid(True, color ='gray', linestyle = '--')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax2.spines['bottom'].set_color("#5998ff")
    ax2.spines['top'].set_color("#5998ff")
    ax2.spines['left'].set_color("#5998ff")
    ax2.spines['right'].set_color("#5998ff")
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    plt.ylabel('MACD')
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(30)

    plt.suptitle(ticker.upper())
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
   
    plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
    plt.show()
#        fig.savefig('example.png',facecolor=fig.get_facecolor())

def plot_hedgefund_trading(ticker, df, realtime = False, source ="cp68"):
   
    df_ohlc = df.copy()
    df_ohlc = df.reset_index()
    if realtime:
      df_ohlc = df_ohlc.rename(columns = {'index': 'Date'})  
    else:
#        df_ohlc = df_ohlc.rename(columns = {'<DTYYYYMMDD>': 'Date'})
        if source == "ssi":
            df_ohlc = df_ohlc.rename(columns = {'DATE': 'Date'})
        else:
            df_ohlc = df_ohlc.rename(columns = {'<DTYYYYMMDD>': 'Date'})

#    return df_ohlc
    #Converting dates column to float values
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)


    
    fig = plt.figure(facecolor='w')
                     

    ax1 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4, facecolor='w')
    candlestick_ohlc(ax1, df_ohlc.values, width=.6, colorup='#53c156', colordown='#ff1717')         

    ax1.plot(df_ohlc['Date'],df_ohlc['EMA18'].values, linewidth=1,label = 'EMA18', color = 'blue')
  
  
#    ax1.grid(True, color ='gray',linestyle = '--')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#    ax1.yaxis.label.set_color("w")
    ax1.spines['bottom'].set_color("#5998ff")
    ax1.spines['top'].set_color("#5998ff")
    ax1.spines['left'].set_color("#5998ff")
    ax1.spines['right'].set_color("#5998ff")
    ax1.tick_params(axis='y')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax1.tick_params(axis='x')
    plt.ylabel('Stock price')

    maLeg = plt.legend(loc=2, ncol=2, prop={'size':9},
               fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = pylab.gca().get_legend().get_texts()
    plt.xticks(rotation=30)
    orders = pd.Series(df_ohlc['Signal'].values, df_ohlc['Date'])
#    order_3_6_18 = pd.Series(df_ohlc['3_6_18_LONG'].values, df_ohlc['Date'])
    plotVline(orders)
#    plotVline(order_3_6_1s8)
    
#    
    ax0 = plt.subplot2grid((6,4), (0,0), sharex=ax1, rowspan=1, colspan=4, facecolor = 'w')
    
    fillcolor = '#00ffe8'
   
    posCol = '#386d13'
    negCol = '#8f2020'
    
    ax0.plot(df_ohlc['Date'], df['MACD_3_6'].values, color='green', lw=1)
    ax0.plot(df_ohlc['Date'], df['MACDSign20'].values, color='red', lw=1)
#    ax0.fill_between(df_ohlc['Date'], df['MACDDiff3620'].values, 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)
    ax0.axhline(0, color = 'gray', linewidth=1, linestyle = '--')
#    ax0.yaxis.label.set_color("w")
    ax0.spines['bottom'].set_color("#5998ff")
    ax0.spines['top'].set_color("#5998ff")
    ax0.spines['left'].set_color("#5998ff")
    ax0.spines['right'].set_color("#5998ff")
    ax0.tick_params(axis='y')
    ax0.tick_params(axis='x')
    plt.ylabel('1st trend')

    ax1v = ax1.twinx()
    ax1v.fill_between(df_ohlc['Date'],df['Volume'].values, facecolor='#00ffe8', alpha=.4)
   # make bar plots and color differently depending on up/down for the day
#    pos = df_ohlc['Open']- df_ohlc['Close'] <0
#    neg = df_ohlc['Open']- df_ohlc['Close'] >0
#    ax1v.bar(df_ohlc['Date'][pos],df_ohlc['Volume'][pos],color='green',width=1,align='center')
#    ax1v.bar(df_ohlc['Date'][neg],df_ohlc['Volume'][neg],color='red',width=1,align='center')
#    
#    #scale the x-axis tight
##    ax1v.set_xlim(min(dates),max(dates))
#    # the y-ticks for the bar were too dense, keep only every third one
#    yticks = ax1v.get_yticks()
#    ax1v.set_yticks(yticks[::3])
    

    
    
    ax1v.axes.yaxis.set_ticklabels([])
    ax1v.grid(False)
    ###Edit this to 3, so it's a bit larger
    ax1v.set_ylim(0, 3*df['Volume'].values.max())
    
    ax1v.spines['bottom'].set_color("#5998ff")
    ax1v.spines['top'].set_color("#5998ff")
    ax1v.spines['left'].set_color("#5998ff")
    ax1v.spines['right'].set_color("#5998ff")
    ax1v.tick_params(axis='x')
    ax1v.tick_params(axis='y')
    
    
    
    
    ax2 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4, facecolor='w')
    fillcolor = '#00ffe8'
#   
#    df['MACD_12_26'] = MACD
#    df['MACDSign9'] = MACDsign
#    df['MACDDiff'] = MACDdiff
    
    ax2.plot(df_ohlc['Date'], df['MACD_50_100'].values, color='green', lw=1)
    ax2.plot(df_ohlc['Date'], df['MACDSign9'].values, color='red', lw=1)
#    ax2.fill_between(df_ohlc['Date'], df['MACDDiff501009'].values, 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)
    ax2.axhline(0, color = 'gray', linewidth=1, linestyle = '--')
#    ax2.grid(True, color ='gray', linestyle = '--')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax2.spines['bottom'].set_color("#5998ff")
    ax2.spines['top'].set_color("#5998ff")
    ax2.spines['left'].set_color("#5998ff")
    ax2.spines['right'].set_color("#5998ff")
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    plt.ylabel('2nd Trend')
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(30)

    plt.suptitle(ticker.upper())
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
   
    plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
    
 
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    candlestick_ohlc(ax, df_ohlc.values, width=.6, colorup='#53c156', colordown='#ff1717')         

    ax.plot(df_ohlc['Date'],df_ohlc['EMA18'].values, linewidth=1,label = 'EMA18', color = 'blue')
    ax.set_facecolor('w')
    maLeg = plt.legend(loc=2, ncol=2, prop={'size':9},
               fancybox=True, borderaxespad=0.)
  
    
#    
#    axv = fig2.add_subplot(2, 1, 2)   
#    axv.plot(df_ohlc['Date'], df['MACD_12_26'].values, color='green', lw=1)
#    axv.plot(df_ohlc['Date'], df['MACDSign9_1226'].values, color='red', lw=1)   
#    axv.axhline(0, color = 'gray', linewidth=1, linestyle = '--') 
#    axv.axes.yaxis.set_ticklabels([])
#    axv.grid(False)      
#    axv.spines['bottom'].set_color("#5998ff")
#    axv.spines['top'].set_color("#5998ff")
#    axv.spines['left'].set_color("#5998ff")
#    axv.spines['right'].set_color("#5998ff")
#    axv.tick_params(axis='x')
#    axv.tick_params(axis='y')
    
    
    
#    ax1.grid(True, color ='gray',linestyle = '--')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#    ax1.yaxis.label.set_color("w")
    ax.spines['bottom'].set_color("#5998ff")
    ax.spines['top'].set_color("#5998ff")
    ax.spines['left'].set_color("#5998ff")
    ax.spines['right'].set_color("#5998ff")
    ax.tick_params(axis='y')    
    ax.tick_params(axis='x')
    ax.autoscale_view()
    plt.ylabel('Stock price')
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(30)
    plt.title(ticker.upper() + " daily")
    
    
    
    
    
    plt.show()
def plot_trading_weekly(ticker, df, realtime = False,  source ="ssi"):
   
    df_ohlc = df['Close'].resample('5D').ohlc()   
    
    df_ohlc = df_ohlc.reset_index()
    if realtime:
      df_ohlc = df_ohlc.rename(columns = {'index': 'Date'})  
    else:
#        df_ohlc = df_ohlc.rename(columns = {'<DTYYYYMMDD>': 'Date'})
       if source == "ssi":
            df_ohlc = df_ohlc.rename(columns = {'DATE': 'Date'})
       else:
            df_ohlc = df_ohlc.rename(columns = {'<DTYYYYMMDD>': 'Date'})

        
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
    EMA18_week = df['EMA18'].resample('5D').mean()
    
    fig, ax = plt.subplots(facecolor='w')
    candlestick_ohlc(ax, df_ohlc.values, width=1.5, colorup='#53c156', colordown='#ff1717')         

    ax.plot(df_ohlc['Date'], EMA18_week.values, linewidth=1,label = 'EMA18', color = 'blue')
    
    maLeg = plt.legend(loc=2, ncol=2, prop={'size':9},
               fancybox=True, borderaxespad=0.)
    ax.set_facecolor('w')
#    ax1.grid(True, color ='gray',linestyle = '--')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#    ax1.yaxis.label.set_color("w")
    ax.spines['bottom'].set_color("#5998ff")
    ax.spines['top'].set_color("#5998ff")
    ax.spines['left'].set_color("#5998ff")
    ax.spines['right'].set_color("#5998ff")
    ax.tick_params(axis='y')    
    ax.tick_params(axis='x')
    ax.autoscale_view()
    plt.ylabel('Stock price weekly')
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(30)
    plt.title(ticker.upper() + " weekly")
    
    plt.show()
    
def plot_shortselling_trading(ticker, df, realtime = False, source ="cp68"):
   
    df_ohlc = df.copy()
    df_ohlc = df.reset_index()
    if realtime:
      df_ohlc = df_ohlc.rename(columns = {'index': 'Date'})  
    else:
#        df_ohlc = df_ohlc.rename(columns = {'<DTYYYYMMDD>': 'Date'})
        if source == "ssi":
            df_ohlc = df_ohlc.rename(columns = {'DATE': 'Date'})
        else:
            df_ohlc = df_ohlc.rename(columns = {'<DTYYYYMMDD>': 'Date'})

#    return df_ohlc
    #Converting dates column to float values
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)


    
    fig = plt.figure(facecolor='w')
                     

    ax1 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4, facecolor='w')
    candlestick_ohlc(ax1, df_ohlc.values, width=.6, colorup='#53c156', colordown='#ff1717')         

       
    ax1.plot(df_ohlc['Date'],df_ohlc['EMA3'].values, linewidth=1, label = 'EMA3')    
    ax1.plot(df_ohlc['Date'],df_ohlc['EMA18'].values, linewidth=1,label = 'EMA18')
    ax1.plot(df_ohlc['Date'],df_ohlc['EMA50'].values, linewidth=1, label = 'EMA50')
    ax1.plot(df_ohlc['Date'],df_ohlc['EMA100'].values, linewidth=1, label = 'EMA100')
    ax1.plot(df_ohlc['Date'],df_ohlc['EMA200'].values, linewidth=1, label = 'EMA200')
  
#    ax1.grid(True, color ='gray',linestyle = '--')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#    ax1.yaxis.label.set_color("w")
    ax1.spines['bottom'].set_color("#5998ff")
    ax1.spines['top'].set_color("#5998ff")
    ax1.spines['left'].set_color("#5998ff")
    ax1.spines['right'].set_color("#5998ff")
    ax1.tick_params(axis='y')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax1.tick_params(axis='x')
    plt.ylabel('Stock price')

    maLeg = plt.legend(loc=2, ncol=2, prop={'size':9},
               fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = pylab.gca().get_legend().get_texts()
    plt.xticks(rotation=30)
    
    orders = pd.Series(df_ohlc['Signal'].values, df_ohlc['Date'])
#    order_3_6_18 = pd.Series(df_ohlc['3_6_18_LONG'].values, df_ohlc['Date'])
    plotVline(orders)
#    plotVline(order_3_6_1s8)
    
#    
    ax0 = plt.subplot2grid((6,4), (0,0), sharex=ax1, rowspan=1, colspan=4, facecolor = 'w')
    
    fillcolor = '#00ffe8'
   
    posCol = '#386d13'
    negCol = '#8f2020'
    
    ax0.plot(df_ohlc['Date'], df['MACD_3_6'].values, color='darkcyan', lw=1)
    ax0.plot(df_ohlc['Date'], df['MACDSign369'].values, color='red', lw=1)
#    ax0.fill_between(df_ohlc['Date'], df['MACDDiff3620'].values, 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)
    ax0.axhline(0, color = 'gray', linewidth=1, linestyle = '--')
#    ax0.yaxis.label.set_color("w")
    ax0.spines['bottom'].set_color("#5998ff")
    ax0.spines['top'].set_color("#5998ff")
    ax0.spines['left'].set_color("#5998ff")
    ax0.spines['right'].set_color("#5998ff")
    ax0.tick_params(axis='y')
    ax0.tick_params(axis='x')
    plt.ylabel('1st trend')

    ax1v = ax1.twinx()
    ax1v.fill_between(df_ohlc['Date'],df['Volume'].values, facecolor='#00ffe8', alpha=.4)
   # make bar plots and color differently depending on up/down for the day
#    pos = df_ohlc['Open']- df_ohlc['Close'] <0
#    neg = df_ohlc['Open']- df_ohlc['Close'] >0
#    ax1v.bar(df_ohlc['Date'][pos],df_ohlc['Volume'][pos],color='green',width=1,align='center')
#    ax1v.bar(df_ohlc['Date'][neg],df_ohlc['Volume'][neg],color='red',width=1,align='center')
#    
#    #scale the x-axis tight
##    ax1v.set_xlim(min(dates),max(dates))
#    # the y-ticks for the bar were too dense, keep only every third one
#    yticks = ax1v.get_yticks()
#    ax1v.set_yticks(yticks[::3])
    

    
    
    ax1v.axes.yaxis.set_ticklabels([])
    ax1v.grid(False)
    ###Edit this to 3, so it's a bit larger
    ax1v.set_ylim(0, 3*df['Volume'].values.max())
    
    ax1v.spines['bottom'].set_color("#5998ff")
    ax1v.spines['top'].set_color("#5998ff")
    ax1v.spines['left'].set_color("#5998ff")
    ax1v.spines['right'].set_color("#5998ff")
    ax1v.tick_params(axis='x')
    ax1v.tick_params(axis='y')
    
    
    
    
    ax2 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4, facecolor='w')
    fillcolor = '#00ffe8'
#   
#    df['MACD_12_26'] = MACD
#    df['MACDSign9'] = MACDsign
#    df['MACDDiff'] = MACDdiff
    
    ax2.plot(df_ohlc['Date'], df['MACD_12_26'].values, color='darkcyan', lw=1)
    ax2.plot(df_ohlc['Date'], df['MACDSign12269'].values, color='red', lw=1)
#    ax2.fill_between(df_ohlc['Date'], df['MACDDiff501009'].values, 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)
    ax2.axhline(0, color = 'gray', linewidth=1, linestyle = '--')
#    ax2.grid(True, color ='gray', linestyle = '--')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax2.spines['bottom'].set_color("#5998ff")
    ax2.spines['top'].set_color("#5998ff")
    ax2.spines['left'].set_color("#5998ff")
    ax2.spines['right'].set_color("#5998ff")
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    plt.ylabel('2nd Trend')
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(30)

    plt.suptitle(ticker.upper())
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
   
    plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
    
 
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    candlestick_ohlc(ax, df_ohlc.values, width=.6, colorup='#53c156', colordown='#ff1717')         

    ax.plot(df_ohlc['Date'],df_ohlc['EMA18'].values, linewidth=1,label = 'EMA18', color = 'blue')
    ax.set_facecolor('w')
    maLeg = plt.legend(loc=2, ncol=2, prop={'size':9},
               fancybox=True, borderaxespad=0.)
  
    
#    
#    axv = fig2.add_subplot(2, 1, 2)   
#    axv.plot(df_ohlc['Date'], df['MACD_12_26'].values, color='green', lw=1)
#    axv.plot(df_ohlc['Date'], df['MACDSign9_1226'].values, color='red', lw=1)   
#    axv.axhline(0, color = 'gray', linewidth=1, linestyle = '--') 
#    axv.axes.yaxis.set_ticklabels([])
#    axv.grid(False)      
#    axv.spines['bottom'].set_color("#5998ff")
#    axv.spines['top'].set_color("#5998ff")
#    axv.spines['left'].set_color("#5998ff")
#    axv.spines['right'].set_color("#5998ff")
#    axv.tick_params(axis='x')
#    axv.tick_params(axis='y')
    
    
    
#    ax1.grid(True, color ='gray',linestyle = '--')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#    ax1.yaxis.label.set_color("w")
    ax.spines['bottom'].set_color("#5998ff")
    ax.spines['top'].set_color("#5998ff")
    ax.spines['left'].set_color("#5998ff")
    ax.spines['right'].set_color("#5998ff")
    ax.tick_params(axis='y')    
    ax.tick_params(axis='x')
    ax.autoscale_view()
    plt.ylabel('Stock price')
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(30)
    plt.title(ticker.upper() + " daily")
    
    
    
    
    
    plt.show()
       