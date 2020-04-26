# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 08:55:26 2018

@author: sonng
"""


import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from agent import Agent
from methods import train_model, evaluate_model, format_position, preprocess_data
import keras.backend as K
import sys
import warnings
# import coloredlogs
import datetime as dt
#import numpy as np

from keras.utils.vis_utils import plot_model
from numpy import savetxt
from numpy import loadtxt
import urllib3

import bs4 as bs
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# logging.basicConfig(filename='runtimetradebot.log',level=logging.DEBUG)

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def get_info_stock(ticker):
    url = 'http://www.cophieu68.vn/snapshot.php?id=' + ticker   
    
    http = urllib3.PoolManager()
    r = http.request('GET', url)
    soup = bs.BeautifulSoup(r.data, 'lxml')
        
#    resp = requests.get(url)
#    print(resp.text)
#    soup = bs.BeautifulSoup(resp.text, 'lxml') 
    
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

def show_train_result(result, val_position, initial_offset):
    """ Displays training results
    """
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3],))


def show_eval_result(model_name, profit, initial_offset):
    """ Displays eval results
    """
    if profit == initial_offset or profit == 0.0:
        logging.info('{}: USELESS\n'.format(model_name))
    else:
        logging.info('{}: {}\n'.format(model_name, format_position(profit)))


def get_stock_data(stock_file):
    """Reads stock data from csv file
    """
    df = pd.read_csv(stock_file)
    return list(df['Adj Close'])


def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.

    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



def symbol_to_path(symbol, filetype="cp68"):
    """Return CSV file path given ticker symbol."""
    if filetype == "cp68":
        fileformat = "excel_{}.csv"        
    if filetype == "ssi":       
        fileformat = "Historical_Price_{}.csv"
    if (filetype == "yahoo") | (filetype == "alpha"):      
        fileformat = "{}.csv"
        
    return os.path.join(filetype, fileformat.format(str(symbol)))



def plot_result(df, history, title="trading session"):
    
    df = df.reset_index()
    # add history to dataframe
    position = [history[0][0]] + [x[0] for x in history]
    actions = ['HOLD'] + [x[1] for x in history]
    df.loc[:,'position'] = position
    df.loc[:,'action'] = actions
    
    df.loc[:,'day'] = df['date'].values
    # df['day'] = df['day'].map(mdates.date2num)
    
#    df['date'] = df['date'].map(mdates.date2num)
    buy = df[df['action']=='BUY']
    sell = df[df['action']=='SELL']    
#    plt.figure(figsize=(8, 8))
#    ax = plt.gca()
#    formatter = mdates.DateFormatter("%Y-%m")
#    ax.xaxis.set_major_formatter(formatter)
#    locator = mdates.DayLocator()
#    ax.xaxis.set_major_locator(locator)
    
    
    
    plt.plot(buy['day'], buy['close'].values, "go")
    plt.plot(sell['day'], sell['close'].values, "ro")
    plt.plot(df['day'], df['close'].values)
    
#    df['Close'].plot(label="close")
#    buy['Close'].plot(kind = 'line', marker='o', markersize=8, markerfacecolor='g', label="buy")
#    sell['Close'].plot(kind = 'line', marker='o', markersize=8, markerfacecolor='r', label="sell")


    ax = plt.gca()    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.legend(['buy', 'sell', 'close'], loc='upper right')
    plt.xlabel("Date")
    plt.ylabel("1K VND price")
    plt.title(title)
    plt.grid(True)
    plt.show()
    
def plot_loss_reward(total_rewards, total_losses):    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(total_rewards, color='blue')
    ax1.set_title('Episode reward over time')    
    ax1.set_ylabel('Rewards')
    ax1.grid(True)
    ax2.plot(total_losses, color='orange')
    ax2.set_title('Losses over time')
    ax2.set_ylabel('Losses')
    fig.subplots_adjust(hspace=0.2)
    plt.xlabel(" Episode ")
    ax2.grid(True)
    plt.show()
    

    
if __name__ == "__main__":    
   
    
    ticker = 'vnm' 
    start ="2006-1-19"
    end = "2020-4-24"
    update = False
    validation_size = 10
    
    # agent, history, val_df, val_profits, train_rewards, train_losses = auto_trading(ticker, start, end, update = False)
    file_path = symbol_to_path(ticker)
    
    df = pd.read_csv(file_path, index_col ="<DTYYYYMMDD>", parse_dates = True, 
                  usecols = ["<DTYYYYMMDD>", "<OpenFixed>","<HighFixed>","<LowFixed>","<CloseFixed>","<Volume>"], na_values = "nan")
    df = df.reset_index()
    df = df.rename(columns = {'<DTYYYYMMDD>': 'date', "<OpenFixed>": 'open', '<HighFixed>': 'high',
                              '<LowFixed>': 'low','<CloseFixed>' : 'close', '<Volume>': 'volume'})
 
    df = df.set_index('date')
    # change date index to increasing order
    df = df.sort_index()   
    # take a part of dataframe
    df = df.loc[start:end]
    
    if update:
        actual_price = get_info_stock(ticker)
        today = dt.datetime.today()
        next_date = today
        df.loc[next_date] = ({ 'close' : actual_price['Close'].iloc[-1],
                              'volume' : actual_price['Volume'].iloc[-1],
                              'open' : actual_price['Open'].iloc[-1],
                              'high' : actual_price['High'].iloc[-1], 
                              'low' : actual_price['Low'].iloc[-1]})
        df = df.reset_index()
        df = df.rename(columns = {'index': 'date'}) 
        df = df.set_index('date')
    
   
    
    valid_range = int(len(df)*validation_size/100)
    df_train = df[:-valid_range]
    df_val = df[-valid_range:]
    train_data =  list(df_train['close'])    
    val_data =  list(df_val['close'])
    
    normalized_data = df_train.copy()
    normalized_data = normalized_data.reset_index()
    normalized_data = preprocess_data(normalized_data, normalize=True)
    
    # print('normalized_data', normalized_data.shape)
    # print('df_train_data', len(df_train))
    
    strategy = "t-dqn"
    state_dim = normalized_data.shape[1]
    
    ep_count = 100
    batch_size = 32
    debug = True
    model_name = ticker + strategy
#    trainedmodel = os.path.join('models', ticker + '_'+ str(ep_count))
    trainedmodel = "models/{}_{}".format(model_name, ep_count)
#    print(trainedmodel)
    pretrained = os.path.exists(trainedmodel)
#    print(pretrained)
    
    
    initial_offset = val_data[1] - val_data[0]
    
    total_rewards = []
    total_losses = []
    
    
    # coloredlogs.install(level="DEBUG")
    switch_k_backend_device()
    
    if  pretrained == False:
        print(" No training data ! ")
        try:
            agent = Agent(state_dim, strategy=strategy, pretrained=False, model_name=model_name)            
            for episode in range(1, ep_count + 1):
                train_result = train_model(agent, episode, normalized_data, train_data, ep_count=ep_count,
                                           batch_size=batch_size)
                val_result, _ = evaluate_model(agent, normalized_data, val_data, debug)
                show_train_result(train_result, val_result, initial_offset)
                
                total_rewards.append(train_result[2])
                total_losses.append(train_result[3])
                
        except KeyboardInterrupt:
            print("Aborted!")  
    else: 
        model_name = model_name +'_'+ str(ep_count)
        agent = Agent(state_dim, pretrained=True, model_name=model_name)

    agent.model.summary()
    plot_model(agent.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    test_result, history = evaluate_model(agent, normalized_data, val_data, debug)
    show_eval_result(model_name, test_result, initial_offset)
    
    # df_val.loc[:,'day'] = df_val.index.values
    
    plot_result(df_val, history, title= "Auto trading " + model_name)
    print('Number episode ', ep_count, 'Deep Q network strategy', strategy, 'Batch size', batch_size)
    
    if  pretrained == False:
        plot_loss_reward(total_rewards, total_losses)
        savetxt("models/{}_{}_total_rewards.csv".format(model_name, ep_count), total_rewards, delimiter=',')
        savetxt("models/{}_{}_total_losses.csv".format(model_name, ep_count), total_losses, delimiter=',')
    else:
        try:
            total_rewards = loadtxt("models/{}_total_rewards.csv".format(model_name), delimiter=',')
            total_losses = loadtxt("models/{}_total_losses.csv".format(model_name), delimiter=',')
            plot_loss_reward(total_rewards, total_losses)
        except OSError:
            print(" File not found total_rewards, total_losses! ")
    print('Final profits: ', test_result)
    
    # sys.stdout = old_stdout
    
    
    
    
    
    
    