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
from methods import train_model, evaluate_model, format_position
import keras.backend as K
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")



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
    # add history to dataframe
    position = [history[0][0]] + [x[0] for x in history]
    actions = ['HOLD'] + [x[1] for x in history]
    df.loc[:,'position'] = position
    df.loc[:,'action'] = actions
    
    df['date'] = df['date'].map(mdates.date2num)
    
#    df['date'] = df['date'].map(mdates.date2num)
    buy = df[df['action']=='BUY']
    sell = df[df['action']=='SELL']    
#    plt.figure(figsize=(8, 8))
#    ax = plt.gca()
#    formatter = mdates.DateFormatter("%Y-%m")
#    ax.xaxis.set_major_formatter(formatter)
#    locator = mdates.DayLocator()
#    ax.xaxis.set_major_locator(locator)
    
    
    plt.plot(df['date'], df['Close'].values)
    plt.plot(buy['date'], buy['Close'].values, "go")
    plt.plot(sell['date'], sell['Close'].values, "ro")
    
#    df['Close'].plot(label="close")
#    buy['Close'].plot(kind = 'line', marker='o', markersize=8, markerfacecolor='g', label="buy")
#    sell['Close'].plot(kind = 'line', marker='o', markersize=8, markerfacecolor='r', label="sell")


    ax = plt.gca()    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.legend(['close', 'buy', 'sell'], loc='upper left')
    plt.xlabel("Date")
    plt.ylabel("1K VND price")
    plt.title(title)
    plt.grid(True)
    plt.show()
    
 
    
def training(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False):
    """ Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python train.py --help]
    """
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    
    train_data = get_stock_data(train_stock)
    val_data = get_stock_data(val_stock)

    initial_offset = val_data[1] - val_data[0]

    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)

def evaluation(eval_stock, window_size, model_name, debug):
    """ Evaluates the stock trading bot.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python eval.py --help]
    """    
    data = get_stock_data(eval_stock)
    initial_offset = data[1] - data[0]

    # Single Model Evaluation
    if model_name is not None:
        agent = Agent(window_size, pretrained=True, model_name=model_name)
        profit, _ = evaluate_model(agent, data, window_size, debug)
        show_eval_result(model_name, profit, initial_offset)
        
    # Multiple Model Evaluation
    else:
        for model in os.listdir("models"):
            if os.path.isfile(os.path.join("models", model)):
                agent = Agent(window_size, pretrained=True, model_name=model)
                profit = evaluate_model(agent, data, window_size, debug)
                show_eval_result(model, profit, initial_offset)
                del agent



def auto_trading(ticker, start, end, validation_size = 180):
    file_path = symbol_to_path(ticker)
    
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
    # take a part of dataframe
    df = df.loc[start:end]
    
    df_train = df[:-validation_size]
    df_val = df[-validation_size:]
    
    strategy = "t-dqn"
    window_size = 20
    ep_count = 50
    batch_size = 32
    debug = False
    model_name = ticker
#    trainedmodel = os.path.join('models', ticker + '_'+ str(ep_count))
    trainedmodel = "models/{}_{}".format(ticker, ep_count)
#    print(trainedmodel)
    pretrained = os.path.exists(trainedmodel)
#    print(pretrained)
    train_data =  list(df_train['Close'])
    
    val_data =  list(df_val['Close'])
    
    initial_offset = val_data[1] - val_data[0]
    
    
#    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()
    
    if  pretrained == False:
        print(" No training data ! ")
        try:
            agent = Agent(window_size, strategy=strategy, pretrained=False, model_name=model_name)            
            for episode in range(1, ep_count + 1):
                train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                           batch_size=batch_size, window_size=window_size)
                val_result, _ = evaluate_model(agent, val_data, window_size, debug)
                show_train_result(train_result, val_result, initial_offset)
        except KeyboardInterrupt:
            print("Aborted!")  
    else: 
        model_name = model_name + '_'+ str(ep_count)
        agent = Agent(window_size, pretrained=True, model_name=model_name)


    
    test_result, history = evaluate_model(agent, val_data, window_size, debug)
    show_eval_result(model_name, test_result, initial_offset)
    
    
#    df = df[['Date', 'Close']]
    # rename feature column names
#    df_val = df_val.rename(columns={'Close': 'actual'})    
#    df_val['actual'] = df_val['Close']  
    df_val.loc[:,'date'] = df_val.index.values
    
    plot_result(df_val, history, title= "Auto trading " + ticker)
    
#    chart = visualize(df_val, history, title= ticker)
#    chart.save('results/{}_robot.html'.format(ticker))
    
    return  df_val, history, test_result


    
if __name__ == "__main__":
    ticker = 'PNJ' 
    start ="2010-3-18"
    end = "2020-4-10"
    df_test, history, result = auto_trading(ticker, start, end)
    
    
    
    
    
    