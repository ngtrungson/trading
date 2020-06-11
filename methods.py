
import logging
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.preprocessing import scale
# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))

# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))


def load_data(ticker, start_date, end_date):         
    filepath = os.path.join("cp68", "excel_{}.csv".format(ticker))        
    # df = pd.read_csv(filepath, index_col ="Date", parse_dates = True,  
    #                  usecols = ["Date", "High", "Low", "Close", "Volume"], na_values = "nan")
   
    df = pd.read_csv(filepath, index_col ="<DTYYYYMMDD>", parse_dates = True, 
             usecols = ["<DTYYYYMMDD>", "<CloseFixed>","<Volume>"], na_values = "nan")
    df = df.reset_index()
    df = df.rename(columns = {'<DTYYYYMMDD>': 'date', '<CloseFixed>' : 'close', '<Volume>': 'volume'})
  
    df = df.set_index('date')
    
    
    # df = df.reset_index()
    # df = df.set_index('Date')
    # df = df.rename(columns = {'Close' : 'close', 'Volume': 'volume','High': 'high', 'Low':'low'}) 
    # # df['hl_pct'] = (df['high'] - df['low']) / df['close']
    # df = df.drop(columns=['high','low'])
    df = df.sort_index()   
    # take a part of dataframe
    df = df.loc[start_date:end_date]    
    
    return df



def rsi(data, window=14):
    diff = data.diff().dropna()
    
    up, down = diff.copy(), diff.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    rolling_up = up.rolling(window).mean()
    rolling_down = down.abs().rolling(window).mean()
    
    RS2 = rolling_up / rolling_down
    return 100 - (100 / (1 + RS2))
    
def momentum(data, window=100):
    def pct_rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1]
    
    return data.rolling(window).apply(pct_rank, raw=True)
    
def preprocess_data(data, normalize=True):
    """calculate returns and percentiles, then removes missing values"""
    data['returns'] = data.close.pct_change()
    data['hl_pct'] = (data.high - data.low)/data.close
    pct_change = (data.close - data.open) / data.open
    vol_ma15 = momentum(data.volume, window = 15)
    data['volume_ratio'] = vol_ma15/data.volume
    data['volume_returns'] = pct_change*data['volume_ratio']
    data = data.drop(['date','open','high','low'], axis = 1) 
    # make volume positive and pre-scale
    data.volume = np.log(data.volume.replace(0, 1))
    
    
    data['close_pct_50'] = momentum(data.close, window=50)
    data['volume_pct_50'] = momentum(data.volume, window=50)
    data['close_pct_30'] = momentum(data.close, window=30)
    data['volume_pct_30'] = momentum(data.volume, window=30)
    data['return_5'] = data.returns.pct_change(5)
    data['return_21'] = data.returns.pct_change(21)
    data['rsi'] = rsi(data.close)
    data = data.replace((np.inf, -np.inf), np.nan)
    data.fillna(method ="ffill", inplace = True)
    data.fillna(method ="backfill", inplace = True)
    
    # data = data.replace((np.inf, -np.inf), np.nan).dropna()
    
    r = data.returns.copy()
    if normalize:
        data = pd.DataFrame(scale(data),
                                 columns=data.columns,
                                 index=data.index)
    data['returns'] = r  # don't scale returns
    
    return data



def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


# def get_state(data, t, n_days):
#     """Returns an n-day state representation ending at time t
#     """
#     d = t - n_days + 1
#     block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1]  # pad with t0
#     res = []
#     for i in range(n_days - 1):
#         res.append(sigmoid(block[i + 1] - block[i]))
#     return np.array([res])

def get_state(normalized_data, t):
    """Returns state at time t
    """
    state = normalized_data.loc[t].values    
    # print(type(state),state.shape)
    
    return state



def train_model(agent, episode, normalized_data, data, ep_count=100, batch_size=32):
    total_profit = 0
    data_length = len(data) - 1
    state_dim = normalized_data.shape[1]
    agent.inventory = []
    avg_loss = []

    state = get_state(normalized_data, 0)
    

    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):        
        reward = 0
        next_state = get_state(normalized_data, t+1)

        
        # select an action
        state1 = state.reshape(-1, state_dim)
        action = agent.act(state1)

        # BUY
        if action == 1:
            agent.inventory.append(data[t])

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta #max(delta, 0)
            total_profit += delta

        # HOLD
        else:
            pass

        done = (t == data_length - 1)
        
        agent.remember(state, action, reward, next_state, 0.0 if done else 1.0)
        
       
        if agent.get_exp_replay_size() > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 10 == 0:
        agent.save(episode)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, normalized_data, data, debug):
    total_profit = 0
    data_length = len(data) - 1
    state_dim = normalized_data.shape[1]
    history = []
    agent.inventory = []
    
    state = get_state(normalized_data, 0)

    for t in range(data_length):        
        reward = 0
        next_state = get_state(normalized_data, t+1)
        
        # select an action
        state1 = state.reshape(-1, state_dim)
        action = agent.act(state1, is_eval=True)

        # BUY
        if action == 1:
            agent.inventory.append(data[t])

            history.append((data[t], "BUY"))
            if debug:
                logging.debug("Buy at: {}".format(format_currency(data[t])))
        
        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta #max(delta, 0)
            total_profit += delta

            history.append((data[t], "SELL"))
            if debug:
                logging.debug("Sell at: {} | Position: {}".format(
                    format_currency(data[t]), format_position(data[t] - bought_price)))
        # HOLD
        else:
            history.append((data[t], "HOLD"))

        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, 0.0 if done else 1.0)

        state = next_state
        if done:
            return total_profit, history
