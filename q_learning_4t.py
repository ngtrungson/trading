# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:50:53 2020

@author: ADMIN
"""

import warnings
warnings.filterwarnings('ignore')


from pathlib import Path
from time import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import tensorflow as tf

import gym
from gym.envs.registration import register

from DDQNAgent import DDQNAgent

sns.set_style('whitegrid')

register(
    id='trading-v0',
    entry_point='trading_env:TradingEnvironment',
    max_episode_steps=1000
)


trading_environment = gym.make('trading-v0')
trading_environment.env.trading_cost_bps = 1e-3
trading_environment.env.time_cost_bps = 1e-4
trading_environment.env.ticker = 'fpt'
trading_environment.seed(42)


state_dim = trading_environment.observation_space.shape[0]  
num_actions = trading_environment.action_space.n
max_episode_steps = trading_environment.spec.max_episode_steps


gamma = .99,  # discount factor
tau = 100  # target network update frequency


architecture = (64, ) * 3  # units per layer
learning_rate = 5e-5  # learning rate
l2_reg = 1e-6  # L2 regularization

replay_capacity = int(1e6)
batch_size = 16

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 1e5

ddqn = DDQNAgent(state_dim=state_dim,
                 num_actions=num_actions,
                 learning_rate=learning_rate,
                 gamma=gamma,
                 epsilon_start=epsilon_start,
                 epsilon_end=epsilon_end,
                 epsilon_decay_steps=epsilon_decay_steps,
                 replay_capacity=replay_capacity,
                 architecture=architecture,
                 l2_reg=l2_reg,
                 tau=tau,
                 batch_size=batch_size)

total_steps = 0
max_episodes = 6000

episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []


ddqn.online_network.summary()

def track_results(episode, episode_nav,
                  market_nav, ratio,
                  epsilon):
    time_ma = np.mean([episode_time[-100:]])
    T = np.sum(episode_time)
    
    template = '{:>4d} | NAV: {:>5.3f} | Market NAV: {:>5.3f} | Delta: {:4.0f} | eps: {:>6.3f}'
    print(template.format(episode, episode_nav, market_nav, ratio, epsilon))
    
    
for episode in range(1, max_episodes + 1):
    this_state = trading_environment.reset()
    for episode_step in range(max_episode_steps):
        # print('-----------debugging --------------')
        # print(this_state.shape)
        # print('-----------end debugging --------------')
        # print(this_state.reshape(-1, state_dim).shape)
        # break;
        action = ddqn.epsilon_greedy_policy(this_state.reshape(-1, state_dim))
        next_state, reward, done, _ = trading_environment.step(action)
        ddqn.memorize_transition(this_state, action, reward, next_state,
                                  0.0 if done else 1.0)
        if ddqn.train:
            ddqn.experience_replay()
        if done:
            break
        this_state = next_state

    result = trading_environment.env.sim.result()
    final = result.iloc[-1]

    nav = final.nav * (1 + final.strategy_return)
    navs.append(nav)

    market_nav = final.market_nav
    market_navs.append(market_nav)

    diff = nav - market_nav
    diffs.append(diff)
    if episode % 50 == 0:
        track_results(episode, np.mean(navs[-100:]),
                      np.mean(market_navs[-100:]),
                      np.sum([s > 0 for s in diffs[-100:]]), 
                      ddqn.epsilon)
    if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
        print(result.tail())
        break        

trading_environment.close()


results = pd.DataFrame({'episode': list(range(1, episode + 1)),
                        'nav': navs,
                        'market_nav': market_navs,
                        'outperform': diffs})

fn = 'trading_agent_result_no_cost.csv'
results.to_csv(fn, index=False)


results = pd.read_csv('trading_agent_result_no_cost.csv')
results.columns = ['Episode', 'Agent', 'Market', 'difference']
results = results.set_index('Episode')
results['Strategy Wins (%)'] = (results.difference > 0).rolling(100).sum()
results.info()

fig, axes = plt.subplots(ncols=2, figsize=(14,4), sharey=True)
(results[['Agent', 'Market']]
  .sub(1)
  .rolling(100)
  .mean()
  .plot(ax=axes[0], 
        title='Annual Returns (Moving Average)', lw=1))
results['Strategy Wins (%)'].div(100).rolling(50).mean().plot(ax=axes[1], title='Agent Outperformance (%, Moving Average)');
for ax in axes:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
axes[1].axhline(.5, ls='--', c='k', lw=1)
fig.tight_layout()
fig.savefig('trading_agent', dpi=300)





