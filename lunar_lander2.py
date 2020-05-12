# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:01:47 2020

@author: ADMIN
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from collections import deque
from random import sample, shuffle
import numpy as np
import pandas as pd
from itertools import product

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt
import seaborn as sns
from agent import Agent

import gym
from gym import wrappers


env = gym.make('LunarLander-v2')
state_dim = env.observation_space.shape[0]  # number of dimensions in state
num_actions = env.action_space.n  # number of actions
max_episode_steps = env.spec.max_episode_steps  # max number of steps per episode
env.seed(42)

# gamma = .99  # discount factor
# learning_rate = 0.0001

# architecture = (256, ) * 3  # units per layer
# l2_reg = 1e-6  # L2 regularization

# tau = 100  # target network update frequency
# replay_capacity = int(1e6)
# batch_size = 512


# epsilon_start = 1.0
# epsilon_end = 0.05
# epsilon_decay_steps = 25000

results_dir = Path('results')
monitor_path = results_dir / 'monitor'
video_freq = 50

strategy = "double-dqn"
dueling_type = 'no'
ep_count = 100
batch_size = 512
model_name = 'LunarLander' + strategy

agent = Agent(state_dim=state_dim, 
              action_size = num_actions,
              strategy=strategy, 
              dueling_type= dueling_type, 
              pretrained=False, 
              model_name=model_name)            
          
agent.model.summary()
# ddqn = DDQNAgent(state_dim=state_dim,
#                  num_actions=num_actions,
#                  learning_rate=learning_rate,
#                  gamma=gamma,
#                  epsilon_start=epsilon_start,
#                  epsilon_end=epsilon_end,
#                  epsilon_decay_steps=epsilon_decay_steps,
#                  replay_capacity=replay_capacity,
#                  architecture=architecture,
#                  l2_reg=l2_reg,
#                  tau=tau,
#                  batch_size=batch_size,
#                  results_dir=results_dir)


env = wrappers.Monitor(env,
                       directory=monitor_path.as_posix(),
                       video_callable=lambda count: count % video_freq == 0,
                      force=True)


# tf.keras.backend.clear_session()


max_episodes = 750
test_episodes = 0

avg_loss = []

while agent.episodes < max_episodes:
    this_state = env.reset()
    done = False
    while not done:
        action = agent.act(this_state.reshape(-1, state_dim))
        next_state, reward, done, _ = env.step(action)
        agent.remember(this_state, action, reward, next_state, 0.0 if done else 1.0)
        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)        
        this_state = next_state

env.close()
# ddqn.store_results()

# results = pd.read_csv(results_dir / 'results.csv').rename(columns=str.capitalize)
# results['MA100'] = results.rolling(window=100, min_periods=25).Rewards.mean()

# results.info()

# fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharex=True)
# results[['Rewards', 'MA100']].plot(ax=axes[0])
# axes[0].set_ylabel('Rewards')
# axes[0].set_xlabel('Episodes')
# axes[0].axhline(200, c='k', ls='--', lw=1)
# results[['Steps', 'Epsilon']].plot(secondary_y='Epsilon', ax=axes[1]);
# axes[1].set_xlabel('Episodes')
# fig.suptitle('Double Deep Q-Network Agent | Lunar Lander', fontsize=16)
# fig.tight_layout()
# fig.subplots_adjust(top=.9)
# fig.savefig('figures/ddqn_lunarlander', dpi=300)