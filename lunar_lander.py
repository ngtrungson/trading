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
from DDQNAgent import DDQNAgent

import gym
from gym import wrappers

# class DDQNAgent:
#     def __init__(self, state_dim,
#                  num_actions,
#                  learning_rate,
#                  gamma,
#                  epsilon_start,
#                  epsilon_end,
#                  epsilon_decay_steps,
#                  replay_capacity,
#                  architecture,
#                  l2_reg,
#                  tau,
#                  batch_size,
#                  results_dir='results'):

#         self.state_dim = state_dim
#         self.num_actions = num_actions
#         self.experience = deque([], maxlen=replay_capacity)
#         self.learning_rate = learning_rate
#         self.gamma = gamma
#         self.architecture = architecture
#         self.l2_reg = l2_reg

#         self.online_network = self.build_model()
#         self.target_network = self.build_model(trainable=False)
#         self.update_target()

#         self.epsilon = epsilon_start
#         self.epsilon_decay_steps = epsilon_decay_steps
#         self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
#         self.epsilon_history = []

#         self.total_steps = self.train_steps = 0
#         self.episodes = self.episode_length = self.train_episodes = 0
#         self.steps_per_episode = []
#         self.episode_reward = 0
#         self.rewards_history = []

#         self.batch_size = batch_size
#         self.tau = tau
#         self.losses = []
#         self.idx = tf.range(batch_size)
#         self.results_dir = results_dir
#         self.train = True

#     def build_model(self, trainable=True):
#         layers = []
#         n = len(self.architecture)
#         for i, units in enumerate(self.architecture, 1):
#             layers.append(Dense(units=self.num_actions if i == n else units,
#                                 input_dim=self.state_dim if i == 1 else None,
#                                 activation='relu',
#                                 kernel_regularizer=l2(self.l2_reg),
#                                 trainable=trainable))

#         model = Sequential(layers)
#         model.compile(loss='mean_squared_error',
#                       optimizer=Adam(lr=self.learning_rate))
#         return model

#     def update_target(self):
#         self.target_network.set_weights(self.online_network.get_weights())

#     def epsilon_greedy_policy(self, state):
#         self.total_steps += 1
#         if self.train:
#             if self.total_steps < self.epsilon_decay_steps:
#                 self.epsilon -= self.epsilon_decay

#         if np.random.rand() <= self.epsilon:
#             return np.random.choice(self.num_actions)
#         q = self.online_network.predict(state)
#         return np.argmax(q, axis=1).squeeze()

#     def memorize_transition(self, s, a, r, s_prime, not_done):
#         if not_done:
#             self.episode_reward += r
#             self.episode_length += 1
#         else:
#             self.episodes += 1
#             self.rewards_history.append(self.episode_reward)
#             self.steps_per_episode.append(self.episode_length)
#             self.epsilon_history.append(self.epsilon)
#             self.episode_reward, self.episode_length = 0, 0
#             print(f'{self.episodes:03} | '
#                   f'Steps: {np.mean(self.steps_per_episode[-100:]):5.1f} | '
#                   f'Rewards: {np.mean(self.rewards_history[-100:]):8.2f} | '
#                   f'epsilon: {self.epsilon:.4f}')

#         self.experience.append((s, a, r, s_prime, not_done))

#     def experience_replay(self):
#         if self.batch_size > len(self.experience):
#             return
#         minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
#         states, actions, rewards, next_states, not_done = minibatch

#         next_q_values = self.online_network.predict_on_batch(next_states)
#         best_actions = tf.argmax(next_q_values, axis=1)

#         next_q_values_target = self.target_network.predict_on_batch(next_states)
#         target_q_values = tf.gather_nd(next_q_values_target,
#                                        tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))

#         targets = rewards + not_done * self.gamma * target_q_values

#         q_values = self.online_network.predict_on_batch(states).numpy()
#         q_values[[self.idx, actions]] = targets

#         loss = self.online_network.train_on_batch(x=states, y=q_values)
#         self.losses.append(loss)

#         if self.total_steps % self.tau == 0:
#             self.update_target()

#     def store_results(self):
#         path = Path(self.results_dir)
#         if not path.exists():
#             path.mkdir()
#         result = pd.DataFrame({'rewards': self.rewards_history,
#                                'steps'  : self.steps_per_episode,
#                                'epsilon': self.epsilon_history})
        
#         result.to_csv(path / 'results.csv', index=False)


env = gym.make('LunarLander-v2')
state_dim = env.observation_space.shape[0]  # number of dimensions in state
num_actions = env.action_space.n  # number of actions
max_episode_steps = env.spec.max_episode_steps  # max number of steps per episode
env.seed(42)

gamma = .99  # discount factor
learning_rate = 0.0001

architecture = (256, ) * 3  # units per layer
l2_reg = 1e-6  # L2 regularization

tau = 100  # target network update frequency
replay_capacity = int(1e6)
batch_size = 512


epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 25000

results_dir = Path('results')
monitor_path = results_dir / 'monitor'
video_freq = 50


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
                 batch_size=batch_size,
                 results_dir=results_dir)


env = wrappers.Monitor(env,
                       directory=monitor_path.as_posix(),
                       video_callable=lambda count: count % video_freq == 0,
                      force=True)


tf.keras.backend.clear_session()


max_episodes = 750
test_episodes = 0


while ddqn.episodes < max_episodes and test_episodes < 100:
    this_state = env.reset()
    done = False
    while not done:
        action = ddqn.epsilon_greedy_policy(this_state.reshape(-1, state_dim))
        next_state, reward, done, _ = env.step(action)
        ddqn.memorize_transition(this_state, action, reward, next_state, 0.0 if done else 1.0)
        if ddqn.train:
            ddqn.experience_replay()
        if done:
            if ddqn.train:
                if np.mean(ddqn.rewards_history[-100:]) > 200:
                    ddqn.train = False
            else:
                test_episodes += 1
            break
        this_state = next_state

env.close()
ddqn.store_results()

results = pd.read_csv(results_dir / 'results.csv').rename(columns=str.capitalize)
results['MA100'] = results.rolling(window=100, min_periods=25).Rewards.mean()

results.info()

fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharex=True)
results[['Rewards', 'MA100']].plot(ax=axes[0])
axes[0].set_ylabel('Rewards')
axes[0].set_xlabel('Episodes')
axes[0].axhline(200, c='k', ls='--', lw=1)
results[['Steps', 'Epsilon']].plot(secondary_y='Epsilon', ax=axes[1]);
axes[1].set_xlabel('Episodes')
fig.suptitle('Double Deep Q-Network Agent | Lunar Lander', fontsize=16)
fig.tight_layout()
fig.subplots_adjust(top=.9)
fig.savefig('figures/ddqn_lunarlander', dpi=300)