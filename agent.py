import random

from collections import deque
from pathlib import Path


import numpy as np
import pandas as pd

import tensorflow as tf
import keras.backend as K

from keras.models import Sequential, Model
from keras.models import load_model, clone_model
from keras.layers import Dense, Lambda, Layer, Input, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2

def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """ Stock Trading Bot """

    def __init__(self, state_dim, 
                 action_size = 3, 
                 strategy="t-dqn", 
                 dueling_type='no', 
                 epsilon_start = 1.0,
                 epsilon_end = 0.01,
                 epsilon_decay_steps = 25000,
                 reset_every=100, pretrained=False, model_name=None):
        self.strategy = strategy

        # agent config
        self.state_dim = state_dim    	# normalized 
        self.action_size = action_size           		# default = 3 [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=100000)
        self.first_iter_trading = False
        
        self.total_steps = 0
        self.episodes = self.episode_length = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []
        self.losses = []
        
        self.epsilon =  epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_history = []
        
        
        # model config
        self.model_name = model_name
        self.gamma = 0.99 # affinity for long term reward
        self.l2_reg = 1e-6
        self.dueling_type = dueling_type
        
        # self.epsilon = 1.0
        # self.epsilon_min = 0.01
        # self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(lr=self.learning_rate)
        
        
        self.pretrained = pretrained
        self.results_dir ='results'
        
        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

       
        
        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.total_steps = 1
            self.reset_every = reset_every

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):
        """Creates the model
        """
       
        model = Sequential()
        # Lunar Lander neural networks 256/128
        model.add(Dense(units=256, activation="relu", kernel_regularizer=l2(self.l2_reg), input_dim=self.state_dim))
        model.add(Dense(units=128, activation="relu", kernel_regularizer=l2(self.l2_reg)))
        
        # model.add(Dense(units=32, activation="relu",  input_dim=self.state_dim))
        # model.add(Dense(units=64, activation="relu"))
        # model.add(Dense(units=16, activation="relu"))
        # model.add(Dense(units=8, activation="relu"))
        
        model.add(Dense(units=self.action_size, activation='linear'))
            
        if self.dueling_type == 'avg':
            layer = model.layers[-2]  # Get the second last layer of the model
            nb_action = model.output._keras_shape[-1]  #  remove the last layer
            y = Dense(nb_action + 1, activation='linear')(layer.output)
             # lambda a: a[:, :] — k.mean(a[:, :], keepdims=True)
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                                 output_shape=(nb_action,))(y)  #  Using the avg dueling type
            model = Model(inputs=model.input, outputs=outputlayer)
        elif self.dueling_type == 'max':
            layer = model.layers[-2]  # Get the second last layer of the model
            nb_action = model.output._keras_shape[-1]  #  remove the last layer
            y = Dense(nb_action + 1, activation='linear')(layer.output)
             # lambda a: a[:, :] — k.mean(a[:, :], keepdims=True)
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                                 output_shape=(nb_action,))(y)  #  Using the max dueling type
            model = Model(inputs=model.input, outputs=outputlayer)
        elif self.dueling_type == 'naive':
            layer = model.layers[-2]  # Get the second last layer of the model
            nb_action = model.output._keras_shape[-1]  #  remove the last layer
            y = Dense(nb_action + 1, activation='linear')(layer.output)
             # lambda a: a[:, :] — k.mean(a[:, :], keepdims=True)
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:],
                                 output_shape=(nb_action,))(y)  #  Using the naive dueling type
            model = Model(inputs=model.input, outputs=outputlayer)
        else:    
            
            print("Dueling disabled, otherwise dueling_type must be one of {'avg','max','naive'}")
            
            
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, not_done):
        """Adds relevant data to memory
        """
        if not_done:
            self.episode_reward += reward
            self.episode_length += 1
        else:
            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.epsilon_history.append(self.epsilon)
            #reset some parameters after done
            self.episode_reward, self.episode_length = 0, 0
            print(f'{self.episodes:03} | '
                  f'Steps: {np.mean(self.steps_per_episode[-100:]):5.1f} | '
                  f'Rewards: {np.mean(self.rewards_history[-100:]):8.2f} | '
                  f'epsilon: {self.epsilon:.4f}')
        
        # not_done = 0.0 if done else 1.0
        
        self.memory.append((state, action, reward, next_state, not_done))

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions
        """
        self.total_steps += 1 
        if not self.pretrained:
            if self.total_steps < self.epsilon_decay_steps:
                self.epsilon -= self.epsilon_decay
        # take random action in order to diversify experience at the beginning
        # if not is_eval and random.random() <= self.epsilon:
        #     return random.randrange(self.action_size)


        if not is_eval and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        if self.first_iter_trading:
            self.first_iter_trading = False
            return 1 # make a definite buy on the first iter

        
        action_probs = self.model.predict(state)
        
        # action = np.argmax(action_probs[0])
        action = np.argmax(action_probs, axis=1).squeeze()
        return action

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory
        """
        # if batch_size > len(self.memory):
        #     return
        minibatch = random.sample(self.memory,batch_size)
        idx = np.arange(batch_size)
        states, actions, rewards, next_states, not_done = map(np.array, zip(*minibatch)) 
        
        
        # DQN
        if self.strategy == "dqn":
            next_q_values = self.model.predict(next_states) 
            best_actions = np.argmax(next_q_values, axis=1)                 
            targets = rewards + not_done* self.gamma * next_q_values[[idx, best_actions]]
            # targets = rewards + not_done* self.gamma * np.amax(next_q_values, axis=1)  
            
            q_values = self.model.predict(states) 
            q_values[[idx, actions]] = targets
            
           
        # DQN with fixed targets
        elif self.strategy == "t-dqn":
            # update q-function parameters based on huber loss gradient
            if self.total_steps % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())  
            next_q_values_targets = self.target_model.predict(next_states)
            best_actions = np.argmax(next_q_values_targets, axis=1)            
            targets = rewards + not_done* self.gamma * next_q_values_targets[[idx, best_actions]]
            # targets = rewards + not_done* self.gamma * np.amax(next_q_values_targets, axis=1)  
           
            q_values = self.model.predict(states) 
            q_values[[idx, actions]] = targets
            
            
        # Double DQN
        elif self.strategy == "double-dqn":
            # update q-function parameters based on huber loss gradient
            if self.total_steps % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())
                
            next_q_values = self.model.predict(next_states)
            best_actions = np.argmax(next_q_values, axis=1)
            next_q_values_targets = self.target_model.predict(next_states)
            target_q_values = next_q_values_targets[[idx, best_actions]]
            targets = rewards + not_done* self.gamma * target_q_values  
            
            q_values = self.model.predict(states) 
            q_values[[idx, actions]] = targets
                
        else:
            raise NotImplementedError()

        loss = self.model.fit(
                x= states, y=q_values,
                epochs=1, verbose=0
            ).history["loss"][0]
        
        self.losses.append(loss)
        
        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        self.model.save("models/{}_{}".format(self.model_name, episode))

    def load(self):
        return load_model("models/" + self.model_name, custom_objects=self.custom_objects)
    
    def store_results(self):
        path = Path(self.results_dir)
        if not path.exists():
            path.mkdir()
        result = pd.DataFrame({'rewards': self.rewards_history,
                               'steps'  : self.steps_per_episode,
                               'epsilon': self.epsilon_history})

        result.to_csv(path / 'results.csv', index=False)
