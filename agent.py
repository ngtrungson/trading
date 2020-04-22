import random

from collections import deque

import numpy as np
import tensorflow as tf
import keras.backend as K
from random import sample
from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense
from keras.optimizers import Adam


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

    def __init__(self, state_dim, strategy="t-dqn", reset_every=1000, pretrained=False, model_name=None):
        self.strategy = strategy

        # agent config
        self.state_dim = state_dim    	# normalized 
        self.action_size = 3           		# [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True
        # model config
        self.model_name = model_name
        self.gamma = 0.95 # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(lr=self.learning_rate)

        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):
        """Creates the model
        """
        model = Sequential()
        model.add(Dense(units=32, activation="relu", input_dim=self.state_dim))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=16, activation="relu"))
        # model.add(Dense(units=8, activation="relu"))
        model.add(Dense(units=self.action_size))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory
        """
        not_done = 0.0 if done else 1.0
        
        self.memory.append((state, action, reward, next_state, not_done))

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions
        """
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1 # make a definite buy on the first iter

        action_probs = self.model.predict(state)
        action = np.argmax(action_probs[0])
        
        return action

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory
        """
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
            if self.n_iter % self.reset_every == 0:
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
            if self.n_iter % self.reset_every == 0:
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

        # update q-function parameters based on huber loss gradient
        loss = self.model.fit(
            x= states, y=q_values,
            epochs=1, verbose=0
        ).history["loss"][0]

        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        self.model.save("models/{}_{}".format(self.model_name, episode))

    def load(self):
        return load_model("models/" + self.model_name, custom_objects=self.custom_objects)
