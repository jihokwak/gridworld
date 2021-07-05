import numpy as np
import gym
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam

class DQN(tf.keras.Model) :
    def __init__(self, action_size, state_size):
        super().__init__()
        self.conv1 = Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=state_size)
        self.conv2 = Conv2D(64, (4,4), strides=(2,2), activation='relu')
        self.conv3 = Conv2D(64, (3,3), strides=(1,1), activation='relu')
        self.flatten = Flatten()
        self.fc = Dense(512, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        q = self.fc_out(x)
        return q

