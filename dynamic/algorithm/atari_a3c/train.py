import os, gym, time, threading, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.compat.v1.train import AdamOptimizer

from skimage.color import rgb2grey
from skimage.transform import resize

# 멀티스레딩을 위한 글로벌 변수
global episode, socre_avg, score_max
episode, score_avg, score_max = 0,0,0
num_episode = 8000000

# ActorCritic 인공신경망
class ActorCritic(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super().__init__()

        self.conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=state_size)
        self.conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.flatten = Flatten()
        self.shared_fc = Dense(512, activation='relu')
        self.policy = Dense(action_size, activation='linear')
        self.value = Dense(1, activation='linear')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.shared_fc(x)

        policy = self.policy(x)
        value = self.value(x)
        return policy, value

# 브레이크아웃에서의 A3CAgent 클래스 (글로벌 신경망)
class A3CAgent() :
    def __init__(self, action_size, env_name):
        self.env_name = env_name
        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # A3C 하이퍼파라미터
        self.discount_factor = 0.99
        self.no_op_steps = 30
        self.lr = 1e-4
        # 스레드의 개수
        self.threads = 16
