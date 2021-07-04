import os
import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from .train import DQN


# 카트풀 예제에서의 DQN 에이전트
class DQNAgent :
    def __init__(self, state_size, action_size):
        #상태와 해동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        #모델과 타깃 모델 생성
        self.model = DQN(action_size)
        self.model.load_weights("./save_model/trainded/model")

    #입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        q_value = self.model(state)
        return np.argmax(q_value[0])


if __name__ == '__main__':
    # Carpole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CarPole-v1')
    state_size = env.observation_space.shape[0]
    action_size  = env.action_space.n

    # DQN 에이전트 생성
    agent = DQNAgent(state_size, action_size)

    num_episode = 10
    for e in range(num_episode) :
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done :
            env.render()

            # 현재 상태로 행동을 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            state = next_state

            if done :
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | scpre {:.3f} ".format(e, score))
