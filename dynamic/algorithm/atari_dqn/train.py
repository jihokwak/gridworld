import numpy as np
import pandas as pd
import gym
import os
import sys
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam

from skimage.color import rgb2grey
from skimage.transform import resize

BASEDIR = os.path.dirname(os.path.abspath(__file__))

class custom_deque(deque) :
    def __init__(self, cols, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cols = cols

    def get_values(self, sample_size):
        df = pd.DataFrame(self, columns=self.cols).sample(sample_size)
        return [df[col].values for col in self.cols]

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

class DQNAgent :
    def __init__(self, action_size, state_size=(84,84,4)):
        self.render = False

        self.state_size = state_size
        self.action_size = action_size

        # 하이퍼 파라미터
        self.discount_factor= 0.99
        self.learning_rate = 1e-4
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.02
        self.exploration_steps = 1000000
        self.epsilon_decay_steps = np.linspace(self.epsilon_start, self.epsilon_end, 1000000)
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000

        # 리플레이 메모리, 최대 크기 100,000
        self.memory = custom_deque(maxlen=100000, cols=['history', 'action', 'reward', 'next_history', 'dead'])
        # 게임시작 후 랜덤하게 움직이지 않는 것에 대한 옵션
        self.no_op_steps = 30

        # 모델과 타깃 모델 생성
        self.model = DQN(action_size, state_size)
        self.target_model = DQN(action_size, state_size)
        self.optimizer = Adam(self.learning_rate, clopnorm=10.)

        # 타깃 모델 초기화
        self.update_target_model()

        self.avg_q_max, self.avg_loss = 0, 0
        os.makedirs(os.path.join(BASEDIR, 'summary'), exist_ok=True)
        os.makedirs(os.path.join(BASEDIR, 'save_model'), exist_ok=True)
        self.writer = tf.summary.create_file_writer(os.path.join(BASEDIR, 'summary', 'breakout_dqn'))
        self.model_path = os.path.join(BASEDIR, 'save_model', 'atari_model')

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else :
            q_value = self.model(history)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'> 을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, dead):
        history = np.float32(history[0] / 255.)
        next_history = np.float32(next_history[0] / 255.)
        self.memory.append((history, action, reward, next_history, dead))

    # 텐서보드에 학습 정보를 기록
    def draw_tensorboard(self, score, step, episode):
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=episode)
            tf.summary.scalar('Average Max Q/Episode', self.avg_q_max / float(step), step=episode)
            tf.summary.scalar('Duration/Episode', step, step=episode)
            tf.summary.scalar('Average Loss/Episode', self.avg_loss / float(step), step=episode)

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end :
            self.epsilon = self.epsilon_decay_steps[0]

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        batch = self.memory.get_values(self.batch_size)
        history, actions, rewards, next_history, dones = batch
        actions = np.int32(actions)

        # 학습 파라미터
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape :
            # 현재 상태에 대한 모델의 큐함수
            predicts = self.model(history)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            # 다음 상태에 대한 타깃 모델의 큐함수
            target_predicts = self.target_model(next_history)

            # 벨만 최적 방정식을 구성하기 위한 타깃과 큐함수의 최댓값 계산
            max_q = np.amax(target_predicts, axis=1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q

            # 후버로스 계산
            error = tf.abs(targets - predicts)
            quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

            self.avg_loss += loss.numpy()

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

    # 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2grey(observe), (84, 84), mode='constant') * 255
    )
    return processed_observe

if __name__ == '__main__':
    # 환경과 DQN 에이전트 생성
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent(action_size=3)

    global_step = 0
    score_avg = 0
    score_max = 0

    # 불필요한 행동을 없애기 위한 딕셔너리 선언
    action_dict = {0:1, 1:2, 2:3, 3:3}

    num_episode = 50000
    for e in range(num_episode):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        # env 초기화
        observe = env.reset()

        # 랜덤으로 뽑힌 값만큼의 프레임 동안 움직이지 않음
        for _ in range(np.random.randint(1, agent.no_op_steps)) :
            observe, _, _, _ = env.step()

        # 프레임을 전처리한 후 4개의 상태를 쌓아서 입력값으로 사용
        state = pre_processing(observe)
        history = np.tile(state[np.newaxis,:,:,:],(1,1,1,4))
        # history = np.stack([state]*4, axis=-1)
        # history = np.reshape([history], (1, 84, 84, 4))

        while not done :
            if agent.render:
                env.render()
            global_step += 1
            step += 1

            # 바로 전 history 를 입력으로 받아 행동을 선택
            action = agent.get_action(history)
            # 1: 정지, 2: 왼쪽, 3: 오른쪽
            real_action = action_dict[action]

            # 죽었을 때 시작하기 위해 발사 행동을 함
            if dead :
                action, real_action, dead = 0, 1, False

            # 선택한 행동을 환경에서 한 타임스텝 진행
            observe, reward, done, info = env.step(real_action)
            # 각 타임스텝마다 상태 전처리
            next_state = pre_processing(observe)
            next_state = next_state[np.newaxis,:,:,np.newaxis] #np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=-1)

            agent.avg_q_max += np.amax(agent.model(np.float32(history / 255.))[0])

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            score += reward
            reward = np.clip(reward, -1., 1.)
            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, dead)

        # 리플레이 메모리 크기가 정한 수치에 도달한 시점부터 모델 학습 시작
        if len(agent.memory) >= agent.train_start:
            agent.train_model()
            # 일정 시간마다 타깃 모델을 모델의 가중치로 업데이트
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

                if dead:
                    history = np.tile(next_state[np.newaxis, :,:,:], (1,1,1,4))
                    # history = np.stack([next_state]*4, axis=2)
                    # history = np.reshape([history], (1, 84, 84, 4))
                else :
                    history = next_history

                if done:
                    # 에피소드당 학습 정보를 기록
                    if global_step > agent.train_start :
                        agent.draw_tensorboard(score, step, e)

                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    score_max = score if score > score_max else score_max

                    log = '''
                    episode: {:5d} | score: {:4.1f} | score max: {:4.1f} | score avg: {:4.1f} | memory length: {:5d} | epsilon: {:.3f} | q avg: {:3.2f} | avg loss: {:3.2f}
                    '''.format(e, score, score_max, score_avg, len(agent.memory), agent.epsilon, agent.avg_q_max/float(step), agent.avg_loss/float(step)).strip()
                    print(log)

                    agent.avg_q_max, agent.avg_loss = 0, 0

            # 1000에피소드마다 모델 저장
            if e % 1000 == 0:
                os.makedirs(os.path.join(BASEDIR, 'save_model'), exist_ok=True)
                agent.model.save_weights(os.path.join(BASEDIR, 'save_model', 'atari_model'), save_format='tf')




