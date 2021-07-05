import matplotlib.pyplot as plt
import numpy as np
from env.deep_sarsa_env import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# 딥살사 인공신경망
class DeepSARSA(tf.keras.Model) :
    def __init__(self, action_size):
        super().__init__()
        self.fc1 = Dense(30, activation='relu')
        self.fc2 = Dense(30, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q

# 그리드월드 예제에서의 딥살사 에이전트
class DeepSARSAgent :
    def __init__(self, state_size, action_size):
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # 딥살사 하이퍼파리미터
        self.discount_factor = 0.99
        self.learning_rate = 0.01
        self.epsilon = 1.
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = DeepSARSA(self.action_size)
        self.optimizer = Adam(lr=self.learning_rate)

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        else :
            q_values = self.model(state)
            return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 학습 파라미터
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape :
            tape.watch(model_params)
            predict = self.model(state)[0]
            one_hot_action = tf.one_hot([action], self.action_size)
            predict = tf.reduce_sum(one_hot_action * predict, axis=1)

            # done = True 일 경우 에피소드가 끝나서 다음 샅태가 없음
            next_q = self.model(next_state)[0][next_action]
            target = reward + (1 - done) * self.discount_factor * next_q

            # MSE 오류함수 계산
            loss = tf.reduce_mean(tf.square(target - predict))

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

if __name__ == '__main__':
    # 환경과 에이전트 생성
    env = Env(render_speed=0.01)
    state_size = 15
    action_space = [0,1,2,3,4]
    action_size = len(action_space)
    agent = DeepSARSAgent(state_size, action_size)

    scores, episodes = [], []

    EPISODES = 1000
    for e in range(EPISODES) :
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done :
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_action = agent.get_action(next_state)

            # 샘플로 모델 학습
            agent.train_model(state, action, reward, next_state, next_action, done)
            score += reward
            state = next_state

            if done :
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:3d} | epsilon: {:.3f}".format(e, score, agent.epsilon))

                scores.append(score)
                episodes.append(e)
                plt.plot(episodes, scores, 'b')
                plt.xlabel("episode")
                plt.ylabel("average score")
                plt.savefig("./save_graph/graph.png", dpi=300)

        if e % 100 == 0:
            agent.model.save_weights('save_model/model', save_format='tf')