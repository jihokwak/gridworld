import numpy as np
from collections import defaultdict
from dynamic.algorithm.deepsarsa.env.sarsa_env import Env

class SARSAgent:
    def __init__(self,actions: list):
        self.actions = actions
        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        # 초깃값으로 0을 가지는 큐함수 테이블 생성
        self.q_table = defaultdict(lambda: np.zeros(4, dtype=np.float32))

    def learn(self, state: list, action: int, reward: int, next_state: list, next_action: int):
        state, next_state = str(state), str(next_state)
        curr_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        td = reward + self.discount_factor * next_state_q - curr_q
        new_q = curr_q + self.step_size * td
        self.q_table[state][action] = new_q

    def get_action(self, state: list) -> int:
        if np.random.rand() < self.epsilon:
            # 무작위 행동반환
            action = np.random.choice(self.actions)
        else :
            # 큐함수에 따른 행동반환
            state = str(state)
            q_list = self.q_table[state]
            action = self.arg_max(q_list)
        return action

    def start(self, iter: int):
        for episode in range(iter):
            print(f"iter #{episode}")
            # 게임 환경과 상태를 초기화
            state = env.reset()
            # 현재 상태에 대한 행동을 선택
            action = self.get_action(state)

            done = False
            while not done:
                env.render()

                # 행동을 취한 후 다음 상태 보상 에피소드의 종료 여부를 받아옴
                next_state, reward, done = env.step(action)
                # 다음 상태에서의 다음 행동 선택
                next_action = self.get_action(next_state)
                self.learn(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action

                # 모든 큐함수를 화면에 표시
                env.print_value_all(agent.q_table)

    @staticmethod
    def arg_max(q_list: np.ndarray) -> int:
        max_idx_list = np.argwhere(q_list == q_list.max()).flatten().tolist()
        return np.random.choice(max_idx_list)

if __name__ == '__main__':
    env = Env()
    agent = SARSAgent(actions=list(range(env.n_actions)))
    agent.start(iter=1000)
