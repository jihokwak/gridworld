import random
import numpy as np
from environment import GraphicDisplay, Env

class PolicyIteration :
    def __init__(self, env):
        self.env = env
        self.value_table = np.zeros((env.height, env.width), dtype=np.float32)
        self.policy_table = np.ones((env.height, env.width,4), dtype=np.float32) * 0.25

        self.policy_table[2,2] = None # 마침 상태 설정
        self.discount_factor = 0.9

    def policy_evaluation(self):
        next_value_table = np.zeros((self.env.height, self.env.width), dtype=np.float32) # 다음가치함수 초기화

        for state in self.env.get_all_states() :
            value = 0.0
            # 마침 상태의 가치함수 = 0
            if state == [2,2] :
                next_value_table[state[0], state[1]] = 0.0
                continue
            # 벨만 기대 방정식
            for action in self.env.possible_actions :
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += self.get_policy(state)[action] * (reward + self.discount_factor * next_value)

            next_value_table[state[0], state[1]] = value

        self.value_table = next_value_table

    # 현재 가치함수에 대해 탐욕 정책 발전
    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states() :
            if state == [2,2] :
                continue

            value_list = []
            # 반환할 정책 초기화
            result = np.zeros(4, dtype=np.float32)

            # 모든 행동에 대해 [보상 + (감가율 * 다음 상태 가치함수)] 계산
            for index, action in enumerate(self.env.possible_actions) :
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value = reward + self.discount_factor * next_value
                value_list.append(value)

                # 받을 보상이 최대인 행동의 인덱스(최대가 복수라면 모두)를 추출
                max_idx_list = np.argwhere(value_list == np.amax(value_list))
                max_idx_list = max_idx_list.flatten().tolist()
                prob = 1 / len(max_idx_list)

            for idx in max_idx_list :
                result[idx] = prob

            next_policy[state[0], state[1]] = result

        self.policy_table = next_policy

    # 특정 상태에서 정책에 따른 행동을 반환
    def get_action(self, state):
        policy = self.get_policy(state)
        return np.random.choice(4, 1, p=policy)[0]

    # 상태에 따른 정책 반환
    def get_policy(self, state):
        return self.policy_table[state[0], state[1]]

    # 가치함수의 값을 반환
    def get_value(self, state):
        # 소수점 둘째 자리까지만 계산
        return self.value_table[state[0], state[1]]

if __name__ == '__main__':
    env = Env()
    policy_iteration = PolicyIteration(env)
    gridworld = GraphicDisplay(policy_iteration)
    gridworld.mainloop()