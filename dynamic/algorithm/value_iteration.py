import numpy as np
from environment import Env, GraphicDisplay

class ValueIteration :
    def __init__(self, env):
        self.env = env
        self.value_table = np.zeros((env.height, env.width), dtype=np.float32)
        self.discount_factor = 0.9

    def value_iteration(self):
        next_value_table = np.zeros((self.env.height, self.env.width), dtype=np.float32)

        for state in self.env.get_all_states() :
            if state==[2,2]:
                next_value_table[state[0], state[1]] = 0.0
                continue

            for action in self.env.possible_actions :
                next_state
