import tkinter as tk
import time
import os
import numpy as np
import random
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 100 # 픽셀 수
HEIGHT = 5 # 그리드월드 세로
WIDTH = 5 # 그리드월드 가로
TRANSITION_PROB = 1
POSSIBLE_ACTIONS = [0,1,2,3] # 상, 하, 좌, 우
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 좌표로 나타낸 행동
REWARDS = []

class GraphicDisplay(tk.Tk) :
    def __init__(self, agent):
        super().__init__()
        self.title('Value Iteration')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT + 50))
        self.texts = []
        self.arrows = []
        self.env = Env()
        self.agent = agent
        self.iter = 0
        self.imp_cnt = 0
        self.is_moving = 0
        (self.up, self.down, self.left, self.right), self.shapes = self.load_images()
        self.canvas = self._build_canvas()

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)

        calc_btn = tk.Button(self, text='Calculate', command=self.calculate)
        calc_btn.configure(width=10, activebackground='#33B5E5')
        canvas.create_window(WIDTH * UNIT * 0.13, (HEIGHT * UNIT) + 10, window=calc_btn)

        policy_btn = tk.Button(self, text='Print', command=self.print_optimal_policy)
        policy_btn.configure(width=10, activebackground='#33B5E5')
        canvas.create_window(WIDTH * UNIT * 0.37, (HEIGHT * UNIT) * 10, window=policy_btn)

        move_btn = tk.Button(self, text='Move', command=self.move)
        move_btn.configure(width=10, activebackground='#33B5E5')
        canvas.create_window(WIDTH * UNIT * 0.62, (HEIGHT * UNIT) * 10, window=move_btn)

        clear_btn = tk.Button(self, text='Clear', command=self.clear)
        clear_btn.configure(width=10, activebackground='#33B5E5')
        canvas.create_window(WIDTH * UNIT * 0.87, (HEIGHT * UNIT) * 10, window=clear_btn)

        # 그리드 생성
        for col in range(0, WIDTH * UNIT, UNIT) :
            x0, y0, x1, y1 = col, 0, col, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for row in range(0, HEIGHT * UNIT, UNIT) :
            x0, y0, x1, y1 = 0, row, HEIGHT * UNIT, row
            canvas.create_line(x0, y0, x1, y1)

        # 캔버스에 이미지 추가
        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])
        canvas.create_image(250, 150, image=self.shapes[1])
        canvas.create_image(150, 250, image=self.shapes[1])
        canvas.create_image(250, 250, image=self.shapes[2])

        canvas.pack()

        return canvas

    def load_images(self):
        PhotoImage = ImageTk.PhotoImage
        base_dir = os.path.dirname(__file__)
        up = PhotoImage(Image.open(os.path.join(base_dir, 'img/up.png')).resize((13, 13)))
        down = PhotoImage(Image.open(os.path.join(base_dir, 'img/down.png')).resize((13, 13)))
        left = PhotoImage(Image.open(os.path.join(base_dir, 'img/left.png')).resize((13, 13)))
        right = PhotoImage(Image.open(os.path.join(base_dir, 'img/right.png')).resize((13, 13)))

        rectangle = PhotoImage(Image.open(os.path.join(base_dir, 'img/rectangle.png')).resize((65, 65)))
        triangle = PhotoImage(Image.open(os.path.join(base_dir, 'img/triangle.png')).resize((65, 65)))
        circle = PhotoImage(Image.open(os.path.join(base_dir, 'img/circle.png')).resize((65, 65)))

        return (up, down, left, right), (rectangle, triangle, circle)

    def clear(self):

        if self.is_moving == 0:
            self.iter = 0
            self.imp_cnt = 0
            for i in self.texts :
                self.canvas.delete(i)
            for i in self.arrows :
                self.canvas.delete(i)

            self.agent.reset()

            x, y = self.canvas.coords(self.rectangle)
            self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rectangle)
        return self.canvas.coords(self.rectangle)

    def text_value(self, row, col, contents, font='Helvetica', size=12, style='normal', anchor='nw'):
        origin_x, origin_y = 85, 70
        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill='black', text=contents, font=font, anchor=anchor)
        return self.texts.append(text)

    def text_reward(self, row, col, contents, font='Helvetica', size=12, style='normal', anchor='nw'):
        origin_x, origin_y = 5,5
        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.creete_text(x, y, fill='black', text=contents, font=font, anchor=anchor)
        return self.texts.append(text)

    def rectangle_move(self, action) :
        base_action = np.array([0,0])
        location = self.find_rectangle()
        self.render()
        if action == 0 and location[0] > 0 : # up
            base_action[1] -= UNIT
        elif action == 1 and location[0] < HEIGHT - 1 : # down
            base_action[1] += UNIT
        elif action == 2 and location[1] > 0 : # left
            base_action[0] -= UNIT
        elif action == 3 and location[1] < WIDTH - 1 : # right
            base_action[0] += UNIT

        self.canvas.move(self.rectangle, base_action[0], base_action[1]) # move agent

    def find_rectangle(self):
        temp = self.canvas.coords(self.rectangle)
        x = (temp[0] / 100) - 0.5
        y = (temp[1] / 100) - 0.5
        return int(y), int(x)

    def move(self):
        if self.imp_cnt != 0 and self.is_moving != 1 :
            self.is_moving = 1
            x, y = self.canvas.coords(self.rectangle)
            self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)

            x, y = self.find_rectangle()
            while len(self.agent.get_action([x, y])) != 0 :
                action = random.sample(self.agent.get_action([x, y]), 1)[0]
                self.after(100, self.rectangle_move(action))
                x, y = self.find_rectangle()
            self.is_moving = 0

    def draw_one_arrow(self, col, row, action):
        if col == 2 and row == 2:
            return
        if action == 0 : # up
            origin_x, origin_y = 50 + (UNIT * row), 10 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.up))

        elif action == 1 : # down
            origin_x, origin_y = 50 + (UNIT * row), 90 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.down))

        elif action == 2 : # right
            origin_x, origin_y = 90 + (UNIT * row), 50 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.right))

        elif action == 3:  # left
            origin_x, origin_y = 10 + (UNIT * row), 50 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.left))

    def draw_from_values(self, state, action_list):
        for action in action_list :
            self.draw_one_arrow(*state, action)

    def print_values(self, values):
        for i in range(WIDTH) :
            for j in range(HEIGHT) :
                self.text_value(i, j, round(values[i][j], 2))

    def render(self):
        time.sleep(0.1)
        self.canvas.tag_raise(self.rectangle)
        self.update()

    def calculate(self):
        self.iter += 1
        for i in self.texts :
            self.canvas.delete(i)
        self.agent.develop()
        self.print_values(self.agent.value_table)

    def print_optimal_policy(self):
        self.imp_cnt += 1
        for i in self.arrows :
            self.canvas.delete(i)
        for state in self.env.get_all_states() :
            action = self.agent.get_action(state)
            self.draw_from_values(state, action)



class Env :
    def __init__(self):
        self.trans_prob = TRANSITION_PROB
        self.width = WIDTH
        self.height = HEIGHT
        self.reward = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        self.possible_actions = POSSIBLE_ACTIONS
        self.reward[2, 2] = 1  # reward 1 for circle
        self.reward[1, 2] = -1  # reward -1 for triangle
        self.reward[2, 1] = -1  # reward -1 for triangle
        self.all_state = [[x, y] for y in range(HEIGHT) for x in range(WIDTH)]

    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)
        return self.reward[next_state[0], next_state[1]]

    def state_after_action(self, state, action_index):
        action = ACTIONS[action_index]
        return self.check_boundary([state[0] + action[0], state[1] + action[1]])

    @staticmethod
    def check_boundary(state):
        state[0] = (0 if state[0] < 0 else WIDTH - 1
                        if state[0] > WIDTH - 1 else state[0])

        state[1] = (0 if state[1] < 0 else HEIGHT - 1
                        if state[1] > HEIGHT - 1 else state[1])

        return state

    def get_transition_prob(self, state, action):
        return self.trans_prob

    def get_all_states(self):
        return self.all_state