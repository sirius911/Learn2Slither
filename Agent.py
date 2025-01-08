import torch
import random
import numpy as np
from collections import deque  # data structure to store memory
from game import SnakeGameAI, Direction, Point  # importing the game created in step 1
from model import Linear_QNet, QTrainer  # importing the neural net from step 2
from game import BLOCK_SIZE

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # learning rate


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)

        # input size, hidden size, output size
        self.model = Linear_QNet(15, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Détecter les pommes dans les 4 directions
        food_left = any(food['type'] == 'green' and food['position'] == point_l for food in game.foods)
        food_right = any(food['type'] == 'green' and food['position'] == point_r for food in game.foods)
        food_up = any(food['type'] == 'green' and food['position'] == point_u for food in game.foods)
        food_down = any(food['type'] == 'green' and food['position'] == point_d for food in game.foods)

        red_food_left = any(food['type'] == 'red' and food['position'] == point_l for food in game.foods)
        red_food_right = any(food['type'] == 'red' and food['position'] == point_r for food in game.foods)
        red_food_up = any(food['type'] == 'red' and food['position'] == point_u for food in game.foods)
        red_food_down = any(food['type'] == 'red' and food['position'] == point_d for food in game.foods)

        # État du jeu
        state = [
            # Dangers
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),  # Danger droit

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),  # Danger à droite

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),  # Danger à gauche

            # Directions actuelles
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Nourriture verte
            food_left,
            food_right,
            food_up,
            food_down,

            # Nourriture rouge
            red_food_left,
            red_food_right,
            red_food_up,
            red_food_down,
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Ajuster la récompense en fonction des priorités du sujet
        if reward > 0:
            reward *= 1.2  # Priorité sur les récompenses positives
        elif reward < 0:
            reward *= 0.8  # Réduire légèrement les pénalités

        self.trainer.train_step(state, action, reward, next_state, done)

    # def train_short_memory(self, state, action, reward, next_state, done):
    #     self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
