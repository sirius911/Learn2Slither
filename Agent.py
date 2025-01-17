import torch
import random
import numpy as np
from collections import deque  # data structure to store memory
from game import SnakeGameAI, Point  # importing the game created in step 1
from model import Linear_QNet, QTrainer  # importing the neural net from step 2
from game import BLOCK_SIZE
from helper import Direction, get_snake_direction, calc_epsilon

MAX_MEMORY = 100_000
BATCH_SIZE = 128
LR = 0.001  # learning rate


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.99  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.exploration_count = 0
        self.exploitation_count = 0
        self.BATCH_SIZE = BATCH_SIZE
        # input size, hidden size, output size
        self.model = Linear_QNet(39, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached
        # if len(self.memory) == 1 or len(self.memory) % 100 == 0:  # Afficher au début et toutes les 100 transitions
        #     print(f"Sample memory: {self.memory[-1]}")

    def train_long_memory(self):
        current_batch_size = self.adjust_batch_size()  # Taille dynamique
        if len(self.memory) > current_batch_size:
            mini_sample = random.sample(self.memory, current_batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, game, state, learning=True):
        # self.epsilon = 80 - self.n_games
        self.epsilon = calc_epsilon(self.n_games)
        # self.epsilon = max(0.05, 500 - (self.n_games * 2))  # Réduction mais avec un minimum de 5%
        
        final_move = [0, 0, 0, 0]
        # if learning and random.random() < self.epsilon:  # Comparaison avec une probabilité continue
        if random.randint(0, 100) < self.epsilon:
            current_dir = get_snake_direction(game)
            dir_arriere = Direction.opposite(current_dir)
            while 42:
                move = random.choice(list(Direction))
                if move != dir_arriere:
                    break
            final_move[move.value] = 1
            self.exploration_count += 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            # print(f"State: {state}, Predicted Q-values: {prediction.tolist()}")
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            self.exploitation_count += 1

        return final_move


    def adjust_batch_size(self):
        # Augmente progressivement la taille du mini-lot avec le nombre de parties
        return min(128 + (self.n_games // 10), 512)  # Limite à 512

