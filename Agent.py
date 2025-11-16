import json
import os
import torch
import random
from collections import deque  # data structure to store memory
from model import Linear_QNet, QTrainer  # importing the neural net from step 2
from helper import calc_epsilon
from constantes import MAX_MEMORY, GAMMA, LR, MODEL_FOLDER_PATH


class Agent:
    def __init__(self, name="last_model"):
        self.n_games = 0
        self.epsilon = 0.1
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.exploration_count = 0
        self.exploitation_count = 0
        self.model = Linear_QNet(16, 512, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.name = name

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        current_batch_size = self.adjust_batch_size()
        if len(self.memory) > current_batch_size:
            mini_sample = random.sample(self.memory, current_batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(
            torch.cat(states, dim=0),
            torch.tensor(actions, dtype=torch.long).view(-1, 1),
            torch.tensor(rewards, dtype=torch.float),
            torch.cat(next_states, dim=0).to(torch.float),
            torch.tensor(dones, dtype=torch.bool)
        )

    def train_short_memory(self, state, action, reward, next_state, done):
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(state, dtype=torch.float)
        self.trainer.train_step(
            state,
            torch.tensor(action, dtype=torch.long).view(-1, 1),
            torch.tensor([reward], dtype=torch.float),
            next_state,
            torch.tensor([done], dtype=torch.bool)
        )

    def get_action(self, state, learning=True):
        self.epsilon = calc_epsilon(self.n_games)

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float)

        prediction: torch.Tensor = self.model(state)
        prediction = prediction.squeeze(0)  # Supprime la dimension batch inutile

        # Exploration vs exploitation
        if learning and random.uniform(0, 1) < self.epsilon:
            # print("EXPLORATION")
            self.exploration_count += 1
            choix = random.randint(1, 3)
            move = torch.argsort(prediction, descending=True)[choix].item()
        else:
            # print("EXPLOITATION")
            self.exploitation_count += 1
            move = torch.argmax(prediction).item()
        # print(f"move = {move}")

        return move

    def adjust_batch_size(self):
        # Augmente progressivement la taille du mini-lot avec le nombre de parties
        return min(128 + (self.n_games // 10), 1024)  # Limite à 1024

    def save(self, filename=None):
        if filename is None:
            filename = f"{self.name}"
        file_info = f"{filename}_info.json"
        file_info = os.path.join(MODEL_FOLDER_PATH, file_info)

        # Crée le dossier s'il n'existe pas
        os.makedirs(os.path.dirname(file_info), exist_ok=True)

        filename += ".pth"
        self.model.save(file_name=filename)

        # Sauvegarde des informations dans un fichier JSON
        info = {"n_games": self.n_games}
        with open(file_info, "w") as f:
            json.dump(info, f)

        print(f"Model saved as {filename}, info saved in {file_info}")

    def load(self, filename=None):
        if filename is None:
            filename = self.name
        file_info = f"{filename}_info.json"
        file_info = os.path.join(MODEL_FOLDER_PATH, file_info)

        filename += ".pth"

        # Chargement du nombre de parties
        try:
            with open(file_info, "r") as f:
                info = json.load(f)
                self.n_games = info.get("n_games", 0)
                print(f"Loaded n_games: {self.n_games} from {file_info}")

        except FileNotFoundError:
            print(f"Info file not found[{file_info}], starting with n_games = 0")

        return self.model.load(filename)
