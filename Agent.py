import json
import os
import torch
import random
from collections import deque  # data structure to store memory
from model import Linear_QNet, QTrainer
from helper import calc_epsilon
from constantes import MAX_MEMORY, GAMMA, LR, MODEL_FOLDER_PATH
try:
    from torchviz import make_dot
except ImportError:
    make_dot = None


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

    def visualize_model(self, filename=None, example_state=None):
        """
        Génère un fichier PNG représentant le graphe du réseau de neurones
        actuel (avec les poids déjà chargés/entraînés).

        Le fichier est sauvegardé dans MODEL_FOLDER_PATH, avec suffixe '_graph.png'.
        """
        if make_dot is None:
            print("torchviz n'est pas installé. Fais : pip install torchviz graphviz")
            return

        # Nom de fichier de base
        if filename is None:
            filename = self.name

        # Base du chemin sans extension
        output_base = os.path.join(MODEL_FOLDER_PATH, f"{filename}_graph")

        # Exemple d'état : soit fourni, soit généré
        if example_state is None:
            first_param = next(self.model.parameters())
            # Pour une couche Linear, weight a la forme (out_features, in_features)
            if first_param.dim() >= 2:
                in_features = first_param.shape[1]
            else:
                raise RuntimeError("Impossible de déduire le nombre de features en entrée.")

            # On crée un batch de 1 avec le bon nombre de features
            example_state = torch.randn(1, in_features, dtype=torch.float32)
        else:
            if not isinstance(example_state, torch.Tensor):
                example_state = torch.tensor(example_state, dtype=torch.float32)
            if example_state.dim() == 1:
                example_state = example_state.unsqueeze(0)

        # Forward pour construire le graphe
        y = self.model(example_state)

        # Création du graphe
        dot = make_dot(y, params=dict(self.model.named_parameters()))

        # Sauvegarde en PNG (dot.render ajoute '.png')
        dot.render(output_base, format="png")

        print(f"Graphe du modèle sauvegardé dans : {output_base}.png")

    def export_onnx(self, filename=None):
        """
        Exporte le modèle actuel au format ONNX pour visualisation (ex : dans Netron).

        Le fichier est sauvegardé dans MODEL_FOLDER_PATH avec l'extension .onnx.
        """
        # Nom de base
        if filename is None:
            filename = self.name

        onnx_path = os.path.join(MODEL_FOLDER_PATH, f"{filename}.onnx")

        # Déduction automatique du nombre de features en entrée
        first_param = next(self.model.parameters())
        if first_param.dim() >= 2:
            in_features = first_param.shape[1]
        else:
            raise RuntimeError("Impossible de déduire le nombre de features en entrée.")

        # Tensor d'entrée fictif : batch de 1
        example_state = torch.randn(1, in_features, dtype=torch.float32)

        # Très important : passer en mode eval pour l'export
        self.model.eval()

        print(f"Export du modèle en ONNX vers {onnx_path} ...", end=" ")

        torch.onnx.export(
            self.model,
            example_state,         # input exemple
            onnx_path,             # chemin du fichier de sortie
            input_names=["state"],  # nom logique de l'entrée
            output_names=["q_values"],  # nom logique de la sortie
            dynamic_axes={         # batch dimension flexible
                "state": {0: "batch_size"},
                "q_values": {0: "batch_size"},
            },
            opset_version=17       # version ONNX raisonnablement récente
        )

        print("OK")
        return onnx_path
