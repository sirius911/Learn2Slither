import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from constantes import MODEL_FOLDER_PATH


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # building the input, hidden and output layer
        super(Linear_QNet, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

    def save(self, file_name='model.pth'):
        # saving the model

        file_name = os.path.join(MODEL_FOLDER_PATH, file_name)

        print(f"Sauvegarde de {file_name}", end=" ... ")
        torch.save(self.state_dict(), file_name)
        print("Ok")

    def load(self, file_name='model.pth'):
        try:
            file_name = os.path.join(MODEL_FOLDER_PATH, file_name)
            print(f"Chargement de {file_name}", end=" ... ")
            self.load_state_dict(torch.load(file_name, weights_only=True))
            print("Ok")
            return True
        except Exception as e:
            print(f"ERREUR: {e}")
            return False


class QTrainer:
    def __init__(self, model: Linear_QNet, lr: float, gamma: float):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Effectue un pas d'entraînement avec les états,
        actions, récompenses et états suivants.
        """

        # Prédiction avec le modèle
        pred = self.model(state)

        # Sélection de la valeur Q associée à l'action
        q_value = pred.gather(1, action).squeeze(-1)

        # Calcul de la valeur cible
        with torch.no_grad():
            next_pred = self.model(next_state)
            next_action = torch.max(next_pred, dim=1)[0]  # Max Q-value
            next_q_value = reward + (self.gamma * next_action * (~done))

        # Mise à jour du modèle
        self.optimizer.zero_grad()
        loss = self.loss(q_value, next_q_value)
        loss.backward()
        self.optimizer.step()

        return loss.item()
