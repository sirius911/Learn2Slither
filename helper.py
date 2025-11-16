import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from directions import Direction
import pandas as pd

plt.ion()

# Initialisation de la figure globale et des axes
fig, ax1, ax2, ax3 = None, None, None, None


def plot(scores, mean_scores, epsilon_values, learning):
    global fig, ax1, ax2, ax3  # Variables globales pour la figure et les axes

    title = ('Training' if learning else 'Gaming')

    # Initialisation unique de la figure avec 3 sous-graphiques
    if fig is None:
        # if learning:
        #     fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        #     ax1, ax2, ax3 = axes
        # else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 12))
        ax1 = axes
        fig.suptitle(title)

    # # Effacer le contenu des axes sans recréer la figure
    # if learning:
    #     ax1.clear()
    #     ax2.clear()
    #     ax3.clear()  # Toujours nettoyé, mais n'est pas utilisé si `learning=False`
    # else:
    ax1.clear()

    # --- Premier graphique : Scores et Epsilon ---
    ax1.set_title('Scores')
    ax1.set_xlabel('Number of Games')
    ax1.set_ylabel('Score')

    # Calcul de la moyenne glissante des scores sur 100 jeux
    window_size = 100
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
    else:
        moving_avg = []

    # Tracer les scores individuels
    ax1.plot(scores, label='Scores', alpha=0.5, color='blue')

    # Tracer la moyenne cumulative des scores
    ax1.plot(mean_scores, label='Mean Scores', color='orange')

    # Tracer la moyenne glissante sur 100 jeux
    if len(moving_avg) > 0:
        ax1.plot(
            range(window_size-1, len(scores)),
            moving_avg,
            label='Moving Avg (100 Games)',
            linestyle='--',
            color='green')

    # Ajouter un axe secondaire pour Epsilon
    if learning:
        if len(epsilon_values) > 0 and learning:
            ax4 = ax1.twinx()
            ax4.set_ylabel('Epsilon (%)', color='orange')
            ax4.plot(epsilon_values, label='Epsilon', color='orange', linestyle='--')
            ax4.set_ylim(0, 10)
            ax4.tick_params(axis='y', labelcolor='orange')

    # Annotations des dernières valeurs
    if len(scores) > 0:
        ax1.text(len(scores)-1, scores[-1], f"{scores[-1]:.2f}", color="blue")
    if len(mean_scores) > 0:
        ax1.text(len(mean_scores)-1, mean_scores[-1], f"{mean_scores[-1]:.2f}", color="orange")
    if len(moving_avg) > 0:
        ax1.text(len(moving_avg) + window_size - 2, moving_avg[-1], f"{moving_avg[-1]:.2f}", color="green")
    if len(epsilon_values) > 0:
        ax1.text(len(epsilon_values)-1, epsilon_values[-1], f"{epsilon_values[-1]:.2f}", color="orange")

    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    # Mettre à jour la disposition et afficher
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.subplots_adjust(top=0.9, bottom=0.1)  # Ajustez les valeurs selon vos besoins
    plt.pause(0.1)


def print_tensor(tensor: torch.Tensor):
    index = [dir.name for dir in Direction.directions()]
    df = pd.DataFrame(tensor.numpy().reshape(4, 4),
                      index=index,
                      columns=["Wall", "Danger", "Apple_Green", "Apple_Red"])
    print(df)


# def find_snake_direction(liste):
#     # Convertir la liste en tableau 12x12
#     tableau = np.array(liste).reshape(12, 12)

#     # Trouver les positions de la tête (valeur 2) et des queues (valeur 3)
#     head_pos = np.argwhere(tableau == 2)
#     tail_pos = np.argwhere(tableau == 3)

#     if len(head_pos) == 0 or len(tail_pos) == 0:
#         raise ValueError("Le tableau ne contient pas de tête (2) ou de queue (3).")

#     # La tête et la queue sont des tableaux avec une seule position
#     head_pos = head_pos[0]  # Exemple : [i, j] pour la tête
#     tail_pos = tail_pos[0]  # Exemple : [i, j] pour la queue

#     # Calculer la direction en fonction des coordonnées
#     if head_pos[0] == tail_pos[0]:  # Même ligne
#         if head_pos[1] < tail_pos[1]:
#             return Direction.LEFT
#         else:
#             return Direction.RIGHT
#     elif head_pos[1] == tail_pos[1]:  # Même colonne
#         if head_pos[0] < tail_pos[0]:
#             return Direction.UP
#         else:
#             return Direction.DOWN
#     else:
#         raise ValueError("La tête et la queue ne sont pas alignées correctement")


def calc_epsilon(n_games):
    # epsilon = max(0.01, 80 - (n_games // 200))
    epsilon = 0.099 * math.exp(-n_games / 1000)
    return epsilon

# def calculate_proportions(n_games):
#     epsilon = calc_epsilon(n_games)

#     # Proportions
#     proportion_random = epsilon / 100
#     proportion_nn = 1 - proportion_random

#     return proportion_random, proportion_nn
