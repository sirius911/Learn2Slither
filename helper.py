import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from directions import Direction
import pandas as pd

plt.ion()

# Initialisation de la figure globale et des axes
fig, ax1 = None, None


def plot(scores, mean_scores, epsilon_values, learning):
    global fig, ax1  # Variables globales pour la figure et les axes

    title = ('Training' if learning else 'Gaming')

    # Initialisation unique de la figure avec 3 sous-graphiques
    if fig is None:
        fig, axes = plt.subplots(1, 1, figsize=(10, 12))
        ax1 = axes
        fig.suptitle(title)

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

    plt.pause(0.1)


def print_tensor(tensor: torch.Tensor):
    index = [dir.name for dir in Direction.directions()]
    df = pd.DataFrame(tensor.numpy().reshape(4, 4),
                      index=index,
                      columns=["Wall", "Danger", "Apple_Green", "Apple_Red"])
    print(df)


def calc_epsilon(n_games):
    # epsilon = max(0.01, 80 - (n_games // 200))
    epsilon = 0.099 * math.exp(-n_games / 1000)
    return epsilon
