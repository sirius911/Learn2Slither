import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
import random


class Direction(Enum):
    UP = (0, 0, -1)    # (index, dx, dy)
    LEFT = (1, -1, 0)  # (index, dx, dy)
    DOWN = (2, 0, 1)   # (index, dx, dy)
    RIGHT = (3, 1, 0)  # (index, dx, dy)

    def opposite(self):
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.LEFT: Direction.RIGHT,
            Direction.DOWN: Direction.UP,
            Direction.RIGHT: Direction.LEFT
        }
        return opposites[self]

    def __str__(self):
        return self.name

    @property
    def value(self):
        return self._value_[0]  # Retourne uniquement l'indice

    @property
    def vector(self):
        return self._value_[1:]  # Retourne le vecteur (dx, dy)

    @staticmethod
    def vue(direction: "Direction"):
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        current_index = direction.value

        # Calcul des indices pour Gauche, EnFace, et Droite
        left = directions[(current_index - 1) % 4]
        front = directions[current_index]
        right = directions[(current_index + 1) % 4]

        return [left, front, right]

plt.ion()

# Initialisation de la figure globale et des axes
fig, ax1, ax2 = None, None, None

def plot(scores, mean_scores, explorations, exploitations, epsilon_values, title='Training'):
    global fig, ax1, ax2  # Utilisation des variables globales pour la figure et les axes

    if fig is None or ax1 is None or ax2 is None:  # Initialisation unique
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(title)

    # Effacer le contenu des axes sans recréer de figure
    ax1.clear()
    ax2.clear()

    # --- Premier graphique : Scores et Epsilon ---
    ax1.set_title('Scores and Epsilon')
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
        ax1.plot(range(window_size-1, len(scores)), moving_avg, label='Moving Avg (100 Games)', linestyle='--', color='green')

    # Ajouter un axe secondaire pour Epsilon
    if len(epsilon_values) > 0:
        ax3 = ax1.twinx()
        ax3.set_ylabel('Epsilon (%)', color='orange')
        ax3.plot(epsilon_values, label='Epsilon', color='orange', linestyle='--')
        ax3.set_ylim(0, 100)
        ax3.tick_params(axis='y', labelcolor='orange')

    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    # --- Deuxième graphique : Proportions Random/NN ---
    ax2.set_title('Exploration vs Exploitation')
    ax2.set_xlabel('Number of Games')
    ax2.set_ylabel('Proportion (%)')

    # Calcul des proportions
    total_actions = np.array(explorations) + np.array(exploitations)
    with np.errstate(divide='ignore', invalid='ignore'):
        random_percent = np.divide(np.array(explorations), total_actions, where=total_actions != 0) * 100
        nn_percent = np.divide(np.array(exploitations), total_actions, where=total_actions != 0) * 100

    # Tracer les proportions
    ax2.plot(random_percent, label='Random (%)', color='red')
    ax2.plot(nn_percent, label='NN (%)', color='green')
    ax2.set_ylim(0, 100)

    # Annotations des dernières proportions
    if len(random_percent) > 0:
        ax2.text(len(random_percent)-1, random_percent[-1], f"{random_percent[-1]:.2f}%", color="red")
    if len(nn_percent) > 0:
        ax2.text(len(nn_percent)-1, nn_percent[-1], f"{nn_percent[-1]:.2f}%", color="green")

    ax2.legend()
    ax2.grid(alpha=0.3)

    # Mettre à jour la disposition et afficher
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.pause(0.1)

def find_snake_direction(liste):
    # Convertir la liste en tableau 12x12
    tableau = np.array(liste).reshape(12, 12)

    # Trouver les positions de la tête (valeur 2) et des queues (valeur 3)
    head_pos = np.argwhere(tableau == 2)
    tail_pos = np.argwhere(tableau == 3)

    if len(head_pos) == 0 or len(tail_pos) == 0:
        raise ValueError("Le tableau ne contient pas de tête (2) ou de queue (3).")

    # La tête et la queue sont des tableaux avec une seule position
    head_pos = head_pos[0]  # Exemple : [i, j] pour la tête
    tail_pos = tail_pos[0]  # Exemple : [i, j] pour la queue

    # Calculer la direction en fonction des coordonnées
    if head_pos[0] == tail_pos[0]:  # Même ligne
        if head_pos[1] < tail_pos[1]:
            return Direction.LEFT
        else:
            return Direction.RIGHT
    elif head_pos[1] == tail_pos[1]:  # Même colonne
        if head_pos[0] < tail_pos[0]:
            return Direction.UP
        else:
            return Direction.DOWN
    else:
        raise ValueError("La tête et la queue ne sont pas alignées correctement")


def get_state(game):
    # Calculer les distances
    distances = game.calculate_distances()

    # Encoder les positions
    aligned_vision = game.get_snake_aligned_vision()
    positions = []

    for row in aligned_vision:
        for cell in row:
            if cell == " ":
                continue
            elif cell == "W":
                positions.append(1)
            elif cell == "H":
                positions.append(2)
            elif cell == "S":
                positions.append(3)
            elif cell == "G":
                positions.append(4)
            elif cell == "R":
                positions.append(5)
            elif cell == "0":
                positions.append(0)

    # Vérifie que positions contient bien 23 éléments
    positions = np.array(positions, dtype=np.float32)
    if len(positions) < 23:
        positions = np.pad(positions, (0, 23 - len(positions)), constant_values=0)

    # Combiner distances et positions
    state = np.concatenate([distances, positions], axis=0)

    return state


def check_adjacent_values(liste, direction):
    # Convertir la liste en tableau 12x12
    tableau = np.array(liste).reshape(12, 12)

    # Trouver la position de la tête (valeur 2)
    head_pos = np.argwhere(tableau == 2)

    if len(head_pos) == 0:
        raise ValueError("Le tableau ne contient pas de tête (2).")

    head_pos = head_pos[0]  # Exemple : [i, j] pour la tête
    i, j = head_pos

    # Déterminer les positions adjacentes selon la direction
    if direction == Direction.RIGHT:
        left = tableau[i - 1, j] if i - 1 >= 0 else None
        right = tableau[i + 1, j] if i + 1 < 12 else None
        forward = tableau[i, j + 1] if j + 1 < 12 else None
    elif direction == Direction.LEFT:
        left = tableau[i + 1, j] if i + 1 < 12 else None
        right = tableau[i - 1, j] if i - 1 >= 0 else None
        forward = tableau[i, j - 1] if j - 1 >= 0 else None
    elif direction == Direction.UP:
        left = tableau[i, j - 1] if j - 1 >= 0 else None
        right = tableau[i, j + 1] if j + 1 < 12 else None
        forward = tableau[i - 1, j] if i - 1 >= 0 else None
    elif direction == Direction.DOWN:
        left = tableau[i, j + 1] if j + 1 < 12 else None
        right = tableau[i, j - 1] if j - 1 >= 0 else None
        forward = tableau[i + 1, j] if i + 1 < 12 else None
    else:
        raise ValueError("Direction invalide.")
    return (decide_next_move(int(forward), int(right), int(left)))

def decide_next_move(forward, right, left):
    # Priorité pour la case contenant 4
    if 4 in (forward, right, left):
        if forward == 4:
            return 0  
        elif right == 4:
            return 1
        elif left == 4:
            return 2

    return random.randint(0, 2)


def get_snake_direction(game):
    if len(game.snake) < 2:
        raise ValueError("Le serpent doit avoir au moins deux segments pour déterminer la direction.")

    head = game.snake[0]  # Position de la tête
    next_segment = game.snake[1]  # Position du segment suivant

    # Calcul de la différence de position
    dy = head.y - next_segment.y
    dx = head.x - next_segment.x

    # Mapping des directions vers des indices
    if dy == -game.block_size and dx == 0:  # Haut
        return Direction.UP
    elif dy == 0 and dx == game.block_size:  # Gauche
        return Direction.LEFT
    elif dy == game.block_size and dx == 0:  # Bas
        return Direction.DOWN
    elif dy == 0 and dx == -game.block_size:  # Droite
        return Direction.RIGHT
    else:
        return Direction.Up

def absolute_to_relative_action(current_direction, absolute_action):
    print(f"'{current_direction}', '{absolute_action}' -> ", end='')
    absolute_moves = {
        Direction.UP: [Direction.LEFT, Direction.UP, Direction.RIGHT],  # UP -> [LEFT, STRAIGHT, RIGHT]
        Direction.LEFT: [Direction.DOWN, Direction.LEFT, Direction.UP],  # LEFT -> [LEFT, STRAIGHT, RIGHT]
        Direction.DOWN: [Direction.RIGHT, Direction.DOWN, Direction.LEFT],  # DOWN -> [LEFT, STRAIGHT, RIGHT]
        Direction.RIGHT: [Direction.UP, Direction.RIGHT, Direction.DOWN],  # RIGHT -> [LEFT, STRAIGHT, RIGHT]
    }
    if absolute_action not in absolute_moves[current_direction]:
        move = absolute_action.value
    else:
        move = absolute_moves[current_direction].index(absolute_action)
        # raise ValueError(f"Action absolue {absolute_action} invalide pour direction {current_direction}")
    print(Direction(move))
    return move

def calc_epsilon(n_games):
    # Epsilon décroît de 1% tous les 200 jeux, avec un minimum de 0.01%
    epsilon = max(0.01, 80 - (n_games // 200))
    return epsilon

def calculate_proportions(n_games):
    epsilon = calc_epsilon(n_games)

    # Proportions
    proportion_random = epsilon / 100
    proportion_nn = 1 - proportion_random

    return proportion_random, proportion_nn
