import pygame
import random
from collections import deque, namedtuple
import numpy as np
import torch
from directions import Direction
from constantes import GRID_SIZE, BLOCK_SIZE, SPEED
from constantes import reward_game_over, reward_red_apple, reward_nothing
from constantes import reward_green_apple, reward_infini_boucle
import colors

pygame.init()
font = pygame.font.SysFont('arial', 18)


Point = namedtuple('Point', 'x, y')


class SnakeGameAI:
    def __init__(self, verbose=False, graphique=True, back_function=None):
        self.w = GRID_SIZE * BLOCK_SIZE
        self.h = GRID_SIZE * BLOCK_SIZE
        self.w_window = (GRID_SIZE + 2) * BLOCK_SIZE
        self.h_window = (GRID_SIZE + 2) * BLOCK_SIZE
        self.verbose = verbose
        self.graphique = graphique
        self.back_function = back_function
        self.block_size = BLOCK_SIZE
        if self.graphique:
            self.display = pygame.display.set_mode((self.w_window, self.h_window))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
        self.best_score = 0
        self.nb_game = 0
        self.nb_infini = 0
        self.recent_paths = deque(maxlen=100)  # Stocke les 100 derniers états du serpent
        self.reset()

    def _update_ui(self, temp_surface=None):
        if temp_surface is None:
            surface = self.display
        else:
            surface = temp_surface
        surface.fill(colors.BLACK)

        # Dimensions de la zone jouable (sans les murs)
        play_area_x = BLOCK_SIZE  # Décalage pour les murs
        play_area_y = BLOCK_SIZE
        play_area_w = GRID_SIZE * BLOCK_SIZE
        play_area_h = GRID_SIZE * BLOCK_SIZE

        # Dessiner les murs (ajouter +BLOCK_SIZE à droite et en bas pour bien les voir)
        # Mur haut
        pygame.draw.rect(surface, colors.GRAY,
                         pygame.Rect(0, 0, self.w + 2 * BLOCK_SIZE, BLOCK_SIZE))
        # Mur bas
        pygame.draw.rect(surface, colors.GRAY,
                         pygame.Rect(0, self.h + BLOCK_SIZE, self.w + 2 * BLOCK_SIZE, BLOCK_SIZE))
        # Mur gauche
        pygame.draw.rect(surface, colors.GRAY,
                         pygame.Rect(0, 0, BLOCK_SIZE, self.h + 2 * BLOCK_SIZE))
        # Mur droit
        pygame.draw.rect(surface, colors.GRAY,
                         pygame.Rect(self.w + BLOCK_SIZE, 0, BLOCK_SIZE, self.h + 2 * BLOCK_SIZE))

        # Dessiner la grille de jeu (cases jouables)
        for x in range(play_area_x, play_area_x + play_area_w, BLOCK_SIZE):
            pygame.draw.line(surface, colors.GRAY, (x, play_area_y), (x, play_area_y + play_area_h), 1)
        for y in range(play_area_y, play_area_y + play_area_h, BLOCK_SIZE):
            pygame.draw.line(surface, colors.GRAY, (play_area_x, y), (play_area_x + play_area_w, y), 1)

        # Dessiner le serpent (en tenant compte du décalage des murs)
        for i, pt in enumerate(self.snake[1:]):
            snake_x = pt.x + BLOCK_SIZE  # Décalage pour compenser les murs
            snake_y = pt.y + BLOCK_SIZE
            # if i == 0:  # Tête
            #     self._draw_snake_head_on_surface(surface, Point(snake_x, snake_y))
            # else:
            pygame.draw.rect(surface, colors.BLUE1, pygame.Rect(snake_x, snake_y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(surface, colors.BLUE2, pygame.Rect(snake_x + 4, snake_y + 4, 12, 12))
        # dessine tete:
        snake_x = self.snake[0].x + BLOCK_SIZE
        snake_y = self.snake[0].y + BLOCK_SIZE
        self._draw_snake_head_on_surface(surface, Point(snake_x, snake_y))
        # Dessiner les pommes (elles doivent aussi être décalées)
        for food in self.foods:
            food_x = food['position'].x + BLOCK_SIZE
            food_y = food['position'].y + BLOCK_SIZE
            color = colors.GREEN if food['type'] == 'green' else colors.RED
            pygame.draw.rect(surface, color, pygame.Rect(food_x, food_y, BLOCK_SIZE, BLOCK_SIZE))

        # Afficher le score
        text_score = font.render(f"Score: {round(self.score, 3)}", True, colors.WHITE)
        surface.blit(text_score, (self.w_window - text_score.get_width(), 0))

        # afficher numero game:
        text_nb_game = font.render(f"Game # {self.nb_game + 1}", True, colors.WHITE)
        surface.blit(text_nb_game, [0, 0])

        # afficher best score
        text_best_score = font.render(f"Best score : {self.best_score}", True, colors.WHITE)
        surface.blit(text_best_score, [0, self.h_window - text_best_score.get_height()])
        if temp_surface is None:
            self.surface = surface
            pygame.display.flip()

        return surface

    def reset(self):
        # Placer le serpent
        self._place_snake()
        self.score = 0
        self.food = []
        self._place_initial_food()
        self.frame_iteration = 0
        self.recent_paths = []
        if self.graphique:
            self._update_ui()

    def _place_snake(self):
        # Choisir une orientation aléatoire : horizontal ou vertical
        horizontal = random.choice([True, False])

        # Générer une position aléatoire pour la tête
        if horizontal:
            head_x = random.randint(2, GRID_SIZE - 3) * BLOCK_SIZE  # Assure 2 cases à gauche et à droite
            head_y = random.randint(0, GRID_SIZE - 1) * BLOCK_SIZE  # Toute la hauteur
            self.direction = random.choice([Direction.LEFT, Direction.RIGHT])
        else:
            head_x = random.randint(0, GRID_SIZE - 1) * BLOCK_SIZE  # Toute la largeur
            head_y = random.randint(2, GRID_SIZE - 3) * BLOCK_SIZE  # Assure 2 cases en haut et en bas
            self.direction = random.choice([Direction.UP, Direction.DOWN])

        # Positionner la tête
        self.head = Point(head_x, head_y)

        # Positionner les deux segments suivants en fonction de la direction
        if self.direction == Direction.RIGHT:
            self.snake = [self.head,
                          Point(self.head.x - BLOCK_SIZE, self.head.y),
                          Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]
        elif self.direction == Direction.LEFT:
            self.snake = [self.head,
                          Point(self.head.x + BLOCK_SIZE, self.head.y),
                          Point(self.head.x + 2 * BLOCK_SIZE, self.head.y)]
        elif self.direction == Direction.DOWN:
            self.snake = [self.head,
                          Point(self.head.x, self.head.y - BLOCK_SIZE),
                          Point(self.head.x, self.head.y - 2 * BLOCK_SIZE)]
        elif self.direction == Direction.UP:
            self.snake = [self.head,
                          Point(self.head.x, self.head.y + BLOCK_SIZE),
                          Point(self.head.x, self.head.y + 2 * BLOCK_SIZE)]

    def _place_initial_food(self):
        self.foods = []
        for _ in range(2):  # Deux pommes vertes
            self.foods.append({'type': 'green', 'position': self._get_random_position()})
        self.foods.append({'type': 'red', 'position': self._get_random_position()})  # Une pomme rouge

    def _get_random_position(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        position = Point(x, y)
        # Assure que la pomme n'est pas placée sur le serpent
        while position in self.snake or any(food['position'] == position for food in self.foods):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            position = Point(x, y)
        return position

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        if random.random() < 0.67:  # 67% chance for green apple
            self.food = {'type': 'green', 'position': Point(x, y)}
        else:  # 33% chance for red apple
            self.food = {'type': 'red', 'position': Point(x, y)}

        if self.food['position'] in self.snake:
            self._place_food()

    def is_collision(self, pt=None):
        if pt is None:  # pt is the head of the snake
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True  # if snake hits the side
        if pt in self.snake[1:]:
            return True  # if snake hits itself
        return False

    def play_step(self, action: "Direction", verbose: bool = False):
        reward = 0
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.back_function is not None:
                    self.back_function()
                pygame.quit()
                quit()

        # Move Snake
        self._move(action)
        self.snake.insert(0, self.head)

        # **Mémorisation du chemin récent**
        # Copie de la liste du serpent sous forme de tuple immuable
        current_path = tuple(self.snake[:])
        self.recent_paths.append(current_path)
        # print(f"recent_path = {self.recent_paths}\n")

        # **Détection de boucle infinie**
        if self.detect_loop():
            if verbose:
                print("🚨 Boucle infinie détectée ! 🚨")
            reward = reward_infini_boucle
            return reward, True, self.score

        # control collisions
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = reward_game_over
            return reward, game_over, self.score

        # controle eating apple
        eaten_food = next((food for food in self.foods if food['position'] == self.head), None)
        if eaten_food:
            if eaten_food['type'] == 'green':
                # positiv reward for green apple
                self.score += 1
                reward = reward_green_apple
                self.snake.append(self.snake[-1])  # Augmente la longueur
            elif eaten_food['type'] == 'red':
                # penality for red apple
                self.score -= 1
                reward = reward_red_apple
                if len(self.snake) > 1:
                    self.snake.pop()  # Réduit la longueur
                else:
                    game_over = True
                    reward = reward_game_over  # Perte si la longueur tombe à 0

            # Remplace la pomme mangée
            eaten_food['position'] = self._get_random_position()
            self.recent_paths = []
        else:
            reward = reward_nothing
        # if no eating apple, pop the queue
        self.snake.pop()

        if self.graphique:
            # Mettre à jour l'interface utilisateur
            self._update_ui()
            self.clock.tick(SPEED)

        if self.score < 0:
            self.score = 0
        return reward, game_over, self.score

    def wait(self):
        # 3. Gérer la pause sans bloquer la fenêtre
        waiting = True
        print("Appuyez sur ENTER pour continuer...")
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:  # Vérifie si ENTER est pressé
                        waiting = False
            # Limiter l'utilisation CPU en attendant
            pygame.time.wait(100)

    def detect_loop(self, loop_threshold=4):
        """
        Détecte si le serpent boucle en répétant exactement le même chemin.
        Si une même séquence de positions se répète plusieurs fois, on considère que c'est une boucle infinie.

        loop_threshold : Nombre de répétitions nécessaires pour considérer que c'est une boucle.
        """
        # On mémorise seulement les N derniers mouvements (correspondant à la longueur du serpent)
        recent_moves = self.recent_paths[-len(self.snake):]

        # On compte combien de fois cette séquence s'est répétée dans `recent_path`
        count = sum(1 for i in range(len(self.recent_paths) - len(recent_moves) + 1)
                    if self.recent_paths[i:i+len(recent_moves)] == recent_moves)

        if count >= loop_threshold:
            self.nb_infini += 1
            return True

        return False

    def _move(self, action):
        self.direction = action
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def get_snake_aligned_vision(self):
        # Obtenir les coordonnées de la tête du serpent
        head_x = int(self.head.x // BLOCK_SIZE)
        head_y = int(self.head.y // BLOCK_SIZE)

        # Initialiser une grille vide (incluant les murs aux extrémités)
        aligned_vision = [[" " for _ in range(GRID_SIZE + 2)] for _ in range(GRID_SIZE + 2)]

        # Direction Nord (haut)
        for i in range(1, 12):  # Inclure 10 cases + 1 mur
            y = head_y - i
            if y < 0:  # Mur au Nord
                aligned_vision[0][head_x + 1] = "W"
                break
            aligned_vision[y + 1][head_x + 1] = self._get_grid_content(head_x, y)

        # Direction Sud (bas)
        for i in range(1, 12):  # Inclure 10 cases + 1 mur
            y = head_y + i
            if y >= GRID_SIZE:  # Mur au Sud
                aligned_vision[GRID_SIZE + 1][head_x + 1] = "W"
                break
            aligned_vision[y + 1][head_x + 1] = self._get_grid_content(head_x, y)

        # Direction Est (droite)
        for i in range(1, 12):  # Inclure 10 cases + 1 mur
            x = head_x + i
            if x >= GRID_SIZE:  # Mur à l'Est
                aligned_vision[head_y + 1][GRID_SIZE + 1] = "W"
                break
            aligned_vision[head_y + 1][x + 1] = self._get_grid_content(x, head_y)

        # Direction Ouest (gauche)
        for i in range(1, 12):  # Inclure 10 cases + 1 mur
            x = head_x - i
            if x < 0:  # Mur à l'Ouest
                aligned_vision[head_y + 1][0] = "W"
                break
            aligned_vision[head_y + 1][x + 1] = self._get_grid_content(x, head_y)

        # Placer la tête du serpent
        aligned_vision[head_y + 1][head_x + 1] = "H"

        return aligned_vision

    def _get_grid_content(self, grid_x, grid_y):
        # Murs
        if grid_x < 0 or grid_y < 0 or grid_x >= GRID_SIZE or grid_y >= GRID_SIZE:
            return "W"

        # Pomme
        for food in self.foods:
            if food['position'].x // BLOCK_SIZE == grid_x and food['position'].y // BLOCK_SIZE == grid_y:
                return "G" if food['type'] == 'green' else "R"

        # Serpent (tête ou corps)
        for segment in self.snake:
            if segment.x // BLOCK_SIZE == grid_x and segment.y // BLOCK_SIZE == grid_y:
                return "H" if segment == self.head else "S"

        # Espace vide
        return "0"

    def print_snake_vision(self):
        # for y in range(-1, 11):
        #     for x in range(-1, 11):
        #         print(self._get_grid_content(x,y), end='')
        #     print("")
        aligned_vision = self.get_snake_aligned_vision()

        # Afficher ligne par ligne
        for row in aligned_vision:
            print("".join(row))  # Joindre chaque ligne pour former une chaîne
        print("-" * 20)  # Ligne de séparation

    def calculate_distances(self):
        NUM_DIRECTIONS = len(Direction)  # Haut, Bas, Gauche, Droite
        FEATURES = 4  # Mur, Corps, Pomme Verte, Pomme Rouge
        distances = np.full((NUM_DIRECTIONS, FEATURES), -1, dtype=np.int32)  # Initialisation à -1

        # Position de la tête
        head_position = self.head

        for direction in Direction:
            direction_id = direction.value  # Utilisation de la valeur de l'Enum
            dx, dy = direction.vector  # Récupère (dx, dy) depuis Direction

            y, x = head_position.y, head_position.x
            distance = 0

            while 0 <= y < self.h and 0 <= x < self.w:
                y += dy * BLOCK_SIZE
                x += dx * BLOCK_SIZE
                distance += 1

                # Collision avec un mur
                if x < 0 or x >= self.w or y < 0 or y >= self.h:
                    distances[direction_id, 0] = distance  # Distance au mur
                    break

                # Collision avec le corps du serpent (première occurrence)
                if distances[direction_id, 1] == -1 and Point(x, y) in self.snake:
                    distances[direction_id, 1] = distance

                # Collision avec une pomme verte (première occurrence)
                if distances[direction_id, 2] == -1 and \
                   any(food['position'] == Point(x, y) and food['type'] == 'green' for food in self.foods):
                    distances[direction_id, 2] = distance

                # Collision avec une pomme rouge (première occurrence)
                if distances[direction_id, 3] == -1 and\
                   any(food['position'] == Point(x, y) and food['type'] == 'red' for food in self.foods):
                    distances[direction_id, 3] = distance

                # Si toutes les caractéristiques ont été trouvées, on peut arrêter
                if all(distances[direction_id, :] != -1):
                    break

        # print(f"Distances calculées : {distances}")
        return distances.tolist()  # Retourne une liste de listes

    def is_apple_in_direction(self, direction, color="green"):
        """
        Vérifie s'il y a une pomme verte dans une direction donnée à partir de la tête du serpent.
        :param direction: Une direction (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)
        :return: True s'il y a une pomme verte dans la direction, sinon False
        """
        # Obtenir la position de la tête du serpent
        head_x, head_y = self.head.x, self.head.y

        # Obtenir le vecteur de déplacement pour la direction
        dx, dy = direction.vector

        # Parcourir la grille dans la direction donnée
        x, y = head_x, head_y
        while 0 <= x < self.w and 0 <= y < self.h:  # Tant que l'on reste dans les limites
            x += dx * self.block_size
            y += dy * self.block_size

            # Vérifier s'il y a une pomme verte à cette position
            for food in self.foods:
                if food['type'] == color and food['position'] == Point(x, y):
                    return True  # Pomme verte trouvée

        return False  # Pas de pomme verte trouvée

    def save_map(self, filename="map.png"):
        """
        Sauvegarde l'image actuelle de la carte (avec le serpent, les pommes, etc.) dans un fichier,
        même si le mode graphique est désactivé.
        :param filename: Nom du fichier pour l'image (par défaut 'map.png').
        """
        # Si le mode graphique est activé, utilise la surface existante
        if self.graphique:
            self._update_ui()
            pygame.image.save(self.display, filename)
        else:
            # Crée une surface temporaire pour dessiner la carte
            temp_display = pygame.Surface((self.w_window, self.h_window))
            temp_display = self._update_ui(temp_display)

            # Sauvegarder la surface temporaire
            pygame.image.save(temp_display, filename)
        if self.verbose:
            print(f"Carte sauvegardée sous le nom : {filename}")

    def _draw_snake_head_on_surface(self, surface, head_position):
        """Dessine la tête du serpent sous forme de triangle orienté sur une surface spécifique."""
        if self.direction == Direction.UP:
            # Sommet
            points = [(head_position.x + BLOCK_SIZE // 2, head_position.y),
                      (head_position.x, head_position.y + BLOCK_SIZE),
                      (head_position.x + BLOCK_SIZE, head_position.y + BLOCK_SIZE)]
        elif self.direction == Direction.DOWN:
            points = [(head_position.x + BLOCK_SIZE // 2, head_position.y + BLOCK_SIZE),
                      (head_position.x, head_position.y),
                      (head_position.x + BLOCK_SIZE, head_position.y)]
        elif self.direction == Direction.LEFT:
            points = [(head_position.x, head_position.y + BLOCK_SIZE // 2),  # Sommet
                      (head_position.x + BLOCK_SIZE, head_position.y),       # Haut droit
                      (head_position.x + BLOCK_SIZE, head_position.y + BLOCK_SIZE)]  # Bas droit
        elif self.direction == Direction.RIGHT:
            points = [(head_position.x + BLOCK_SIZE, head_position.y + BLOCK_SIZE // 2),  # Sommet
                      (head_position.x, head_position.y),                               # Haut gauche
                      (head_position.x, head_position.y + BLOCK_SIZE)]                  # Bas gauche

        # Dessiner le triangle représentant la tête
        pygame.draw.polygon(surface, colors.RED, points)

    def get_state(self):
        """
        Retourne un tenseur PyTorch correctement formaté :
        - Forme correcte: `[1, 16]`
        - Type `torch.float32`
        """
        # Liste des directions (UP, LEFT, DOWN, RIGHT)
        directions = [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT]

        # Construire la liste des états
        state_list = []
        for direction in directions:
            state_list.extend([
                self._wall_distance(self.head.x, self.head.y, direction),
                self._danger_distance(self.head.x, self.head.y, direction),
                self._apple_distance(self.head.x, self.head.y, "green", direction),
                self._apple_distance(self.head.x, self.head.y, "red", direction)
            ])

        # Convertir en tenseur PyTorch avec la bonne forme
        state_tensor = torch.tensor(state_list, dtype=torch.float32).unsqueeze(0)

        # print(f"DEBUG: get_state() retourne un tensor de forme {state_tensor.shape}")

        return state_tensor  # Renvoie un `[1, 16]`

    def _wall_distance(self, x, y, direction):
        """
        Retourne la distance normalisée entre la tête du serpent et le mur dans la direction donnée.
        """
        distance = 0
        dx, dy = direction.vector
        while 0 <= x < self.w and 0 <= y < self.h:
            x += dx * self.block_size
            y += dy * self.block_size
            distance += 1

        return self._normalize(distance)

    def _danger_distance(self, x, y, direction):
        """
        Retourne la distance entre la tête du serpent
        et un danger (mur, corps,
        ou pomme rouge si la taille du serpent est 1).
        """
        distance = 0
        dx, dy = direction.vector
        no_snake_body = len(self.snake) == 1  # True si le serpent est de taille 1

        while 0 <= x < self.w and 0 <= y < self.h:
            x += dx * self.block_size
            y += dy * self.block_size
            distance += 1

            # Collision avec un mur
            if x < 0 or x >= self.w or y < 0 or y >= self.h:
                return self._normalize(distance)  # Distance normalisée

            # Collision avec le corps du serpent
            if Point(x, y) in self.snake:
                return self._normalize(distance)

            # Si le serpent est de taille 1, considérer les pommes rouges comme danger
            if no_snake_body and any(food['position'] == Point(x, y) and food['type'] == "red" for food in self.foods):
                return self._normalize(distance)

        return 1.0  # Aucune collision détectée

    def _apple_distance(self, x, y, apple_type, direction):
        """
        Retourne la distance entre la tête du serpent et la pomme de type spécifié dans la direction donnée.
        """
        distance = 0
        dx, dy = direction.vector
        while 0 <= x < self.w and 0 <= y < self.h:
            x += dx * self.block_size
            y += dy * self.block_size
            distance += 1
            if any(food['position'] == Point(x, y) and food['type'] == apple_type for food in self.foods):
                return self._normalize(distance)
        return 1.0  # Aucune pomme trouvée

    def _normalize(self, distance):
        """
        Normalise une distance en la divisant par la plus grande distance possible.
        """
        return distance / (GRID_SIZE - 1)
