import pygame
import random
from collections import namedtuple
import numpy as np
from helper import Direction, find_snake_direction

pygame.init()
font = pygame.font.SysFont('arial', 25)


Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 255, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

GRID_SIZE = 10  # 10x10
BLOCK_SIZE = 20
SPEED = 80  # ajust the speed of the snake to your liking 

reward_green_apple = 800
reward_red_apple = -300
reward_nothing = -2
reward_game_over = -500


class SnakeGameAI:
    def __init__(self, verbose=False, graphique=True, back_function=None):
        self.w = GRID_SIZE * BLOCK_SIZE
        self.h = GRID_SIZE * BLOCK_SIZE
        self.reset()
        self.verbose = verbose
        self.graphique = graphique
        self.back_function = back_function
        self.block_size = BLOCK_SIZE
        if self.graphique:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()

    def _update_ui(self):
        self.display.fill(BLACK)

        # drawing snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # drawing apples
        for food in self.foods:
            color = (0, 255, 0) if food['type'] == 'green' else (255, 0, 0)  # Vert pour les pommes vertes, rouge pour les rouges
            pygame.draw.rect(self.display, color, pygame.Rect(food['position'].x, food['position'].y, BLOCK_SIZE, BLOCK_SIZE))

        # print score
        text = font.render("Score: " + str(round(self.score, 3)), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def reset(self):
        # game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = []
        self._place_initial_food()
        self.frame_iteration = 0

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

    def play_step(self, action, step=False):
        reward = 0
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.back_function is not None:
                    self.back_function()
                pygame.quit()
                quit()

        # Move Snake
        self._move(action, step)
        self.snake.insert(0, self.head)

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

        else:
            # if no eating apple, pop the queue
            reward = reward_nothing
            self.snake.pop()

        # Afficher la vision directionnelle
        if step:
            self.print_snake_vision()
            input("Tap ENTER")

        if self.graphique:
            # Mettre à jour l'interface utilisateur
            self._update_ui()
            self.clock.tick(SPEED)

        return reward, game_over, self.score
    
    def _move(self, action, step):
        directions = [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT]
        self.direction = directions[action.index(1)]  # Map the action to the correct direction

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
            if step:
                print("RIGHT")
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
            if step:
                print("LEFT")
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
            if step:
                print("DOWN")
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            if step:
                print("UP")
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
        distances = np.zeros((NUM_DIRECTIONS, FEATURES), dtype=np.float32)

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

                # Collision avec le corps du serpent
                if Point(x, y) in self.snake:
                    distances[direction_id, 1] = distance  # Distance au corps
                    break

                # Collision avec une pomme verte
                if any(food['position'] == Point(x, y) and food['type'] == 'green' for food in self.foods):
                    distances[direction_id, 2] = distance  # Distance à une pomme verte
                    break

                # Collision avec une pomme rouge
                if any(food['position'] == Point(x, y) and food['type'] == 'red' for food in self.foods):
                    distances[direction_id, 3] = distance  # Distance à une pomme rouge
                    break

        return distances.flatten()


    # def calculate_distances(self):
    #     NUM_DIRECTIONS = 4  # Haut, Bas, Gauche, Droite
    #     FEATURES = 4  # Mur, Corps, Pomme Verte, Pomme Rouge
    #     distances = np.zeros((NUM_DIRECTIONS, FEATURES), dtype=np.float32)

    #     # Position de la tête
    #     head_position = self.head

    #     for direction in MoveTo:
    #         id = direction.id
    #         id = direction.value
    #         dy, dx = direction.direction

    #         y, x = head_position.y, head_position.x
    #         distance = 0

    #         while 0 <= y < self.h and 0 <= x < self.w:
    #             y += dy * BLOCK_SIZE
    #             x += dx * BLOCK_SIZE
    #             distance += 1

    #             # Collision avec un mur
    #             if x < 0 or x >= self.w or y < 0 or y >= self.h:
    #                 distances[id, 0] = distance  # Distance au mur
    #                 break

    #             # Collision avec le corps du serpent
    #             if Point(x, y) in self.snake:
    #                 distances[id, 1] = distance  # Distance au corps
    #                 break

    #             # Collision avec une pomme verte
    #             if any(food['position'] == Point(x, y) and food['type'] == 'green' for food in self.foods):
    #                 distances[id, 2] = distance  # Distance à une pomme verte
    #                 break

    #             # Collision avec une pomme rouge
    #             if any(food['position'] == Point(x, y) and food['type'] == 'red' for food in self.foods):
    #                 distances[id, 3] = distance  # Distance à une pomme rouge
    #                 break

    #     return distances.flatten()