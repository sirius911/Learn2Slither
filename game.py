import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


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


class SnakeGameAI:
    def __init__(self):
        self.w = GRID_SIZE * BLOCK_SIZE
        self.h = GRID_SIZE * BLOCK_SIZE
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

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
        text = font.render("Score: " + str(self.score), True, WHITE)
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

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move Snake
        self._move(action)
        self.snake.insert(0, self.head)

        # control collisions
        reward = -1
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -100
            return reward, game_over, self.score
        
        # controle eating apple
        eaten_food = next((food for food in self.foods if food['position'] == self.head), None)
        if eaten_food:
            if eaten_food['type'] == 'green':
                # positiv reward for green apple
                self.score += 1
                reward = 10
                self.snake.append(self.snake[-1])  # Augmente la longueur
            elif eaten_food['type'] == 'red':
                # penality for red apple
                self.score -= 1
                reward = -5
                if len(self.snake) > 1:
                    self.snake.pop()  # Réduit la longueur
                else:
                    game_over = True
                    reward = -100  # Perte si la longueur tombe à 0

            # Remplace la pomme mangée
            eaten_food['position'] = self._get_random_position()

        else:
            # if no eating apple, pop the queue
            self.snake.pop()

        # Afficher la vision directionnelle
        # self.print_snake_vision()

        # Mettre à jour l'interface utilisateur
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score
    
    def _move(self, action):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]): # straight
            new_dir = clock_wise[idx]  
        elif np.array_equal(action, [0, 1, 0]): #right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  
        else:  #[0,0,1] aka left turn 
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  
        self.direction = new_dir

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

        # Initialiser une grille vide
        aligned_vision = [[" " for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        # Placer les informations visibles dans les 4 directions
        # Direction Nord (haut)
        for y in range(head_y - 1, -1, -1):  # Vers le haut
            aligned_vision[y][head_x] = self._get_grid_content(head_x, y)

        # Direction Sud (bas)
        for y in range(head_y + 1, GRID_SIZE):  # Vers le bas
            aligned_vision[y][head_x] = self._get_grid_content(head_x, y)

        # Direction Est (droite)
        for x in range(head_x + 1, GRID_SIZE):  # Vers la droite
            aligned_vision[head_y][x] = self._get_grid_content(x, head_y)

        # Direction Ouest (gauche)
        for x in range(head_x - 1, -1, -1):  # Vers la gauche
            aligned_vision[head_y][x] = self._get_grid_content(x, head_y)

        # Placer la tête du serpent
        aligned_vision[head_y][head_x] = "H"

        return aligned_vision

    def _get_grid_content(self, grid_x, grid_y):
        # Murs
        if grid_x <= 0 or grid_y <= 0 or grid_x > GRID_SIZE or grid_y > GRID_SIZE:
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
        aligned_vision = self.get_snake_aligned_vision()

        # Afficher ligne par ligne
        for row in aligned_vision:
            print("".join(row))  # Joindre chaque ligne pour former une chaîne
        print("-" * 20)  # Ligne de séparation
