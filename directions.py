from enum import Enum


class RelativeMove(Enum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2
    BACK = 3

    def __str__(self):
        return self.name


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
        directions = Direction.directions()
        current_index = direction.value

        # Calcul des indices pour Gauche, EnFace, Droite et Arrière
        right = directions[(current_index - 1) % 4]
        front = directions[current_index]
        left = directions[(current_index + 1) % 4]
        back = directions[(current_index + 2) % 4]  # Demi-tour

        return [left, front, right, back]  # Ajout de `back`

    @classmethod
    def directions(cls):
        return [cls.UP, cls.LEFT, cls.DOWN, cls.RIGHT]

    @classmethod
    def relative_direction(cls, current_direction: "Direction", absolute_move: "Direction") -> "RelativeMove":
        """
        Convertit une action absolue (UP, LEFT, DOWN, RIGHT) en action relative
        (LEFT, STRAIGHT, RIGHT, BACK) en fonction de la direction actuelle du serpent.
        """
        relative_moves = cls.vue(current_direction)  # Liste [LEFT, STRAIGHT, RIGHT, BACK]

        if absolute_move not in relative_moves:
            raise ValueError(f"L'action absolue {absolute_move}\
                 n'est pas valide pour la direction actuelle {current_direction}")

        return RelativeMove(relative_moves.index(absolute_move))  # Renvoie un `RelativeMove`

    @classmethod
    def absolute_direction(cls, current_direction: "Direction", relative_move: list[int]):
        """
        Retourne la direction absolue en fonction de la direction actuelle
        et d'un mouvement relatif (liste [left, straight, right]).
        """
        if len(relative_move) != 3 or sum(relative_move) != 1:
            raise ValueError("relative_move doit être une liste de longueur 3 avec un seul élément égal à 1.")

        directions = cls.directions()
        current_index = directions.index(current_direction)

        # Trouver l'index relatif (LEFT = -1, STRAIGHT = 0, RIGHT = +1)
        relative_index = relative_move.index(1) - 1  # [1, 0, 0] -> -1, [0, 1, 0] -> 0, [0, 0, 1] -> +1

        # Calculer l'index absolu
        absolute_index = (current_index + relative_index) % 4

        return directions[absolute_index]
