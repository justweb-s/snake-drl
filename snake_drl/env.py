from __future__ import annotations

import random

import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class SnakeGame:
    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        obstacles: bool = False,
        curriculum_level: int = 0,
        max_steps_factor: int = 100,
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.use_obstacles = obstacles
        self.curriculum_level = curriculum_level
        self.max_steps_factor = max_steps_factor

        self.obstacles: list[tuple[int, int]] = []
        self.reset()

    def reset(self):
        self.snake = [
            (
                random.randint(3, self.grid_width - 4),
                random.randint(3, self.grid_height - 4),
            )
        ]
        self.direction = random.randint(0, 3)
        self.score = 0
        self.steps_without_food = 0
        self.max_steps_without_food = self.max_steps_factor * max(1, len(self.snake))

        self.obstacles = []
        if self.use_obstacles:
            num_obstacles = 5 + self.curriculum_level * 2
            for _ in range(num_obstacles):
                self.add_obstacle()

        self.food = self.generate_food()
        self.game_over = False
        return self.get_state()

    def add_obstacle(self) -> None:
        for _ in range(10):
            obstacle = (
                random.randint(0, self.grid_width - 1),
                random.randint(0, self.grid_height - 1),
            )
            if obstacle in self.snake or obstacle in self.obstacles:
                continue

            head_x, head_y = self.snake[0]
            if abs(obstacle[0] - head_x) > 2 or abs(obstacle[1] - head_y) > 2:
                self.obstacles.append(obstacle)
                break

    def generate_food(self):
        valid_positions = []
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                pos = (x, y)
                if pos not in self.snake and pos not in self.obstacles:
                    valid_positions.append(pos)

        if not valid_positions:
            self.game_over = True
            return (0, 0)

        return random.choice(valid_positions)

    def is_collision(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        if (x, y) in self.snake[1:]:
            return True
        if (x, y) in self.obstacles:
            return True
        return False

    def get_state(self) -> np.ndarray:
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        danger_n = self.is_collision(head_x, head_y - 1)
        danger_ne = self.is_collision(head_x + 1, head_y - 1)
        danger_e = self.is_collision(head_x + 1, head_y)
        danger_se = self.is_collision(head_x + 1, head_y + 1)
        danger_s = self.is_collision(head_x, head_y + 1)
        danger_sw = self.is_collision(head_x - 1, head_y + 1)
        danger_w = self.is_collision(head_x - 1, head_y)
        danger_nw = self.is_collision(head_x - 1, head_y - 1)

        dir_up = self.direction == UP
        dir_right = self.direction == RIGHT
        dir_down = self.direction == DOWN
        dir_left = self.direction == LEFT

        food_up = food_y < head_y
        food_right = food_x > head_x
        food_down = food_y > head_y
        food_left = food_x < head_x

        food_dist_x = (food_x - head_x) / self.grid_width
        food_dist_y = (food_y - head_y) / self.grid_height

        norm_head_x = head_x / self.grid_width
        norm_head_y = head_y / self.grid_height

        norm_length = len(self.snake) / (self.grid_width * self.grid_height)

        nearest_obstacle_dist = 1.0
        if self.obstacles:
            distances = [abs(ob[0] - head_x) + abs(ob[1] - head_y) for ob in self.obstacles]
            nearest_obstacle_dist = min(distances) / (self.grid_width + self.grid_height)

        state = [
            danger_n,
            danger_ne,
            danger_e,
            danger_se,
            danger_s,
            danger_sw,
            danger_w,
            danger_nw,
            dir_up,
            dir_right,
            dir_down,
            dir_left,
            food_up,
            food_right,
            food_down,
            food_left,
            food_dist_x,
            food_dist_y,
            norm_head_x,
            norm_head_y,
            norm_length,
            nearest_obstacle_dist,
        ]

        return np.asarray(state, dtype=np.float32)

    def step(self, action: int):
        if action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction - 1) % 4

        head_x, head_y = self.snake[0]

        if self.direction == UP:
            head_y -= 1
        elif self.direction == RIGHT:
            head_x += 1
        elif self.direction == DOWN:
            head_y += 1
        elif self.direction == LEFT:
            head_x -= 1

        reward = 0.0
        if self.is_collision(head_x, head_y):
            self.game_over = True
            reward = -10.0
            return self.get_state(), reward, True

        new_head = (head_x, head_y)
        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self.steps_without_food = 0
            self.max_steps_without_food = self.max_steps_factor * max(1, len(self.snake))
            self.food = self.generate_food()
        else:
            self.snake.pop()
            self.steps_without_food += 1

        if self.steps_without_food > self.max_steps_without_food:
            self.game_over = True
            reward = -5.0
            return self.get_state(), reward, True

        food_x, food_y = self.food
        prev_head_x, prev_head_y = self.snake[1] if len(self.snake) > 1 else (head_x, head_y)
        prev_dist = abs(prev_head_x - food_x) + abs(prev_head_y - food_y)
        curr_dist = abs(head_x - food_x) + abs(head_y - food_y)

        if curr_dist < prev_dist:
            reward += 0.05

        reward -= 0.01

        return self.get_state(), reward, self.game_over

    def render(self, screen, cell_size: int = 20, draw_grid: bool = True):
        import pygame

        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (255, 0, 0)
        green = (0, 255, 0)
        gray = (128, 128, 128)

        screen.fill(black)

        for obstacle in self.obstacles:
            obstacle_rect = pygame.Rect(
                obstacle[0] * cell_size,
                obstacle[1] * cell_size,
                cell_size,
                cell_size,
            )
            pygame.draw.rect(screen, gray, obstacle_rect)

        food_rect = pygame.Rect(
            self.food[0] * cell_size,
            self.food[1] * cell_size,
            cell_size,
            cell_size,
        )
        pygame.draw.rect(screen, red, food_rect)

        for i, segment in enumerate(self.snake):
            segment_rect = pygame.Rect(
                segment[0] * cell_size,
                segment[1] * cell_size,
                cell_size,
                cell_size,
            )
            color = (0, 200, 0) if i == 0 else green
            pygame.draw.rect(screen, color, segment_rect)

        if draw_grid:
            width = self.grid_width * cell_size
            height = self.grid_height * cell_size
            for x in range(0, width, cell_size):
                pygame.draw.line(screen, (30, 30, 30), (x, 0), (x, height))
            for y in range(0, height, cell_size):
                pygame.draw.line(screen, (30, 30, 30), (0, y), (width, y))

        return white
