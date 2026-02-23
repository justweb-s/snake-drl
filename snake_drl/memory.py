from __future__ import annotations

from collections import deque

import numpy as np


class PrioritizedReplayMemory:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size: int):
        self.beta = min(1.0, self.beta + self.beta_increment)

        priorities = np.asarray(self.priorities, dtype=np.float64) ** self.alpha
        prob_sum = float(priorities.sum())
        if prob_sum <= 0:
            probabilities = np.ones_like(priorities) / len(priorities)
        else:
            probabilities = priorities / prob_sum

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max() if weights.size else 1.0

        return samples, indices, weights

    def update_priorities(self, indices, errors) -> None:
        for idx, error in zip(indices, errors):
            self.priorities[idx] = float(error) + self.epsilon

    def __len__(self) -> int:
        return len(self.memory)
