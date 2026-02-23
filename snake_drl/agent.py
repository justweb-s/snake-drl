from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from snake_drl.memory import PrioritizedReplayMemory
from snake_drl.model import DQN


@dataclass
class AgentCheckpoint:
    episode: int
    model_state_dict: dict
    target_model_state_dict: dict
    optimizer_state_dict: dict
    epsilon: float
    avg_score: float
    config: dict


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        dropout_rate: float = 0.2,
        memory_capacity: int = 100_000,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_increment: float = 0.001,
        grad_clip_norm: float = 1.0,
        device: str | None = None,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.grad_clip_norm = grad_clip_norm

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.memory = PrioritizedReplayMemory(
            capacity=memory_capacity,
            alpha=per_alpha,
            beta=per_beta,
            beta_increment=per_beta_increment,
        )

        self.model: nn.Module = DQN(state_size, hidden_size, action_size, dropout_rate=dropout_rate).to(self.device)
        self.target_model: nn.Module = DQN(state_size, hidden_size, action_size, dropout_rate=0.0).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )

        self.recent_scores = deque(maxlen=50)
        self.avg_score = 0.0

    def update_target_model(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() <= self.epsilon:
            return int(np.random.randint(self.action_size))

        self.model.train(mode=training)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def train_step(self, batch_size: int) -> float | None:
        if len(self.memory) < batch_size:
            return None

        batch, indices, weights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        self.model.train(True)
        current_q = self.model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            self.model.eval()
            self.target_model.eval()
            next_actions = self.model(next_states_t).max(1)[1].unsqueeze(1)
            next_q = self.target_model(next_states_t).gather(1, next_actions).squeeze(1)

        target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        loss_vec = F.smooth_l1_loss(current_q, target_q, reduction="none")
        loss = (weights_t * loss_vec).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

    def update_metrics(self, score: int) -> None:
        self.recent_scores.append(score)
        self.avg_score = float(sum(self.recent_scores) / len(self.recent_scores))

    def make_checkpoint(self, episode: int, config: dict) -> AgentCheckpoint:
        return AgentCheckpoint(
            episode=episode,
            model_state_dict=self.model.state_dict(),
            target_model_state_dict=self.target_model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            epsilon=float(self.epsilon),
            avg_score=float(self.avg_score),
            config=dict(config),
        )

    def save(self, path: str, episode: int, config: dict, extra: dict | None = None) -> None:
        ckpt = self.make_checkpoint(episode=episode, config=config)
        payload = dict(ckpt.__dict__)
        if extra:
            payload.update(dict(extra))
        torch.save(payload, path)

    def load(self, path: str) -> dict:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.target_model.load_state_dict(payload.get("target_model_state_dict", payload["model_state_dict"]))
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.epsilon = float(payload.get("epsilon", 0.0))
        self.avg_score = float(payload.get("avg_score", 0.0))
        return payload
