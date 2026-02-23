import numpy as np

from snake_drl.agent import DQNAgent
from snake_drl.env import SnakeGame


def test_env_state_shape():
    env = SnakeGame(grid_width=10, grid_height=10, obstacles=False)
    state = env.reset()
    assert state.shape == (22,)
    assert state.dtype == np.float32


def test_agent_act_and_train_step_smoke():
    env = SnakeGame(grid_width=10, grid_height=10, obstacles=False)
    state = env.reset()

    agent = DQNAgent(state_size=state.shape[0], action_size=3)

    action = agent.act(state, training=True)
    assert action in (0, 1, 2)

    next_state, reward, done = env.step(action)
    agent.memory.push(state, action, float(reward), next_state, bool(done))

    loss = agent.train_step(batch_size=1)
    assert loss is not None
