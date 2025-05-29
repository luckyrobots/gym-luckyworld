import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import gym_luckyworld  # noqa: F401


@pytest.mark.parametrize(
    "env_task, obs_type",
    [
        ("LuckyWorld-PickandPlace-v0", "pixels_agent_pos"),
        ("LuckyWorld-Navigation-v0", "pixels_agent_pos"),
    ],
)
def test_luckyworld(env_task, obs_type):
    env = gym.make(f"gym_luckyworld/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped)
