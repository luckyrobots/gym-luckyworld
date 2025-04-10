import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import luckyrobots as lr
import numpy as np
from gymnasium import spaces

from .config.tasks import Navigation, PickandPlace


class LuckyWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    """
    A gymnasium-compatible environment for the LuckyWorld simulator.
    """

    def __init__(
        self,
        task: str,
        robot_type: str,
        obs_type: str,
        timeout: float = 1.0,
        render_mode: str = "human",
    ):
        super().__init__()

        self.timeout = timeout
        self.render_mode = render_mode

        self._setup_task(task, robot_type)
        self._setup_spaces(robot_type, obs_type)

        self.robot_observation_history = deque(maxlen=10)

        self._loop = asyncio.get_event_loop()
        asyncio.set_event_loop(self._loop)

        lr.start(task_name=task)

    def _setup_spaces(self, robot_type: str, obs_type: str) -> None:
        """Set up gymnasium-style observation and action spaces."""
        with open(Path(__file__).parent / "config/robot.json") as f:
            robot_config = json.load(f)[robot_type]

        # Get action dimension from robot config
        action_dim = len(robot_config["action_space"]["joint_names"])
        action_limits = robot_config["action_space"]["joint_limits"]
        self.action_space = spaces.Box(
            low=np.array([limit["lower"] for limit in action_limits]),
            high=np.array([limit["upper"] for limit in action_limits]),
            shape=(action_dim,),
            dtype=np.float32,
        )

        # Get observation dimension from robot config
        obs_dim = len(robot_config["observation_space"]["joint_names"])
        obs_limits = robot_config["observation_space"]["joint_limits"]
        self.observation_space = spaces.Box(
            low=np.array([limit["lower"] for limit in obs_limits]),
            high=np.array([limit["upper"] for limit in obs_limits]),
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _setup_task(self, task: str, robot_type: str) -> None:
        """Set up the task."""
        if task == "pickandplace":
            self.task = PickandPlace(robot_type)
        elif task == "navigation":
            self.task = Navigation(robot_type)
        else:
            raise ValueError(f"Invalid task type: {task}")

    @lr.message_receiver
    async def observation_sub(self, message: np.ndarray, robot_images: np.ndarray) -> None:
        """Subscribes to the observation."""
        self.robot_observation_history.append((message, robot_images))

    async def action_pub(self, action: np.ndarray) -> None:
        """Publishes the action."""
        await lr.send_commands(action)

    def _get_observation(self) -> np.ndarray:
        """Get the observation from the history."""
        start_time = time.time()

        while len(self.robot_observation_history) == 0:
            if time.time() - start_time > self.timeout:
                raise TimeoutError("No observations received within timeout period")
            time.sleep(0.01)

        raw_obs = self.robot_observation_history[-1]

        info = None

        # TODO: Process raw observation into gymnasium-compatible observation
        return raw_obs, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment.
        """
        super().reset(seed=seed)

        self.task.reset(seed=seed)

        try:
            observation, info = self._get_observation()
        except TimeoutError as err:
            raise RuntimeError("Failed to get observation from robot") from err

        info["is_success"] = False

        return observation, info

    def _get_reward(self, observation: np.ndarray, info: dict) -> float:
        """Get the reward from the task."""
        return self.task.get_reward(observation, info)

    def _is_terminated(self, observation: np.ndarray, info: dict) -> bool:
        """Check if the episode is terminated."""
        return self.task.is_terminated(observation, info)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Perform a step in the environment.
        """
        # Normalize the action
        normalized_action = self.action_space.high * action

        self._loop.run_until_complete(self.action_pub(normalized_action))

        try:
            observation, info = self._get_observation()
        except TimeoutError as err:
            raise RuntimeError("Failed to get observation from robot") from err

        reward = self._get_reward(observation, info)
        terminated = self._is_terminated(observation, info)
        truncated = False  # TimeLimit wrapper will handle this
        info["is_success"] = reward == 5

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """
        Render the environment.
        """
        self.task.render(self.render_mode)

    def close(self) -> None:
        """
        Close the environment.
        """
        self.robot_observation_history.clear()
        lr.LuckyRobots.run_exit_handler()
