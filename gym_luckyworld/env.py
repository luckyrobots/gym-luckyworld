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
        self._setup_spaces(robot_type, obs_type)

        self.robot_observation_history = deque(maxlen=10)

        self._loop = asyncio.get_event_loop()
        asyncio.set_event_loop(self._loop)

        # lr.start()

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

    @lr.message_receiver
    async def observation_sub(self, message, robot_images) -> Tuple[np.ndarray, np.ndarray]:
        """Subscribes to the observation."""
        self.robot_observation_history.append(message)
        return message, robot_images

    async def action_pub(self, action: np.ndarray) -> None:
        """Publishes the action."""
        await lr.send_commands(action)

    def _get_observation(self) -> np.ndarray:
        """Get the observation from the robot."""
        raw_obs = self._get_raw_observation()

        # TODO: Process raw observation into gymnasium-compatible observation
        return raw_obs

    def _get_raw_observation(self) -> np.ndarray:
        """Get the raw observation from the robot."""
        start_time = time.time()

        while len(self.robot_observation_history) == 0:
            if time.time() - start_time > self.timeout:
                raise TimeoutError("No observations received within timeout period")
            time.sleep(0.01)

        return self.robot_observation_history[-1]

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment.
        """
        super().reset(seed=seed)

        try:
            observation = self._get_observation()
        except TimeoutError as err:
            raise RuntimeError("Failed to get observation from robot") from err

        info = {"is_success": False}

        return observation, info

    def _get_reward(self) -> float:
        """Get the reward from the task."""
        return 0.0

    def _is_terminated(self) -> bool:
        """
        Episode is terminated when:
            - Failed grasp
            - Failed placement
            - Constraints violated
            - Task completion
        """
        return False  # TODO: Add termination condition

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Perform a step in the environment.
        """
        # Normalize the action
        normalized_action = self.action_space.high * action

        self._loop.run_until_complete(self.action_pub(normalized_action))

        try:
            observation = self._get_observation()
        except TimeoutError as err:
            raise RuntimeError("Failed to get observation from robot") from err

        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = False  # TimeLimit wrapper will handle this
        info = {"is_success": reward == 5}  # NOTE: Needs to represent a successful trajectory

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """
        Render the environment.
        """
        pass

    def close(self) -> None:
        """
        Close the environment.
        """
        self.robot_observation_history.clear()
        lr.run_exit_handler()
