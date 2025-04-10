import abc

import numpy as np


class Task(abc.ABC):
    """
    Abstract base class for tasks.
    """

    @abc.abstractmethod
    def __init__(self, robot_type: str) -> None:
        self.robot_type = robot_type

        self.info = {}

    @abc.abstractmethod
    def reset(self, seed: int = None) -> None:
        pass

    @abc.abstractmethod
    def get_reward(self, observation: np.ndarray, info: dict) -> float:
        pass

    @abc.abstractmethod
    def is_terminated(self, observation: np.ndarray, info: dict) -> bool:
        pass

    @abc.abstractmethod
    def render(self, render_mode: str) -> None:
        pass


class PickandPlace(Task):
    """
    Pick and Place task.
    """

    def __init__(
        self,
        robot_type: str,
        grasp_reward: float = 1.0,
        lift_reward: float = 1.0,
        place_reward: float = 3.0,
    ) -> None:
        self.robot_type = robot_type
        self.grasp_reward = grasp_reward
        self.lift_reward = lift_reward
        self.place_reward = place_reward

        self.has_grasped = False
        self.has_lifted = False

    def _object_grasped(self, observation: np.ndarray, info: dict) -> bool:
        """Check if object is currently grasped."""
        # TODO: Implement based on observation
        return False

    def _object_lifted(self, observation: np.ndarray, info: dict) -> bool:
        """Check if object is lifted above surface."""
        # TODO: Implement based on observation
        return False

    def _object_at_target(self, observation: np.ndarray, info: dict) -> bool:
        """Check if object is placed at target location."""
        # TODO: Implement based on observation
        return False

    def reset(self, seed: int = None) -> None:
        """
        Reset the task.
        """
        self.has_grasped = False
        self.has_lifted = False
        pass

    def get_reward(self, observation: np.ndarray, info: dict) -> float:
        """Calculate reward based on task progress."""
        reward = 0.0

        if self._object_grasped(observation, info) and not self.has_grasped:
            reward += self.grasp_reward
            self.has_grasped = True

        if self._object_lifted(observation, info) and not self.has_lifted:
            reward += self.lift_reward
            self.has_lifted = True

        if self._object_at_target(observation, info):
            reward += self.place_reward

        return reward

    def is_terminated(self, observation: np.ndarray, info: dict) -> bool:
        """Check termination conditions."""
        return (
            self._object_at_target(observation, info)  # Success
            or (self.has_grasped and not self._object_grasped(observation, info))  # Dropped object
        )

    def render(self, render_mode: str) -> None:
        """
        Render the task.
        """
        pass


class Navigation(Task):
    """
    Navigation task.
    """

    def __init__(self, robot_type: str) -> None:
        self.robot_type = robot_type

    def reset(self, seed: int = None) -> None:
        """
        Reset the task.
        """
        pass

    def get_reward(self, observation: np.ndarray, info: dict) -> float:
        """
        Get the reward for the task.
        """
        pass

    def is_terminated(self, observation: np.ndarray, info: dict) -> bool:
        pass

    def render(self, render_mode: str) -> None:
        """
        Render the task.
        """
        pass
