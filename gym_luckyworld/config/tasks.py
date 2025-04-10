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
        raise NotImplementedError

    @abc.abstractmethod
    def get_reward(self, observation: np.ndarray, info: dict) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminated(self, observation: np.ndarray, info: dict) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, render_mode: str) -> None:
        raise NotImplementedError


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
        super().__init__(robot_type)
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
        if seed is not None:
            np.random.seed(seed)

        # Generate random target position within workspace
        # TODO: Replace with actual workspace limits
        self.target_position = np.random.uniform(low=[-1.0, -1.0, 0.0], high=[1.0, 1.0, 1.0], size=3)

        self.has_grasped = False
        self.has_lifted = False

    def get_reward(self, observation: np.ndarray, info: dict) -> float:
        """
        Calculate reward based on:
        - Grasping object (sparse reward)
        - Lifting object (sparse reward)
        - Placing object at target (sparse reward)
        """
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
        """
        Episode terminates if:
        - Object is placed at target
        - Object is dropped
        """
        return (
            self._object_at_target(observation, info)  # Place at target
            or (self.has_grasped and not self._object_grasped(observation, info))  # Dropped object
        )

    def render(self, render_mode: str) -> None:
        """
        Render the task.
        """
        pass


class Navigation(Task):
    """
    Navigation task where the robot needs to reach a target position while avoiding obstacles.
    """

    def __init__(
        self,
        robot_type: str,
        target_reward: float = 5.0,
        distance_reward_scale: float = 0.1,
        collision_penalty: float = -1.0,
        target_tolerance: float = 0.1,
    ) -> None:
        super().__init__(robot_type)
        self.target_reward = target_reward
        self.distance_reward_scale = distance_reward_scale
        self.collision_penalty = collision_penalty
        self.target_tolerance = target_tolerance

        self.target_position = None
        self.previous_distance = None
        self.has_collided = False

    def reset(self, seed: int = None) -> None:
        """Reset task state and generate new target."""
        if seed is not None:
            np.random.seed(seed)

        # Generate random target position within workspace
        # TODO: Replace with actual workspace limits
        self.target_position = np.random.uniform(low=[-1.0, -1.0, 0.0], high=[1.0, 1.0, 1.0], size=3)

        self.previous_distance = None
        self.has_collided = False

    def _get_robot_position(self, observation: np.ndarray) -> np.ndarray:
        """Extract robot position from observation."""
        pass

    def _check_collision(self, observation: np.ndarray, info: dict) -> bool:
        """Check if robot has collided with obstacles."""
        pass

    def _get_distance_to_target(self, robot_position: np.ndarray) -> float:
        """Calculate distance to target."""
        return np.linalg.norm(robot_position - self.target_position)

    def get_reward(self, observation: np.ndarray, info: dict) -> float:
        """
        Calculate reward based on:
        - Distance to target (continuous reward)
        - Reaching target (sparse reward)
        - Collisions (penalty)
        """
        if self.target_position is None:
            return 0.0

        robot_position = self._get_robot_position(observation)
        current_distance = self._get_distance_to_target(robot_position)

        reward = 0.0

        # Distance-based reward
        if self.previous_distance is not None:
            # Reward for moving closer to target
            distance_improvement = self.previous_distance - current_distance
            reward += self.distance_reward_scale * distance_improvement

        self.previous_distance = current_distance

        # Target reached reward
        if current_distance < self.target_tolerance:
            reward += self.target_reward

        # Collision penalty
        if self._check_collision(observation, info) and not self.has_collided:
            reward += self.collision_penalty
            self.has_collided = True

        return reward

    def is_terminated(self, observation: np.ndarray, info: dict) -> bool:
        """
        Episode terminates if:
        - Robot reaches target
        - Robot collides with obstacle
        """
        if self.target_position is None:
            return False

        robot_position = self._get_robot_position(observation)
        distance = self._get_distance_to_target(robot_position)

        return (
            distance < self.target_tolerance  # Reached target
            or self._check_collision(observation, info)  # Collision
        )

    def render(self, render_mode: str) -> None:
        """
        Render the task visualization.
        """
        pass
