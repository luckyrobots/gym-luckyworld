import abc
import logging

import numpy as np
from luckyrobots import ActionModel, LuckyRobots, Node, Reset, Step, run_coroutine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("task")


class Task(abc.ABC, Node):
    """
    Abstract base class for a task.
    """

    @abc.abstractmethod
    def __init__(
        self,
        binary_path: str,
        robot_type: str,
        namespace: str = "",
        timeout: float = 30,
    ) -> None:
        self.node_name = self.__class__.__name__.lower()
        self.binary_path = binary_path
        self.robot_type = robot_type
        self.namespace = namespace
        self.timeout = timeout

        Node.__init__(self, self.node_name, namespace)

        self.luckyrobots = LuckyRobots()
        self.luckyrobots.register_node(self)

    async def _setup_async(self) -> None:
        """Setup the task"""
        self.reset_client = self.create_client(Reset, "/reset")
        self.step_client = self.create_client(Step, "/step")

    def _wait_for_luckyworld(self) -> None:
        logger.info("Waiting for LuckyWorld client to connect...")
        if not self.luckyrobots.wait_for_world_client(timeout=self.timeout):
            logger.error("No Lucky World client connected within timeout period")
            raise TimeoutError("No Lucky World client connected within timeout period")

        logger.info("LuckyRobots initialized successfully")

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Request a reset of the environment with a seed and options"""
        logger.info(f"Resetting environment with seed: {seed} and options: {options}")

        request = Reset.Request(seed=seed, options=options)
        future = run_coroutine(self.reset_client.call(request, timeout=self.timeout))
        response = future.result()

        if not response.success:
            logger.error(f"Failed to reset environment: {response.message}")
            raise RuntimeError(f"Failed to reset environment: {response.message}")

        raw_observation = response.observation
        info = response.info if response.info is not None else {}

        return raw_observation, info

    def step(self, action: ActionModel) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Request a step of the environment with an action"""
        logger.info(f"Stepping environment with action: {action}")

        request = Step.Request(action=action)
        future = run_coroutine(self.step_client.call(request, timeout=self.timeout))
        response = future.result()

        if not response.success:
            logger.error(f"Failed to step environment: {response.message}")
            raise RuntimeError(f"Failed to step environment: {response.message}")

        raw_observation = response.observation
        info = response.info if response.info is not None else {}

        reward = self.get_reward(raw_observation, info)
        terminated = self.is_terminated(raw_observation, info)
        truncated = False

        return raw_observation, reward, terminated, truncated, info

    @abc.abstractmethod
    def get_reward(self, observation: np.ndarray, info: dict) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminated(self, observation: np.ndarray, info: dict) -> bool:
        raise NotImplementedError

    def shutdown(self) -> None:
        self.luckyrobots.shutdown()


class PickandPlace(Task):
    """
    Pick and Place task.
    """

    def __init__(
        self,
        binary_path: str,
        robot_type: str,
        namespace: str = "",
        grasp_reward: float = 1.0,
        lift_reward: float = 1.0,
        place_reward: float = 3.0,
    ) -> None:
        super().__init__(binary_path, robot_type, namespace)

        self.grasp_reward = grasp_reward
        self.lift_reward = lift_reward
        self.place_reward = place_reward

        self.has_grasped = False
        self.has_lifted = False
        self.has_placed = False

        # self.luckyrobots.start(robot_type=robot_type, task="pickandplace", binary_path=self.binary_path)
        self.luckyrobots.start(binary_path=self.binary_path)

        self._wait_for_luckyworld()

    def _object_grasped(self, observation: np.ndarray, info: dict) -> bool:
        """Check if object is currently grasped."""
        # TODO: Implement based on observation
        return False

    def _object_at_target(self, observation: np.ndarray, info: dict) -> bool:
        """Check if object is placed at target location."""
        # TODO: Implement based on observation
        return False

    def reset(self, seed: int = None, options: dict[str, any] = None) -> tuple[np.ndarray, dict]:
        """
        Reset the task.
        """
        self.has_grasped = False
        self.has_lifted = False

        raw_observation, info = super().reset(seed=seed, options=options)

        return raw_observation, info

    def get_reward(self, observation: np.ndarray, info: dict) -> float:
        """We're focusing on imitation learning, so we don't need to calculate rewards."""
        return 0.0

    def is_terminated(self, observation: np.ndarray, info: dict) -> bool:
        """
        Episode terminates if:
        - Object is placed at target
        - Object is dropped
        """
        return self._object_at_target(observation, info) or (  # Place at target
            self.has_grasped and not self._object_grasped(observation, info)
        )  # Dropped object


class Navigation(Task):
    """
    Navigation task where the robot needs to reach a target position while avoiding obstacles.
    """

    def __init__(
        self,
        binary_path: str,
        robot_type: str,
        target_reward: float = 5.0,
        distance_reward_scale: float = 0.1,
        collision_penalty: float = -1.0,
        target_tolerance: float = 0.1,
        timeout: float = 10.0,
    ) -> None:
        super().__init__(binary_path, robot_type, timeout)
        self.target_reward = target_reward
        self.distance_reward_scale = distance_reward_scale
        self.collision_penalty = collision_penalty
        self.target_tolerance = target_tolerance

        self.target_position = None
        self.previous_distance = None
        self.has_collided = False

        self.luckyrobots.start(binary_path=self.binary_path, robot_type=robot_type, task="navigation")

        self._wait_for_luckyworld()

    def reset(
        self, seed: int | None = None, options: dict[str, any] | None = None
    ) -> tuple[np.ndarray, dict[str, any]]:
        """Reset task state and generate new target."""
        self.previous_distance = None
        self.has_collided = False

        raw_observation, info = super().reset(seed=seed, options=options)

        return raw_observation, info

    def _get_robot_position(self, observation: np.ndarray) -> np.ndarray:
        """Extract robot position from observation."""
        # Extract position from the agent_pos field of the observation dictionary
        if isinstance(observation, dict) and "agent_pos" in observation:
            # Return the first 3 values as position (assuming xyz coordinates)
            return observation["agent_pos"][:3]
        return np.zeros(3)  # Default position if not available

    def _check_collision(self, observation: np.ndarray, info: dict) -> bool:
        """Check if robot has collided with obstacles."""
        # In a real implementation, this would check collision data from the simulator
        # For now, return False as a placeholder
        return False

    def _get_distance_to_target(self, robot_position: np.ndarray) -> float:
        """Calculate distance to target."""
        return np.linalg.norm(robot_position - self.target_position)

    def get_reward(self, observation: np.ndarray, info: dict) -> float:
        """We're focusing on imitation learning, so we don't need to calculate rewards."""
        return 0.0

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
