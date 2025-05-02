import abc
import asyncio
import concurrent.futures
import logging

import numpy as np
from luckyrobots import ActionModel, LuckyRobots, Node, Reset, Step

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
        timeout: float = 500,
    ) -> None:
        self.binary_path = binary_path
        self.robot_type = robot_type
        self.namespace = namespace
        self.timeout = timeout

        Node.__init__(self, self.__class__.__name__, namespace)

        self.luckyrobots = LuckyRobots()
        self.luckyrobots.register_node(self)

    def _setup(self) -> None:
        """Setup the interface node"""
        logger.info("Setting up interface node...")

        self.reset_client = self.create_client(Reset, "/reset")
        self.step_client = self.create_client(Step, "/step")

        self.action_pub = self.create_publisher(ActionModel, "/action")

        logger.info("Interface node setup complete")

    def _wait_for_luckyworld(self) -> None:
        logger.info("Waiting for LuckyWorld client to connect...")
        if not self.luckyrobots.wait_for_world_client(timeout=self.timeout):
            logger.error("No Lucky World client connected within timeout period")
            raise TimeoutError("No Lucky World client connected within timeout period")

        logger.info("LuckyRobots initialized successfully")

    def request_reset(self, seed: int = None) -> Reset.Response:
        """Request a reset of the environment synchronously"""
        logger.info(f"Resetting environment with seed: {seed}")

        def async_reset():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                request = Reset.Request(seed=seed)
                response = loop.run_until_complete(self.reset_client.call(request, timeout=self.timeout))
            except Exception as e:
                logger.error(f"Error in reset request: {e}")
                response = Reset.Response(success=False, message=str(e), observation=None, info=None)
            finally:
                loop.close()

            return response

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(async_reset)
            try:
                response = future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                logger.error("Reset request timed out")
                response = Reset.Response(
                    success=False,
                    message="Reset request timed out",
                    observation=None,
                    info=None,
                )

        return response

    def request_step(self, action: ActionModel) -> Step.Response:
        """Request a step of the environment synchronously"""
        logger.info(f"Stepping environment with action: {action}")

        def async_step():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                request = Step.Request(action=action)
                response = loop.run_until_complete(self.step_client.call(request, timeout=self.timeout))
            except Exception as e:
                logger.error(f"Error in step request: {e}")
                response = Step.Response(success=False, message=str(e), observation=None, info=None)
            finally:
                loop.close()

            return response

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(async_step)
            try:
                response = future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                logger.error("Step request timed out")
                response = Step.Response(
                    success=False,
                    message="Step request timed out",
                    observation=None,
                    info=None,
                )

        return response

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

    def _object_lifted(self, observation: np.ndarray, info: dict) -> bool:
        """Check if object is lifted above surface."""
        # TODO: Implement based on observation
        return False

    def _object_at_target(self, observation: np.ndarray, info: dict) -> bool:
        """Check if object is placed at target location."""
        # TODO: Implement based on observation
        return False

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        """
        Reset the task.
        """
        if seed is not None:
            np.random.seed(seed)

        self.has_grasped = False
        self.has_lifted = False

        response = self.request_reset(seed)
        if not response.success:
            logger.error("Failed to reset task")
            raise RuntimeError("Failed to reset task")

        raw_observation = response.observation
        info = response.info if response.info is not None else {}

        return raw_observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute an action in the environment.
        """
        # Send action to the robot
        response = self.request_step(action)
        if not response.success:
            logger.error(f"Failed to step environment: {response.message}")
            raise RuntimeError(f"Failed to step environment: {response.message}")

        raw_observation = response.observation
        info = response.info if response.info is not None else {}

        reward = self.get_reward(raw_observation, info)
        terminated = self.is_terminated(raw_observation, info)
        truncated = False

        return raw_observation, reward, terminated, truncated, info

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
        return self._object_at_target(observation, info) or (  # Place at target
            self.has_grasped and not self._object_grasped(observation, info)
        )  # Dropped object

    def render(self, render_mode: str) -> None:
        """
        Render the task.
        """
        pass

    def get_target_position(self) -> np.ndarray:
        """Return the target position for object placement."""
        # In a real implementation, this would return the actual target position
        # For now, return a fixed position as a placeholder
        return np.array([0.5, 0.5, 0.1])  # Example target position


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

    def reset(self, seed: int = None, options: dict = None) -> None:
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

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute an action in the environment.
        """
        # Send action to the robot
        success = self.luckyrobots_interface.request_step(action)
        if not success:
            logger.error("Failed to step environment")
            raise RuntimeError("Failed to step environment")

        # Get observation
        observation = self.luckyrobots_interface.get_observation()

        # Calculate reward and termination
        info = {}
        reward = self.get_reward(observation, info)
        terminated = self.is_terminated(observation, info)
        truncated = False

        return observation, reward, terminated, truncated, info

    def render(self, render_mode: str) -> None:
        """
        Render the task visualization.
        """
        pass

    def get_target_position(self) -> np.ndarray:
        """Return the current target position."""
        if self.target_position is None:
            # Return a default position if not set
            return np.zeros(3)
        return self.target_position
