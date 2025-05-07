import abc
import logging

import numpy as np
from luckyrobots import ActionModel, LuckyRobots, Node, Reset, Step, run_coroutine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("task")


class Task(abc.ABC, Node):
    @abc.abstractmethod
    def __init__(
        self,
        scene: str,
        task: str,
        robot_type: str,
        binary_path: str,
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
        self.reset_client = self.create_client(Reset, "/reset")
        self.step_client = self.create_client(Step, "/step")

    def _wait_for_luckyworld(self) -> None:
        logger.info("Waiting for LuckyWorld client to connect...")
        if not self.luckyrobots.wait_for_world_client(timeout=self.timeout):
            logger.error("No Lucky World client connected within timeout period")
            raise TimeoutError("No Lucky World client connected within timeout period")

        logger.info("LuckyRobots initialized successfully")

    def reset(
        self, seed: int | None = None, options: dict[str, any] | None = None
    ) -> tuple[np.ndarray, dict[str, any]]:
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

    def step(self, action: ActionModel) -> tuple[np.ndarray, float, bool, bool, dict[str, any]]:
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

    # NOTE: Not used for imitation learning
    @abc.abstractmethod
    def get_reward(self, observation: np.ndarray, info: dict[str, any]) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminated(self, observation: np.ndarray, info: dict[str, any]) -> bool:
        raise NotImplementedError

    def shutdown(self) -> None:
        self.luckyrobots.shutdown()


class PickandPlace(Task):
    def __init__(
        self,
        scene: str,
        task: str,
        robot_type: str,
        binary_path: str,
        namespace: str = "",
        timeout: float = 30,
    ) -> None:
        super().__init__(scene, task, robot_type, binary_path, namespace, timeout)

        self.has_grasped = None

        self.luckyrobots.start(scene, task, robot_type, binary_path)

        self._wait_for_luckyworld()

    def reset(
        self, seed: int | None = None, options: dict[str, any] | None = None
    ) -> tuple[np.ndarray, dict[str, any]]:
        self.has_grasped = False

        raw_observation, info = super().reset(seed=seed, options=options)

        return raw_observation, info

    def get_reward(self, observation: np.ndarray, info: dict[str, any]) -> float:
        return 0.0

    def is_terminated(self, observation: np.ndarray, info: dict[str, any]) -> bool:
        """
        Episode terminates if:
        - Object is placed at target (success)
        - Object was dropped not at target (fail)
        """
        object_grasped = info.get("object_grasped", False)
        self.has_grasped = object_grasped or self.has_grasped

        object_at_target = info.get("object_at_target", False)

        success = object_at_target and not object_grasped
        fail = self.has_grasped and not object_grasped and not object_at_target

        return success or fail


class Navigation(Task):
    def __init__(
        self,
        scene: str,
        task: str,
        robot_type: str,
        binary_path: str,
        namespace: str = "",
        timeout: float = 10.0,
    ) -> None:
        super().__init__(scene, task, robot_type, binary_path, namespace, timeout)

        self.target_position = None
        self.previous_distance = None
        self.has_collided = False

        self.target_tolerance = 0.1

        self.luckyrobots.start(scene, task, robot_type, binary_path)

        self._wait_for_luckyworld()

    def reset(
        self, seed: int | None = None, options: dict[str, any] | None = None
    ) -> tuple[np.ndarray, dict[str, any]]:
        self.previous_distance = None
        self.has_collided = False

        raw_observation, info = super().reset(seed=seed, options=options)

        return raw_observation, info

    def get_reward(self, observation: np.ndarray, info: dict[str, any]) -> float:
        return 0.0

    def is_terminated(self, observation: np.ndarray, info: dict[str, any]) -> bool:
        """
        Episode terminates if:
        - Robot reaches target (success)
        - Robot collides with obstacle (fail)
        """

        reached_target = info.get("reached_target", False)
        collision = info.get("collision", False)

        success = reached_target and not collision
        fail = collision

        return success or fail
