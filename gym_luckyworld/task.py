import abc
import logging

import numpy as np
from luckyrobots import LuckyRobots, Node, Reset, Step, run_coroutine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("task")


class Task(abc.ABC, Node):
    @abc.abstractmethod
    def __init__(
        self,
        scene: str,
        task: str,
        robot: str,
        render_mode: str,
        namespace: str = "",
        debug: bool = False,
        request_timeout: float = 30,
    ) -> None:
        node_name = self.__class__.__name__.lower()

        self.request_timeout = request_timeout

        Node.__init__(self, node_name, namespace, host="localhost", port=3000)

        self.luckyrobots = LuckyRobots()
        self.luckyrobots.register_node(self)

    async def _setup_async(self) -> None:
        self.reset_client = self.create_client(Reset, "/reset")
        self.step_client = self.create_client(Step, "/step")

    def reset(
        self, seed: int | None = None, options: dict[str, any] | None = None
    ) -> tuple[np.ndarray, dict[str, any]]:
        request = Reset.Request(seed=seed, options=options)
        future = run_coroutine(self.reset_client.call(request, timeout=self.request_timeout))
        response = future.result()

        if not response.success:
            logger.error(f"Failed to reset environment: {response.message}")
            self.shutdown()
            raise RuntimeError(f"Failed to reset environment: {response.message}")

        raw_observation = response.observation
        info = response.info if response.info is not None else {}

        return raw_observation, info

    def step(self, actuator_values: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, any]]:
        request = Step.Request(actuator_values=actuator_values.tolist())
        future = run_coroutine(self.step_client.call(request, timeout=self.request_timeout))
        response = future.result()

        if not response.success:
            logger.error(f"Failed to step environment: {response.message}")
            self.shutdown()
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


class PickandPlace(Task):
    def __init__(
        self,
        scene: str,
        task: str,
        robot: str,
        debug: bool,
        render_mode: str,
        namespace: str = "",
        request_timeout: float = 30,
        distance_threshold: float = 10.0,
    ) -> None:
        super().__init__(scene, task, robot, debug, render_mode, namespace, request_timeout)

        self.has_grasped = None
        self.distance_threshold = distance_threshold

        # TODO: Add headless mode for rgb_array render_mode
        self.luckyrobots.start(scene=scene, robot=robot, task=task, debug=debug)
        self.luckyrobots.wait_for_world_client()

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
        Episode terminates successfully if:
        - Object is placed at target and robot returned home
        """
        object_distance = float(info["object_distance_from_target"])
        object_at_target = object_distance < self.distance_threshold

        is_robot_home = bool(int(info["is_robot_home"]))

        success = object_at_target and is_robot_home

        info["is_success"] = success

        return success


class Navigation(Task):
    def __init__(
        self,
        scene: str,
        task: str,
        robot: str,
        render_mode: str,
        debug: bool = False,
        namespace: str = "",
        request_timeout: float = 30.0,
    ) -> None:
        super().__init__(scene, task, robot, render_mode, namespace, request_timeout)

        self.target_tolerance = 0.1

        # TODO: Add headless mode for rgb_array render_mode
        self.luckyrobots.start(scene=scene, task=task, robot=robot, debug=debug)
        self.luckyrobots.wait_for_world_client()

    def reset(
        self, seed: int | None = None, options: dict[str, any] | None = None
    ) -> tuple[np.ndarray, dict[str, any]]:
        raw_observation, info = super().reset(seed=seed, options=options)

        return raw_observation, info

    def get_reward(self, observation: np.ndarray, info: dict[str, any]) -> float:
        return 0.0

    # TODO: Improve the distance threshold once the information dictionary is refined and Stretch is added
    def is_terminated(self, observation: np.ndarray, info: dict[str, any]) -> bool:
        """
        Episode terminates if:
        - Robot reaches target (success)
        - Robot collides with obstacle (fail)
        """
        robot_distance = float(info["robot_distance_from_target"])
        has_collided = bool(info["has_collided"])

        success = robot_distance < self.target_tolerance and not has_collided
        fail = has_collided

        info["is_success"] = success

        return success or fail
