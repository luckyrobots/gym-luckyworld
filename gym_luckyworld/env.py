import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from luckyrobots import ActionModel, ObservationModel
from omegaconf import OmegaConf

from .task import Navigation, PickandPlace

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gym_luckyworld")


class LuckyWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        scene: str,
        task: str,
        robot_type: str,
        obs_type: str,
        timeout: float = 10.0,
        render_mode: str = "human",
        binary_path: str = None,
    ):
        super().__init__()

        self.task = task
        self.robot_type = robot_type
        self.timeout = timeout
        self.obs_type = obs_type
        self.render_mode = render_mode

        self.latest_observation = None

        robots_config = OmegaConf.load(Path(__file__).parent / "config/robots.yaml")

        self._validate_params(robots_config, obs_type, scene, task, robot_type)

        self._setup_task(scene, task, robot_type, binary_path, render_mode)
        self._setup_spaces(robots_config[robot_type], obs_type)

    def _validate_params(
        self, robots_config: dict[str, any], obs_type: str, scene: str, task: str, robot_type: str
    ) -> None:
        if robot_type not in robots_config:
            raise ValueError(f"Invalid robot type: {robot_type}")
        if obs_type not in robots_config[robot_type]["observation_types"]:
            raise ValueError(f"Invalid observation type: {obs_type}")
        if scene not in robots_config[robot_type]["available_scenes"]:
            raise ValueError(f"Invalid scene: {scene}")
        if task not in robots_config[robot_type]["available_tasks"]:
            raise ValueError(f"Invalid task: {task}")

    def _setup_task(self, scene: str, task: str, robot_type: str, binary_path: str, render_mode: str) -> None:
        if task == "pickandplace":
            self.task = PickandPlace(scene, task, robot_type, binary_path, render_mode)
        elif task == "navigation":
            self.task = Navigation(scene, task, robot_type, binary_path, render_mode)
        else:
            raise ValueError(f"Invalid task type: {task}")

    def _setup_spaces(self, robot_config: dict[str, any], obs_type: str) -> None:
        # Set up action space (same for all observation types)
        action_dim = len(robot_config["action_space"]["joint_names"])
        action_limits = robot_config["action_space"]["joint_limits"]
        self.action_space = spaces.Box(
            low=np.array([limit["lower"] for limit in action_limits]),
            high=np.array([limit["upper"] for limit in action_limits]),
            shape=(action_dim,),
        )

        # Set up observation space based on obs_type
        obs_dim = len(robot_config["observation_space"]["joint_names"])
        obs_limits = robot_config["observation_space"]["joint_limits"]

        if obs_type == "environment_state_pixels_agent_pos":
            # Camera image + agent position + target position
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(64, 64, 3),  # Example image dimensions
                        dtype=np.uint8,
                    ),
                    "agent_pos": spaces.Box(
                        low=np.array([limit["lower"] for limit in obs_limits]),
                        high=np.array([limit["upper"] for limit in obs_limits]),
                        shape=(obs_dim,),
                    ),
                }
            )
        else:
            raise ValueError(f"Unknown observation type: {obs_type}")

    # TODO: Fix this once we get the observation images working
    def _convert_observation(self, observation: ObservationModel) -> dict:
        # Initialize observation dictionary with default values
        obs_dict = {
            "agent_pos": np.zeros(6, dtype=np.float32),
            "pixels": np.zeros((64, 64, 3), dtype=np.uint8),
        }

        # Extract data from observation_state
        if hasattr(observation, "observation_state") and observation.observation_state:
            state_dict = observation.observation_state

            # Extract agent position (joint angles) - assuming they're stored as integers
            # We need to convert to the proper data type and scale
            joint_positions = np.zeros(6, dtype=np.float32)

            # Map the appropriate fields from observation_state to joint_positions
            # This mapping depends on how your robot's state is represented
            for i in range(6):
                # Adjust the key names as needed for your specific implementation
                joint_key = f"joint_{i}"
                if joint_key in state_dict:
                    # Convert from int to float and scale appropriately
                    # Assuming the values are stored as integers with some scaling factor
                    joint_positions[i] = float(state_dict[joint_key]) / 1000.0  # Adjust scaling as needed

            obs_dict["agent_pos"] = joint_positions

        # Process camera data
        if hasattr(observation, "observation_cameras") and observation.observation_cameras:
            for camera_data in observation.observation_cameras:
                try:
                    # Check if file path exists
                    if hasattr(camera_data, "file_path") and camera_data.file_path:
                        # In a real implementation, you would load the image from the file path
                        # For example:
                        # import cv2
                        # img = cv2.imread(camera_data.file_path)
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Resize to match the expected dimensions if needed
                        # img = cv2.resize(img, (64, 64))
                        # obs_dict["pixels"] = img

                        # For now, we'll use a placeholder
                        obs_dict["pixels"] = np.zeros((64, 64, 3), dtype=np.uint8)
                        break  # Just use the first camera for now
                except Exception as e:
                    logger.warning(f"Error processing camera data: {e}")
                    # Keep default camera placeholder

        return obs_dict

    def _convert_action(self, action: np.ndarray) -> ActionModel:
        """
        Convert a NumPy array action to an ActionModel instance to pass over websocket to LuckyWorld.
        """
        if action.size == self.action_space.shape[0] and self.action_space.contains(action):
            if self.robot_type == 'so100':
                joint_positions = {f"{i}": action[i] for i in range(action.size)}
                action_model = ActionModel(joint_positions=joint_positions)
            elif self.robot_type == 'stretch_v1':
                joint_velocities = {f"{i}": action[i] for i in range(action.size)}
                action_model = ActionModel(joint_velocities=joint_velocities)

            return action_model
        else:
            raise ValueError(f"Action array with length {action.size} not supported")

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        raw_observation, info = self.task.reset(seed=seed)
        # Store the raw observation for rendering
        self.latest_observation = raw_observation

        observation = self._convert_observation(raw_observation)

        return observation, info

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        raw_action = self._convert_action(action)

        raw_observation, reward, terminated, truncated, info = self.task.step(raw_action)
        # Store the raw observation for rendering
        self.latest_observation = raw_observation

        observation = self._convert_observation(raw_observation)

        return observation, reward, terminated, truncated, info

    # TODO: Fix this once we get the observation images working
    def render(self, camera_index: int = 0) -> np.ndarray:
        if self.latest_observation is None:
            logger.warning("No observation available for rendering")
            return np.zeros((64, 64, 3), dtype=np.uint8)

        if self.render_mode is None or self.render_mode == "human":
            return None
        elif self.render_mode == "rgb_array":
            cameras = self.latest_observation.observation_cameras
            if cameras is None:
                logger.warning("No camera data found in observation")
                return np.zeros((64, 64, 3), dtype=np.uint8)
            if camera_index >= len(cameras):
                logger.warning(f"Camera index {camera_index} out of range. Using camera 0 instead.")
                camera_index = 0

            try:
                camera_data = cameras[camera_index]
                return camera_data.image
            except Exception as e:
                logger.warning(f"Error rendering camera {camera_index} data: {e}")
                return np.zeros((64, 64, 3), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def close(self) -> None:
        if self.task:
            self.task.shutdown()

        logger.info("Environment closed")
