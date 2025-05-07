import json
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from luckyrobots import ActionModel, ObservationModel, PoseModel

from .task import Navigation, PickandPlace

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("luckyworld_env")


class LuckyWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    """
    A gymnasium-compatible environment for the LuckyWorld simulator.
    """

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

        self.scene = scene
        self.task = task
        self.robot_type = robot_type
        self.timeout = timeout
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.binary_path = binary_path

        self.latest_observation = None

        self._setup_task(scene, task, robot_type, binary_path)
        self._setup_spaces(robot_type, obs_type)

    def _setup_task(self, scene: str, task: str, robot_type: str, binary_path: str) -> None:
        """Set up the task."""
        if task == "pickandplace":
            self.task = PickandPlace(scene, task, robot_type, binary_path)
        elif task == "navigation":
            self.task = Navigation(scene, task, robot_type, binary_path)
        else:
            raise ValueError(f"Invalid task type: {task}")

    def _setup_spaces(self, robot_type: str, obs_type: str) -> None:
        """Set up gymnasium-style observation and action spaces."""
        with open(Path(__file__).parent / "config/robots.json") as f:
            robot_config = json.load(f)[robot_type]

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

        self.obs_type = obs_type

    def _convert_observation(self, observation: ObservationModel) -> dict:
        """
        Convert the raw ObservationModel to a dictionary format compatible with Gymnasium.
        """
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
        # Assuming the first 3 values are position (x, y, z)
        # and the next 3-4 values are orientation (could be euler angles or quaternion)
        if len(action) >= 6:  # At least position and orientation
            # Extract position (first 3 values)
            position = {"x": float(action[0]), "y": float(action[1]), "z": float(action[2])}

            # Extract orientation
            if len(action) >= 7:  # Quaternion (x, y, z, w)
                orientation = {
                    "x": float(action[3]),
                    "y": float(action[4]),
                    "z": float(action[5]),
                    "w": float(action[6]),
                }
            else:  # Euler angles (roll, pitch, yaw) - convert to quaternion
                # This is a simplified conversion - you might need a more accurate one
                # depending on your convention (XYZ, ZYX, etc.)
                roll, pitch, yaw = action[3], action[4], action[5]

                # Simple conversion from Euler angles to quaternion
                # Note: This is just an example and might not match your convention
                import math

                cy = math.cos(yaw * 0.5)
                sy = math.sin(yaw * 0.5)
                cp = math.cos(pitch * 0.5)
                sp = math.sin(pitch * 0.5)
                cr = math.cos(roll * 0.5)
                sr = math.sin(roll * 0.5)

                orientation = {
                    "w": cr * cp * cy + sr * sp * sy,
                    "x": sr * cp * cy - cr * sp * sy,
                    "y": cr * sp * cy + sr * cp * sy,
                    "z": cr * cp * sy - sr * sp * cy,
                }

            # Create a PoseModel instance
            pose = PoseModel(position=position, orientation=orientation)

            # Create and return the ActionModel
            return ActionModel(pose=pose)

        elif len(action) == 3:  # Just position
            position = {"x": float(action[0]), "y": float(action[1]), "z": float(action[2])}

            # Default orientation (identity quaternion)
            orientation = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

            pose = PoseModel(position=position, orientation=orientation)
            return ActionModel(pose=pose)

        else:
            # For very short arrays or other formats, you might need to adapt
            # this part to your specific requirements
            raise ValueError(f"Action array with length {len(action)} not supported")

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed, options=options)

        raw_observation, info = self.task.reset(seed=seed)
        # Store the raw observation for rendering
        self.latest_observation = raw_observation

        observation = self._convert_observation(raw_observation)

        return observation, info

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Perform a step in the environment."""
        raw_action = self._convert_action(action)

        raw_observation, reward, terminated, truncated, info = self.task.step(raw_action)
        # Store the raw observation for rendering
        self.latest_observation = raw_observation

        observation = self._convert_observation(raw_observation)

        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """
        Render the environment.
        """
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            pass

        return None

    def close(self) -> None:
        """Close the environment."""
        if self.task:
            self.task.shutdown()

        logger.info("Environment closed")
