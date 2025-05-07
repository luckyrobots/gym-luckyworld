from pathlib import Path

from gymnasium.envs.registration import register

binary_path = Path(__file__).parent.parent.parent / "LuckyWorldV2"

register(
    id="gym_luckyworld/LuckyWorld-PickandPlace-v0",
    entry_point="gym_luckyworld.env:LuckyWorld",
    max_episode_steps=300,
    nondeterministic=True,
    kwargs={
        "scene": "loft",
        "task": "pickandplace",
        "robot_type": "so100",
        "obs_type": "environment_state_pixels_agent_pos",
        "render_mode": "human",
        "binary_path": binary_path,
    },
)

register(
    id="gym_luckyworld/LuckyWorld-Navigation-v0",
    entry_point="gym_luckyworld.env:LuckyWorld",
    max_episode_steps=300,
    nondeterministic=True,
    kwargs={
        "scene": "loft",
        "task": "navigation",
        "robot_type": "stretch_v1",
        "obs_type": "environment_state_pixels_agent_pos",
        "render_mode": "human",
        "binary_path": binary_path,
    },
)
