from gymnasium.envs.registration import register

register(
    id="gym_luckyworld/LuckyWorld-PickandPlace-v0",
    entry_point="gym_luckyworld.env:LuckyWorld",
    max_episode_steps=300,
    nondeterministic=True,
    kwargs={
        "scene": "kitchen",
        "task": "pickandplace",
        "robot_type": "so100",
        "obs_type": "pixels_agent_pos",
        "render_mode": "rgb_array",
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
        "obs_type": "pixels_agent_pos",
        "render_mode": "human",
    },
)
