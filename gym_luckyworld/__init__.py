from gymnasium.envs.registration import register

register(
    id="gym_luckyworld/LuckyWorld-PickandPlace-v0",
    entry_point="gym_luckyworld.env:LuckyWorld",
    max_episode_steps=300,
    nondeterministic=True,
    kwargs={
        "scene": "ArmLevel",
        "task": "pickandplace",
        "robot": "so100",
        "debug": True,
        "obs_type": "pixels_agent_pos",
        "render_mode": "rgb_array",
        # "game_path": None,
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
        "robot": "stretch_v1",
        "obs_type": "pixels_agent_pos",
        "render_mode": "rgb_array",
        # "game_path": None,
    },
)
