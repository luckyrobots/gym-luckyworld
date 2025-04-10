from gymnasium.envs.registration import register

register(
    id="gym_luckyworld/LuckyWorld-PickandPlace-v0",
    entry_point="gym_luckyworld.env:LuckyWorldEnv",
    max_episode_steps=300,
    nondeterministic=True,
    kwargs={
        "task": "pickandplace",
        "robot_type": "so100",
        "obs_type": "pixels_agent_pos",
    },
)

register(
    id="gym_luckyworld/LuckyWorld-Navigation-v0",
    entry_point="gym_luckyworld.env:LuckyWorldEnv",
    max_episode_steps=300,
    nondeterministic=True,
    kwargs={
        "task": "navigation",
        "robot_type": "stretch_v1",
        "obs_type": "pixels_agent_pos",
    },
)
