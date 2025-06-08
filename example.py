import imageio
import numpy as np
import gymnasium as gym
import gym_luckyworld # noqa: F401

game_path = "/mnt/c/Users/ethan/OneDrive/Documents/LuckyWorld/Build/Windows"
env = gym.make("gym_luckyworld/LuckyWorld-PickandPlace-v0", game_path=game_path)

observation, info = env.reset()
frames = []

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    if env.render_mode == "rgb_array":
        frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

if env.render_mode == "rgb_array":  
    imageio.mimsave("example.mp4", np.stack(frames), fps=10)
