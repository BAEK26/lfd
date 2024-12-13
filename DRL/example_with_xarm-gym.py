# example.py
import gymnasium as gym
import gym_xarm

env = gym.make("gym_xarm/XarmLift-v0", render_mode="human")

# from gymnasium import envs
# all_env_keys = envs.registry.keys()
# all_env_keys = list(sorted(all_env_keys))
# xarm_keys = [key for key in all_env_keys if "xarm" in key.lower()]
# print(xarm_keys)
#['gym_xarm/XarmLift-v0']


observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()