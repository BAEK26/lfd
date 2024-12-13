from config import HYPERPARAMS
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from xarm_env import XArmEnv
from collections import deque
import random

# 기존 코드 그대로 유지하되, HYPERPARAMS 사용
def main():
    env = XArmEnv()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]

    q = QNet(input_dim, output_dim)
    q_target = QNet(input_dim, output_dim)
    q_target.load_state_dict(q.state_dict())

    memory = ReplayBuffer(HYPERPARAMS["buffer_limit"])
    optimizer = optim.Adam(q.parameters(), lr=HYPERPARAMS["learning_rate"])
    gamma = HYPERPARAMS["gamma"]
    batch_size = HYPERPARAMS["batch_size"]
    epsilon = HYPERPARAMS["epsilon_start"]
    epsilon_decay = HYPERPARAMS["epsilon_decay"]
    epsilon_min = HYPERPARAMS["epsilon_min"]

    for episode in range(HYPERPARAMS["episodes"]):
        state = env.reset()
        total_reward = 0

        for t in range(HYPERPARAMS["max_timesteps"]):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = q(torch.tensor(state, dtype=torch.float32)).detach().numpy()

            next_state, reward, done, _ = env.step(action)
            memory.put((state, action, reward, next_state, done))

            total_reward += reward
            state = next_state

            train(q, q_target, memory, optimizer, gamma, batch_size)

            if done:
                break

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if episode % 10 == 0:
            q_target.load_state_dict(q.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

    env.arm.disconnect()

if __name__ == "__main__":
    main()
