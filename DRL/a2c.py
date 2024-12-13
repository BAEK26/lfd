from config import HYPERPARAMS
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from xarm_env import XArmEnv

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        pi = torch.softmax(self.actor(x), dim=-1)
        v = self.critic(x)
        return pi, v

def compute_advantage(rewards, values, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    values = values.detach()
    advantage = returns - values
    return advantage, returns

def train(actor_critic, optimizer, states, actions, advantages, returns):
    pi, values = actor_critic(states)
    values = values.squeeze()

    log_probs = torch.log(pi.gather(1, actions.unsqueeze(1)).squeeze())
    policy_loss = -(log_probs * advantages).mean()
    value_loss = nn.functional.mse_loss(values, returns)

    entropy = -(pi * torch.log(pi + 1e-10)).sum(dim=1).mean()
    entropy_loss = -HYPERPARAMS["entropy_beta"] * entropy

    loss = policy_loss + value_loss + entropy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    env = XArmEnv()
    input_dim = env.observation_space["observation"].shape[0]
    action_dim = env.action_space.shape[0]

    actor_critic = ActorCritic(input_dim, action_dim)
    optimizer = optim.Adam(actor_critic.parameters(), lr=HYPERPARAMS["learning_rate"])
    gamma = HYPERPARAMS["gamma"]

    for episode in range(HYPERPARAMS["episodes"]):
        state = env.reset()
        states, actions, rewards = [], [], []

        for t in range(HYPERPARAMS["max_timesteps"]):
            state_tensor = torch.tensor(state["observation"], dtype=torch.float32)
            pi, _ = actor_critic(state_tensor)
            action = torch.multinomial(pi, 1).item()

            next_state, reward, done, _ = env.step(action)

            states.append(state["observation"])
            actions.append(action)
            rewards.append(reward)

            state = next_state

            if done:
                break

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        advantages, returns = compute_advantage(rewards, actor_critic(states)[1].squeeze(), gamma)

        train(actor_critic, optimizer, states, actions, advantages, returns)

        print(f"Episode {episode}, Total Reward: {sum(rewards)}")

    env.arm.disconnect()

if __name__ == "__main__":
    main()
