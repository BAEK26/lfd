import torch
import torch.nn as nn
from stable_baselines3 import PPO

class TrajectoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=50, num_layers=1, batch_first=True)
        self.fc = nn.Linear(50, 6)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

# PPO 모델 정의
class RobotPPOModel:
    def __init__(self, env):
        self.model = PPO('MlpPolicy', env, verbose=1)

    def train(self, timesteps=50000):
        self.model.learn(total_timesteps=timesteps)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)
