import gym
import numpy as np
from arm import RobotArmController

class KalmanFilter:
    def __init__(self, state_dim):
        self.A = np.eye(state_dim)
        self.H = np.eye(state_dim)
        self.Q = np.eye(state_dim) * 0.1
        self.R = np.eye(state_dim) * 0.1
        self.P = np.eye(state_dim)
        self.x = np.zeros((state_dim, 1))

    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))

class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()
        self.robot = RobotArmController()
        self.kalman_filter = KalmanFilter(6)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))

    def reset(self):
        self.state = np.random.uniform(low=-1, high=1, size=6)
        return self.state

    def step(self, action):
        self.kalman_filter.predict()
        filtered_state = self.kalman_filter.update(self.robot.get_state())
        next_state = self.robot.move_arm(action)
        reward = -np.linalg.norm(filtered_state)
        done = reward > -0.1
        return next_state, reward, done, {}

    def render(self, mode="human"):
        pass
