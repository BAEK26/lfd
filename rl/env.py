import gym
import numpy as np
from arm import RobotArm

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
        self.robot_arm = RobotArm()
        self.kalman_filter = KalmanFilter(state_dim=6)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))

    def reset(self):
        self.state = np.zeros(6)  # 초기 상태
        return self.state

    def step(self, action):
        # 칼만 필터 적용 전 상태 업데이트
        self.kalman_filter.predict()
        filtered_state = self.kalman_filter.update(self.robot_arm.get_state())

        # 로봇 암의 상태 업데이트 및 액션 적용
        next_state = self.robot_arm.step(action)
        reward = self._calculate_reward(next_state)
        done = self._is_done(next_state)

        return filtered_state, reward, done, {}

    def _calculate_reward(self, state):
        goal = np.array([0, 0, 0, 0, 0, 0])  # 목표 관절 각도
        distance_to_goal = np.linalg.norm(state - goal)
        reward = -distance_to_goal  # 목표와 가까울수록 보상
        return reward

    def _is_done(self, state):
        return np.linalg.norm(state - np.array([0, 0, 0, 0, 0, 0])) < 0.05
