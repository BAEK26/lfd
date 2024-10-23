import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv

# 환경 정의
class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,))  # 7 DOF for robotic arm
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,))  # Position, velocity, angles etc.

    def reset(self):
        self.state = np.zeros(14)  # 초기 상태
        return self.state

    def step(self, action):
        # 상태 업데이트 (간단한 예시)
        self.state = self.state + action
        reward = self._calculate_reward(action)
        done = self._is_done()
        return self.state, reward, done, {}

    def _calculate_reward(self, action):
        # 보상 함수: 최적 경로에 가까울수록 높은 보상
        distance_to_goal = np.linalg.norm(self.state[:3] - np.array([1, 1, 1]))  # 목표 위치 예시
        reward = -distance_to_goal
        return reward

    def _is_done(self):
        # 종료 조건: 목표 도달 시 종료
        distance_to_goal = np.linalg.norm(self.state[:3] - np.array([1, 1, 1]))
        return distance_to_goal < 0.1

# 환경 초기화
env = DummyVecEnv([lambda: RobotEnv()])
