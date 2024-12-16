import gym
import math
import numpy as np
from gymnasium import spaces
from xarm.wrapper import XArmAPI
from gymnasium_robotics.core import GoalEnv
import os

JOINT_LIMIT =  [
    (-2 * math.pi, 2 * math.pi),
    (-2.042035, 2.024581),
    (-0.061086523819801536, 3.92699),
    (-2 * math.pi, 2 * math.pi),
    (-1.692969, 2.1642082724729685),
    (-2 * math.pi, 2 * math.pi)
],

class XArmEnv(GoalEnv):
    def __init__(self, use_demonstrations=True):
        super(XArmEnv, self).__init__()
        self.arm = XArmAPI('192.168.1.194')
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        print('initializing with state:', self.arm.get_state())

        self._max_episode_steps = 100
        self.goal_space = spaces.Box(
            low=np.array([260, 190, 250, -170, -90, 0]), 
            high=np.array([270, 200, 260, -155, -80, 10]),
            dtype='float32'
        )
        self.goal = self._sample_goal()

        obs = self._get_obs()
        self.action_space = spaces.Box(low=-50, high=50, shape=(3,), dtype='float32')
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
        })

        self.use_demonstrations = use_demonstrations
        self.demonstrations = None
        if self.use_demonstrations:
            self._load_demonstrations() # (CSV에서 데모 로드)

    def _load_demonstrations(self):
        # (데몬스트레이션 로드 부분 수정)
        # pumping_interpolated_trajectory.csv에서 데이터 로드
        demo_path = "./pumping_interpolated_trajectory.csv"
        if os.path.exists(demo_path):
            # CSV 로드 (예: 구분자 콤마, 첫 줄에 헤더가 없다고 가정)
            # 필요한 경우 skiprows나 delimiter 수정 가능
            self.demonstrations = np.loadtxt(demo_path, delimiter=",", skiprows=1) 
            # skiprows=1: 첫 번째 행을 헤더로 가정하고 건너뛰는 경우
            # CSV 파일 형식에 따라 적절히 수정

            print("Demonstration data loaded:", self.demonstrations.shape)
        else:
            print("No demonstration data found. Proceeding without it.")
            self.demonstrations = None

    def step(self, action):
        self.num_steps += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.arm.set_position(*action, relative=True, wait=True)
        
        obs = self._get_obs()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'future_length': self._max_episode_steps - self.num_steps
        }

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info, action)
        done = self.num_steps == self._max_episode_steps
        return obs, reward, done, False, info

    def reset(self, seed=None, maybe_options=None, options2=None):
        super(XArmEnv, self).reset(seed=seed, options=maybe_options)
        self.arm.set_servo_angle(angle=[118.61103,45.350011,45.358376,2.625751,-73.658767,-76.091316], wait=True)
        import time
        time.sleep(5)
        self.goal = self._sample_goal()
        self.num_steps = 0
        return self._get_obs(), {}

    def compute_reward(self, achieved_goal, desired_goal, info, action=None):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        reward = -distance
        if info['is_success']:
            reward += 10

        # 액션 크기 패널티
        if action is not None:
            reward -= 0.001 * np.linalg.norm(action)

        # 데모 기반 shaping
        if self.demonstrations is not None:
            # demonstrations 가 (N, D) 형태라고 가정
            # 여기서는 첫 3개 컬럼이 x, y, z 좌표라고 가정(파일 형식에 맞게 수정)
            demo_positions = self.demonstrations[:, :3]
            agent_pos = achieved_goal[:3]
            # 랜덤 샘플 5개 추출 (N이 5보다 클 것이라 가정)
            idx = np.random.choice(len(demo_positions), size=5, replace=True)
            sampled_positions = demo_positions[idx]
            demo_dist = np.mean([np.linalg.norm(agent_pos - dp) for dp in sampled_positions])
            reward -= 0.001 * demo_dist

        return reward

    def render(self):
        pass

    def close(self):
        self.arm.disconnect()

    def _get_obs(self):
        angle_state = self.arm.get_servo_angle()[1]
        coordinates = self.arm.get_position(is_radian=False)[1]
        external_force = self.arm.ft_ext_force
        gripper_state = self.arm.get_suction_cup()

        obs = np.concatenate([angle_state, coordinates, gripper_state, external_force])
        return {
            'observation': obs.copy(),
            'achieved_goal': np.squeeze(coordinates.copy()),
            'desired_goal': self.goal.copy(),
        }

    def _sample_goal(self):
        goal = np.array(self.goal_space.sample())
        return goal

    def _is_success(self, achieved_goal, desired_goal):
        return np.linalg.norm(achieved_goal - desired_goal) < 0.1

if __name__ == "__main__":
    env = XArmEnv(use_demonstrations=True)
    obs = env.reset()
    print(obs)
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(reward)
        if done:
            break
    env.close()
