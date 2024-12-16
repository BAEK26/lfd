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
    def __init__(self, use_demonstrations=True, max_episodes_for_demo=1000):
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
        self.current_episode = 0  # (주석) 현재 에피소드 카운트
        self.max_episodes_for_demo = max_episodes_for_demo  # (주석) 이 에피소드 수 이후로는 데모 사용량 감소

        if self.use_demonstrations:
            self._load_demonstrations() # (주석) CSV에서 데모 로드

    def _load_demonstrations(self):
        # (주석) demonstration csv 파일 로드
        demo_path = "./pumping_interpolated_trajectory.csv"
        if os.path.exists(demo_path):
            self.demonstrations = np.loadtxt(demo_path, delimiter=",", skiprows=1)
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
        # (주석) 에피소드 카운트 증가
        self.current_episode += 1
        return self._get_obs(), {}

    def compute_reward(self, achieved_goal, desired_goal, info, action=None):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        reward = -distance
        if info['is_success']:
            reward += 10

        # 액션 크기 패널티
        if action is not None:
            reward -= 0.001 * np.linalg.norm(action)

        # (주석) 데모 기반 shaping은 초기 에피소드에만 강하게, 이후 점진적 감소
        if self.demonstrations is not None:
            # 시연 가중치 비율 계산 (0 ~ 1 사이)
            # 예: max_episodes_for_demo까지는 가중치 1, 그 이후 선형 감소
            # 여기서는 max_episodes_for_demo 이전까지는 1, 이후부터는 0이 되도록 하는 단순한 스케줄 예시
            # 실제로는 선형 감소: 예) weight = max(0, 1 - (self.current_episode - max_episodes_for_demo)/max_episodes_for_demo)
            # 여기서는 단순히 max_episodes_for_demo 이하 에피소드면 1, 아니면 0
            if self.current_episode <= self.max_episodes_for_demo:
                demo_weight = 1.0
            else:
                # 이후에는 데모기반 shaping 꺼버림(제로샷)
                demo_weight = 0.0

            if demo_weight > 0:
                # demonstrations의 첫 3개가 (x,y,z)라고 가정
                demo_positions = self.demonstrations[:, :3]
                agent_pos = achieved_goal[:3]
                idx = np.random.choice(len(demo_positions), size=5, replace=True)
                sampled_positions = demo_positions[idx]
                demo_dist = np.mean([np.linalg.norm(agent_pos - dp) for dp in sampled_positions])
                # 데모 거리 기반 패널티를 demo_weight로 가중
                reward -= demo_weight * 0.001 * demo_dist

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
    env = XArmEnv(use_demonstrations=True, max_episodes_for_demo=1000)
    obs = env.reset()
    print(obs)
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(reward)
        if done:
            obs = env.reset()[0]
    env.close()

#env2는 데몬스트레이션 자체를 리워드로.
#env3는 데몬스트레이션 초기 탐색용용에 넣고, 이후는 제로샷 가능하게