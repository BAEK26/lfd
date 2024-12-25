import gym
import math
import numpy as np
from gymnasium import spaces
from xarm.wrapper import XArmAPI
from gymnasium_robotics.core import GoalEnv

JOINT_LIMIT =  [
    (-2 * math.pi, 2 * math.pi),
    (-2.61799, 2.61799),
    (-0.0610865, 5.23599),
    (-2 * math.pi, 2 * math.pi),
    (-2.16421, 2.16421),
    (-2 * math.pi, 2 * math.pi)
]

class XArmEnv(GoalEnv):
    def __init__(self):
        super(XArmEnv, self).__init__()
        self.arm = XArmAPI('192.168.1.194')  
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        print('initializing with state:', self.arm.get_state())


        self._max_episode_steps = 100
        #self.num_steps = 0

        #TODO: goal space to be changed to tilting area
        self.goal_space = spaces.Box(
            low=np.array([260, 190, 250, -170, -90, 0]), 
            high=np.array([270, 200, 260, -155, -80, 10]), 
            dtype='float32'
        )
        self.goal = self._sample_goal()

        obs = self._get_obs()
        # 액션 및 관찰 공간 정의
        # action space: x, y, z displacement -50~50 TODO: roll pitch yaw -> 갑자기 변할 수 있음..
        # self.action_space = spaces.Box(low=-50, high=50, shape=(3,), dtype='float32')
        self.action_space = spaces.Box(low=-50, high=50, shape=(3,), dtype='float32')
        # observation space: joint angles
        # self.observation_space = spaces.Dict(dict( #TODO: change to actual observation space
        #     observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        #     achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        #     desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        # ))
        self.observation_space = spaces.Dict({ #TODO: change to actual observation space
            'observation':spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            'achieved_goal':spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            'desired_goal':spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        })
        self.achieved_goal_index = len(obs['observation']) 
        self.achieved_goal_index = len(obs['observation']) + len(obs['achieved_goal'])

    def step(self, action):
        self.num_steps += 1
        # 액션 실행
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.arm.set_position(*action, relative=True, wait=True)
        
        # 새로운 상태 관찰
        obs = self._get_obs()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'future_length': self._max_episode_steps - self.num_steps
        }
        
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        done = self.num_steps == self._max_episode_steps
        
        return obs, reward, done,None, info




    def reset(self, seed=None, maybe_options:dict | None = None, options2 = None):
        super(XArmEnv, self).reset(seed=seed, options=maybe_options)
        # self.arm.reset(wait=True)
        self.arm.set_servo_angle(angle=[118.61103,45.350011,45.358376,2.625751,-73.658767,-76.091316], wait=True)
        print("RESET.....", end="")
        import time
        time.sleep(5)
        print("...DONE")
        self.goal = self._sample_goal()
        self.num_steps = 0
        return self._get_obs(), {}


    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        reward = -distance  # 거리 기반 보상
        if info['is_success']:
            reward += 10  # 목표 도달 시 보상 추가
        return reward


    def render(self):
        pass

    def close(self):
        self.arm.disconnect()

    def _set_action(self, action):
        pass

    def _get_obs(self):
        angle_state = self.arm.get_servo_angle()[1]
        coordinates = self.arm.get_position(is_radian=False)[1]
        external_force = self.arm.ft_ext_force
        gripper_state = self.arm.get_suction_cup()

        obs = np.concatenate([angle_state, coordinates, gripper_state, external_force])

        # print("achieved_goal", coordinates)
        # print("desired_goal", self.goal)
        return {
            'observation': obs.copy(),
            'achieved_goal': np.squeeze(coordinates.copy()),
            'desired_goal': self.goal.copy(),
            "angle_state": angle_state.copy(),
            "external_force": external_force.copy(),
            "gripper_state": gripper_state.copy()
        }
    
    def _reset_sim(self):
        pass

    def _sample_goal(self):
        goal = np.array(self.goal_space.sample())
        return goal

    def _is_success(self, achieved_goal, desired_goal):
        return np.linalg.norm(achieved_goal - desired_goal) < 0.1


if __name__ == "__main__":
    env = XArmEnv()
    obs = env.reset()
    print(obs)
    for _ in range(1000):
        action = env.action_space.sample()
        # print("action", action)
        obs, reward, done, _ = env.step(action)
        print( reward)
        if done:
            break
    env.close()