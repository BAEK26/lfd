import gym
import math
import numpy as np
from gym import spaces
from xarm.wrapper import XArmAPI

JOINT_LIMIT =  [
    (-2 * math.pi, 2 * math.pi),
    (-2.042035, 2.024581),  
    (-0.061086523819801536, 3.92699),  
    (-2 * math.pi, 2 * math.pi),
    (-1.692969, 2.1642082724729685),  
    (-2 * math.pi, 2 * math.pi)
],

class XArmEnv(gym.GoalEnv):
    def __init__(self):
        super(XArmEnv, self).__init__()
        self.arm = XArmAPI('192.168.1.194')  
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        print('initializing with state:', self.arm.get_state())


        # self.goal_space = spaces.Box(low=-1, high=1, shape=(6,), dtype='float32')
        self.goal_space = spaces.Box(low=np.array([260, 190, 250, -170, -90, 0]), high=np.array([270, 200, 260, -155, -80, 10]), dtype='float32')
        self.goal = self._sample_goal()
        obs = self._get_obs()
        # 액션 및 관찰 공간 정의
        # action space: x, y, z displacement -50~50 TODO: roll pitch yaw -> 갑자기 변할 수 있음..
        self.action_space = spaces.Box(low=-50, high=2*50, shape=(3,), dtype='float32')
        # observation space: joint angles
        self.observation_space = spaces.Dict(dict( #TODO: change to actual observation space
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        ))

    def step(self, action):
        # 액션 실행
        self.arm.set_servo_angle(angle=action.tolist(), is_radian=False)
        
        # 새로운 상태 관찰
        obs = self.arm.get_servo_angle(is_radian=False)
        
        # 보상 계산 (예: 목표 위치까지의 거리)
        reward = -np.linalg.norm(np.array(obs) - np.array([0, 0, 0, 0, 0, 0]))
        
        # 에피소드 종료 조건
        done = False
        
        return obs, reward, done, {}

    def reset(self):
        self.arm.reset(wait=True)
        return self.arm.get_servo_angle(is_radian=False)

    def compute_reward(self, achieved_goal, desired_goal, info):
        pass

    def render(self):
        pass

    def close(self):
        self.arm.disconnect()

    def _set_action(self, action):
        pass

    def _get_obs(self):
        angle_state = self.arm.get_servo_angle()
        coordinates = self.arm.get_position(is_radian=False)
        
        gripper_state = self.arm.get_suction_cup() 

        obs = np.concatenate([angle_state, coordinates, gripper_state])


        return {
            'observation': obs.copy(),
            'achieved_goal': np.squeeze(coordinates.copy()),
            'desired_goal': self.goal.copy(),
        }
    
    def _reset_sim(self):
        pass

    def _sample_goal(self):
        goal = np.array(self.goal_space.sample())
        return goal

    def _is_success(self, achieved_goal, desired_goal):
        pass


if __name__ == "__main__":
    env = XArmEnv()
    obs = env.reset()
    print(obs)
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print(obs, reward)
        if done:
            break
    env.close()