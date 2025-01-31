import gym
import math
import numpy as np
from gymnasium import spaces
from xarm.wrapper import XArmAPI
from gymnasium_robotics.core import GoalEnv
import os
# 리워드 고도화, 액션 변경
import torch
from torch.utils.tensorboard import SummaryWriter

print(torch.cuda.is_available())  # True가 출력되어야 함
print(torch.cuda.get_device_name(0))  # GPU 이름 출력


# JOINT_LIMIT = [
#     (-math.pi, math.pi),        # Joint 1
#     (-math.pi / 2, math.pi / 2),  # Joint 2
#     (-math.pi / 2, math.pi / 2),  # Joint 3
#     (-math.pi, math.pi),        # Joint 4
#     (-math.pi / 2, math.pi / 2),  # Joint 5
#     (-math.pi, math.pi),        # Joint 6
# ]

JOINT_LIMIT = [
    (-180, 180),        # Joint 1
    (-180 / 2, 180 / 2),  # Joint 2
    (-180 / 2, 180 / 2),  # Joint 3
    (-180, 180),        # Joint 4
    (-180 / 2, 180 / 2),  # Joint 5
    (-180, 180),        # Joint 6
]
#길이 6정도 리스트고, 각 원소는 min_limit, max_limit 형태 튜플임.

# 2. 초기 위치 정의 (CSV에서 첫째 자리까지 반영)
INITIAL_POSITION = {
    "x": 263.5,
    "y": 194.0,
    "z": 254.5,
    "roll": -174.9,
    "pitch": -84.8,
    "yaw": 18.4,
    "joint_angles": [39.6, 26.0, 52.7, 19.1, -59.2, -8.0]
}

class XArmEnv(GoalEnv):
    def __init__(self, use_demonstrations=True, max_episodes_for_demo=1000):
        super(XArmEnv, self).__init__()
        self.arm = XArmAPI()
        self.arm.connect('127.0.0.1')
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
            
    def _sample_goal(self):
        """
        목표(goal) 공간 내에서 무작위로 목표를 샘플링하는 메서드
        """
        goal = np.random.uniform(self.goal_space.low, self.goal_space.high)
        return goal

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
        # 이전 상태와 현재 상태 계산
        prev_state = self._get_obs()
        
        joint_angles = self.arm.get_servo_angle()[1]
        
        # 액션 적용
        #scaled_action = action * 0.2  # 액션 크기 축소
        action *= 0.2
        action = np.clip(action, self.action_space.low, self.action_space.high)
        try:
            self.arm.set_position(*action, relative=True, wait=True)
            
        except Exception as e:
            print(f"Kinematic Error during set_position: {e}")
            return prev_state, -20, True, False, {"error": "kinematic error"}

    
        #except Exception as e:
            #print(f"Kinematic Error during set_position: {e}")
            # 복구 동작: 안전한 초기 상태로 복귀
            #action *= 0.5
            #self.arm.set_servo_angle(angle=[0, 0, 0, 0, 0, 0], wait=True)
            #try : 
            #    self.arm.set_position(*action, relative=True, wait=True)
            
            #except Exception as e:
                # 액션을 더 작게 줄였는데도 실패하면 에피소드 종료
            #    print(f"Failed even after scaling down action: {e}")
            
            #return prev_state, -20, True, False, {"error": "kinematic error"}
            # 새로운 상태를 가져오고, 시간에 따른 변화 반영
            # done = True
            #reward = -10
            #obs = self._get_obs()
            #info = {'collision': True}
            #return obs, reward, done, False, info
        
        curr_state = self._get_obs()
        
        # 제한 초과 여부 확인
        
        joint_angles = self.arm.get_servo_angle()[1]
        out_of_range = False
        for angle, (min_limit, max_limit) in zip(joint_angles, JOINT_LIMIT):
            if not (min_limit <= angle <= max_limit):
                out_of_range = True
                break

        # 제한 초과 시 복구
        #if out_of_range:
        #    print("Warning: Joint angle out of range. Recovering...")
        #    self.arm.set_servo_angle(angle=INITIAL_POSITION["joint_angles"], wait=True)
        #    penalty = -10
        #    return curr_state, penalty, True, False, {"error": "joint limit exceeded"}
        
         # 제한 초과 시 복구 시도
        if out_of_range:
            print("Warning: Joint angle out of range. Attempting recovery...")
            recovery_action = [
                max(min(angle, max_limit), min_limit)
                for angle, (min_limit, max_limit) in zip(joint_angles, JOINT_LIMIT)
            ]
            self.arm.set_servo_angle(angle=recovery_action, wait=True)
            penalty = -10  # 패널티를 완화 (원래는 -5였음음)
            return curr_state, penalty, False, False, {"error": "joint limit exceeded"}

        
        #이거 너무 궁금한데, 복구를 시켜버려야 하는지 아니면 패널티를 줘야하는지 모르겠음. 범위 벗어났을 때...
        #일단은 이거 패널티만 줄게여
        # penalty ===> 복구?
        #if out_of_range:
            #print("Warning: Joint angle out of range. Applying penalty...")
            #penalty = -10
            #return curr_state, penalty, False, False, {"error": "joint limit exceeded"}


        # 외력으로 충돌 감지
        # ext_force = self.arm.ft_ext_force
        ext_force = curr_state["external_force"]
        force_threshold = 10.0
        collision_detected = np.linalg.norm(ext_force) > force_threshold
        if collision_detected:
            print("Collision detected based on external force.")
            return curr_state, -15, False, False, {"collision": True}
        
         # 보상 및 종료 조건 계산
        delta_state = curr_state['observation'] - prev_state['observation']
    
        # 진행 상황 및 목표 도달 여부를 계산
        info = {
            'is_success': self._is_success(curr_state['achieved_goal'], self.goal),
            'future_length': self._max_episode_steps - self.num_steps,
            'state_change': np.linalg.norm(delta_state),  # 상태 변화량
            'collision': collision_detected,
            "angle_state": curr_state["angle_state"],
            "external_state":ext_force
        }  # 충돌 여부 추가}

        # 보상 계산
        reward, reward_dict = self.compute_reward(curr_state['achieved_goal'], self.goal, info, action)
        info.update({'reward_dict':reward_dict})
        # 종료 여부 체크
        done = self.num_steps == self._max_episode_steps
        return curr_state, reward, done, False, info

    def reset(self, seed=None, maybe_options=None, options2=None):
        super(XArmEnv, self).reset(seed=seed, options=maybe_options)
        #self.arm.set_servo_angle(angle=INITIAL_POSITION["joint_angles"], wait=True)
        import time
        time.sleep(5)
        
        # 시연 데이터를 초기화에 활용 (초기 몇 에피소드만)
        if self.use_demonstrations and self.current_episode < self.max_episodes_for_demo:
            if self.demonstrations is not None:
                demo_index = np.random.randint(len(self.demonstrations))
                demo_state = self.demonstrations[demo_index]
                # self.arm.set_servo_angle(angle=demo_state[6:], wait=True)
                self.arm.set_servo_angle(angle=demo_state[7:], wait=True)
                print(f"Reset using demonstration: {demo_state}")
            else:
                print("No demonstration data available for reset.")
                self.arm.set_servo_angle(angle=INITIAL_POSITION["joint_angles"], wait=True)
        else:
            # 기본 초기 위치로 이동
            self.arm.set_servo_angle(angle=INITIAL_POSITION["joint_angles"], wait=True)

        self.goal = self._sample_goal()
        self.num_steps = 0
        # (주석) 에피소드 카운트 증가
        self.current_episode += 1
        return self._get_obs(), {}
  
    def compute_reward(self, achieved_goal, desired_goal, info, action=None):
        
        # 리워드 초기화
        joint_penalty_weight = 0.1
        distance_reward_weight = 5 # 5
        distance_penalty_weight = 0.1 # 1
        direction_reward_weight = 0.01
        collision_penalty_weight = 10
        success_reward_weight = 10
        demo_reward_weight = 10
        action_penalty_weight = 0.001
        state_change_penalty_weight = 0.01
        consistency_reward_weight = 0.01
        total_reward = 0.0
        #관절 상태 기반 패널티 계산 (JOINT_LIMIT 활용)
        joint_penalty = 0.0
        try:
            # joint_angles = self.arm.get_servo_angle()[1]  # joint_angles는 [angle_1, angle_2, ..., angle_6] 형태라고 가정
            joint_angles = info["angle_state"]
            angle_excess = 0.0
            for angle, (min_limit, max_limit) in zip(joint_angles, JOINT_LIMIT):
                # 각도가 최소 범위보다 작을 경우 초과하는 만큼 페널티
                if angle < min_limit:
                    angle_excess += (min_limit - angle)
                # 각도가 최대 범위보다 클 경우 초과하는 만큼 페널티
                elif angle > max_limit:
                    angle_excess += (angle - max_limit)
            joint_penalty = -(joint_penalty_weight * angle_excess)
            total_reward += joint_penalty
        except Exception as e:
            print(f"Joint angle error: {e}")
            

        # 목표와 현재 위치(achieved_goal) 간 거리 (x, y, z만 사용) 맞는건지 확신은 없음
        position_distance = np.linalg.norm(achieved_goal[:3] - desired_goal[:3])
        max_position_distance = np.linalg.norm(self.goal_space.high[:3] - self.goal_space.low[:3])

        # 목표 거리 기반 보상
        
        distance_penalty = -(distance_penalty_weight * position_distance)
        total_reward += distance_penalty
        
        distance_reward = 0.0
        if position_distance < 0.1:
            distance_reward += distance_reward_weight * (0.1 - position_distance)
        elif position_distance < 0.5:
            distance_reward += 0.1 * distance_reward_weight * (0.5 - position_distance)
        total_reward += distance_reward
        
        # reward -= position_distance
        # if position_distance < 0.1:
        #     reward += 5 * (0.1 - position_distance)
        # elif position_distance < 0.5:
            # reward += 0.5 * (0.5 - position_distance)

        # 방향성 보상 계산 시 목표와 현재 성취 좌표의 첫 3차원만 사용
        direction_reward = 0.0
        if action is not None and position_distance > 1e-3:
            direction_vector = desired_goal[:3] - achieved_goal[:3]
            direction_norm = np.linalg.norm(direction_vector)
            if direction_norm > 1e-6:
                # direction_vector와 action을 이용한 방향성 보상
                direction_reward += direction_reward_weight * np.dot(action, direction_vector) / direction_norm
                # reward += 0.01 * direction_reward

                # 단위벡터로 정규화한 방향성 보상 (이거 계산 잘못해서 0나오면 에러생긴다고 해서 넣음.)
                direction_unit_vector = direction_vector / direction_norm
                action_norm = np.linalg.norm(action)
                if action_norm > 1e-6:
                    action_unit_vector = action / action_norm
                    direction_reward += direction_reward_weight * np.dot(action_unit_vector, direction_unit_vector)
                    # reward += 0.01 * direction_reward
        total_reward += direction_reward
        
        # 데몬스트레이션 기반 보상 (여기서도 direction 관련 계산 시 첫 3차원만 사용)
        demo_reward = 0.0
        if self.demonstrations is not None:
            demo_weight = max(0, (1 - (self.current_episode - self.max_episodes_for_demo) / self.max_episodes_for_demo))
            if demo_weight > 0:
                demo_positions = self.demonstrations[:, :3]
                agent_pos = achieved_goal[:3]
                max_distance = np.linalg.norm(self.goal_space.high[:3] - self.goal_space.low[:3])
                # 여기서도 desired_goal[:3]와 achieved_goal[:3] 사용
                dg_ag_vector = (desired_goal[:3] - achieved_goal[:3])
                dg_ag_norm = np.linalg.norm(dg_ag_vector)
                if action is not None and dg_ag_norm > 1e-6:
                    direction_reward = np.dot(action, dg_ag_vector) / dg_ag_norm
                    similarity = 1 - (np.linalg.norm(agent_pos - demo_positions, axis=1).min() / max_distance)
                    demo_reward += demo_reward_weight * demo_weight * similarity
        
        total_reward += demo_reward
        # 충돌 페널티
        collision_penalty = 0.0
        if info.get("collision", False):
            collision_penalty = -(collision_penalty_weight)
            # reward -= 10
        total_reward += collision_penalty
        
        # 관절 상태 기반 패널티 : 중복 계산이라서 뺌. 위에서 일단 함.
        #try:
            #joint_angles = self.arm.get_servo_angle()[1]
            #joint_penalty = sum(max(0, abs(angle) - limit) for angle, limit in zip(joint_angles, JOINT_LIMIT[0]))
            #reward -= 0.1 * joint_penalty
        #except Exception as e:
            #print(f"Joint angle error: {e}")

        # 목표 도달 시 큰 보상
        success_reward = 0.0
        if info.get('is_success', False):
            success_reward = success_reward_weight
            # reward += 10
        total_reward += success_reward
        
        # 액션 크기 및 변화 패널티 - 액션 반영이 안되는것 같아서 넣어봄봄
        action_penalty = 0.0
        if action is not None:
            action_penalty += -(action_penalty_weight * np.linalg.norm(action))
            # reward -= 0.001 * np.linalg.norm(action)
            if hasattr(self, 'prev_action'):
                action_penalty += -(action_penalty_weight * 10 * np.linalg.norm(action - getattr(self, 'prev_action', np.zeros_like(action))))
            self.prev_action = action
        total_reward += action_penalty
        # 상태 변화량 패널티 : 이거는 너무 크게 움직이면 넣자.
        # 로봇이 큰 변화 보이면 페널티 주는거. 불필요한 움직임 억제.
        # info 딕셔너리에도 넣긴 했음 (나중에 열어서 봐봐)
        #delta_state는 이전 관측값과 현재 관측값(관절 상태, 좌표 등)의 차이
        #np.linalg.norm(delta_state)는 벡터의 크기(즉, 상태 변화량)를 계산
    
        #고민하다가, 로봇이 너무 크게 흔들리고 난리칠까봐 일단 씀
        state_change_penalty = 0.0
        state_change = info.get('state_change', 0)
        if state_change > 0:
            state_change_penalty = -(state_change_penalty_weight * state_change)
        total_reward += state_change_penalty
    

        # 일관성 보상: 이전 거리 대비 현재 목표 거리 감소 시 보상
        consistency_reward = 0.0
        if self.num_steps > 0:
            prev_distance = getattr(self, 'prev_distance', max_position_distance)
            consistency_reward = consistency_reward_weight * (prev_distance - position_distance)
            self.prev_distance = position_distance
        total_reward += consistency_reward
        reward_dict = {'joint_penalty': -joint_penalty, 'distance_penalty': -distance_penalty,
                       'distance_reward': distance_reward, 'direction_reward': direction_reward,
                       'demo_reward': demo_reward, 'collision_penalty': -collision_penalty,
                       'success_reward': success_reward, 'action_penalty': -action_penalty,
                       'state_change_penalty': -state_change_penalty, 'consistency_reward': consistency_reward, 'position_distance': position_distance}
         
        # 리워드 클리핑
        return total_reward, reward_dict
        # return np.clip(total_reward, -10, 20), reward_dict



    def render(self):
        pass

    def close(self):
        self.arm.disconnect()

    def _get_obs(self):
        angle_state = self.arm.get_servo_angle()[1]
        coordinates = self.arm.get_position(is_radian=False)[1]
        external_force = self.arm.ft_ext_force
        gripper_state = list(self.arm.get_suction_cup())
        
        obs = np.concatenate([angle_state, coordinates, gripper_state, external_force])
        return {
            'observation': obs.copy(),
            'achieved_goal': np.squeeze(coordinates.copy()),
            'desired_goal': self.goal.copy(),
            "angle_state": angle_state.copy(),
            "external_force":external_force.copy(),
            "gripper_state":gripper_state.copy()
            
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