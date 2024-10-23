import gym
import numpy as np
from ufactory import uArm

class RobotArm:
    def __init__(self):
        # Lite6 초기화 (SDK 사용)
        self.arm = uArm.UFactoryLite6()
        self.arm.connect()  # 로봇 암 연결
        self.arm.initialize()  # 초기화
        self.state = np.zeros(6)  # 6축 로봇 암의 초기 상태 (관절 각도)

    def step(self, action):
        # SDK를 사용해 관절 각도 조정
        target_angles = np.clip(self.state + action, -180, 180)  # 각도 제한
        self.arm.set_joint_positions(target_angles.tolist())
        self.state = target_angles
        return self.state

    def get_state(self):
        # 로봇 암의 현재 관절 각도를 반환
        current_angles = self.arm.get_joint_positions()
        return np.array(current_angles)

    def disconnect(self):
        self.arm.disconnect()  # 로봇 암 연결 종료
