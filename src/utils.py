# 직접적인 연관은 없으나, 디버깅 과정에서 도움이 될 기능을 모아두는 파일입니다.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

try:
    from trajectory import Trajectory  # 직접 실행할 경우
except ImportError:
    from src.trajectory import Trajectory  # main.py에서 실행할 경우

# 시각화
class Visualize:
    # (theta, d, alpha, a)
    dh_params = [
        (0, 243.3, -90, 0),  # Joint 1
        (-90, 0, 180, 200),  # Joint 2
        (-90, 0, 90, 87),    # Joint 3
        (0, 227.6, 90, 0),   # Joint 4
        (0, 0, -90, 0),      # Joint 5
        (0, 61.5, 0, 0)      # Joint 6
    ]

    #DH 변환 행렬 계산
    @staticmethod
    def __dh_transform(theta, d, alpha, a):
        theta_rad = np.radians(theta)
        alpha_rad = np.radians(alpha)
        return np.array([
            [np.cos(theta_rad), -np.sin(theta_rad) * np.cos(alpha_rad), np.sin(theta_rad) * np.sin(alpha_rad), a * np.cos(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad) * np.cos(alpha_rad), -np.cos(theta_rad) * np.sin(alpha_rad), a * np.sin(theta_rad)],
            [0, np.sin(alpha_rad), np.cos(alpha_rad), d],
            [0, 0, 0, 1]
        ])

    # 로봇팔 시각화 함수
    # 로봇팔을 xyz 및 euler에 기반해 plotting합니다.
    # :param: 생략
    # :return: end effector의 위치
    @classmethod
    def plot_robot_arm(cls, ax, joints):
        T = np.eye(4)
        positions = [T[:3, 3]]

        for i, joints in enumerate(joints):
            theta, d, alpha, a = cls.dh_params[i]
            T_i = cls.__dh_transform(joints + theta, d, alpha, a)
            T = np.dot(T, T_i)
            positions.append(T[:3, 3])

        positions = np.array(positions)

        # Draw links
        for i in range(len(positions) - 1):
            ax.plot(
                [positions[i, 0], positions[i + 1, 0]],
                [positions[i, 1], positions[i + 1, 1]],
                [positions[i, 2], positions[i + 1, 2]],
                'k-', linewidth=2
            )

        return positions[-1]
    
    @classmethod
    def simulate(cls, trajectory, frame_rate=20):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        trajectory_points = []

        def update(frame):
            ax.cla()
            ax.set_title(f"Robot Arm Animation\nCurrent Time: {trajectory.timestamp[frame] / 1000:.03f} s")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Plot end-effector trajectory
            ax.plot(trajectory.xyz[:, 0], trajectory.xyz[:, 1], trajectory.xyz[:, 2], 'r--', label='Planned Trajectory')
            ax.plot(trajectory.xyz[0, 0], trajectory.xyz[0, 1], trajectory.xyz[0, 2], 'r^', label='Start Point') 
            ax.plot(trajectory.xyz[-1, 0], trajectory.xyz[-1, 1], trajectory.xyz[-1, 2], 'ro', label='End Point')
            
            # Get joint angles for the current frame
            joints = trajectory.joints[frame]
            end_effector_pos = cls.plot_robot_arm(ax, joints)

            # Store end-effector path
            if frame == 0:
                ax.legend()
                trajectory_points.clear()

            trajectory_points.append(end_effector_pos)
            traj_np = np.array(trajectory_points)

            ax.plot(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2], 'b-', label='End-Effector Path')


        frame_rate_ms = frame_rate / 1000 # ms 단위로 변환

        # 초당 frame_rate개 프레임이 표시된다면, 전체 timestamp에서는 몇 개 프레임이 사용되는지 
        frame_num = int(trajectory.timestamp[-1] * frame_rate_ms)

        frames = range(0, len(trajectory.timestamp), len(trajectory.timestamp) // frame_num)

        ani = FuncAnimation(fig, update, frames=frames, interval=int(1 / frame_rate_ms))
        plt.show()


def random_near_endpoints(trajectory, option='end', random_rate=0.1):
    """
    시작점(start)과 도착점(end) 사이의 거리를 100%로 했을 때,
    시작점 인근 10% 이내 또는 도착점 인근 10% 이내에 있는 임의의 점을 반환하는 함수.
    
    :param start: (x, y, z) 형태의 시작점 배열
    :param end: (x, y, z) 형태의 도착점 배열
    :param option: 'start' 또는 'end'를 선택하여 반환할 점의 위치를 결정
    :return: 선택된 범위 내의 임의의 점 (x, y, z)
    """
    start = trajectory.xyz[0]
    end = trajectory.xyz[-1]
    
    # 시작점과 도착점 사이의 거리 계산
    distance = np.linalg.norm(end - start)
    radius = random_rate * distance
    
    # 선택한 점의 중심 설정
    if option == 'start':
        center = start
    elif option == 'end':
        center = end
    else:
        raise ValueError("option must be 'start' or 'end'")
    
    while True:
        random_offset = np.random.uniform(-radius, radius, size=3)
        random_point = center + random_offset
        if np.linalg.norm(random_offset) <= radius:
            break
    
    return tuple(random_point)

# 예제 코드
if __name__ == "__main__":
    
    # CSV 불러와 Trajectory 객체 생성
    traj = Trajectory.load_csv("test_sumin_a.csv")
    Visualize.simulate(traj, frame_rate=15)
