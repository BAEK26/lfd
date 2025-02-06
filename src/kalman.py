import numpy as np
import os
import sys
from pykalman import KalmanFilter

try:
    from trajectory import Trajectory  # 직접 실행할 경우
except ImportError:
    from src.trajectory import Trajectory  # main.py에서 실행할 경우

def kalman_filter(trajectory, process_var=1e-4, measurement_var=1e-1):
    if trajectory.target == 'euler':
        data = trajectory.euler_angles
        state_dim = 3
    elif trajectory.target == 'joint':
        data = trajectory.joints
        state_dim = 6
    else:
        data = trajectory.xyz
        state_dim = 3

    kf = KalmanFilter(
        transition_matrices=np.eye(state_dim),  # 상태 전이 행렬 (항등 행렬)
        observation_matrices=np.eye(state_dim),  # 관측 행렬 (항등 행렬)
        transition_covariance=process_var * np.eye(state_dim),  # 프로세스 노이즈
        observation_covariance=measurement_var * np.eye(state_dim),  # 측정 노이즈
        initial_state_mean=data[0],  # 초기 상태값 (첫 번째 데이터)
        initial_state_covariance=np.eye(state_dim)  # 초기 상태 공분산
    )

    filtered_state_means, _ = kf.filter(data)  # 전체 범위 필터링


    filtered_trajectory = trajectory.copy()

    if trajectory.target == 'E':
        filtered_trajectory.euler_angles = filtered_state_means
    elif trajectory.target == 'J':
        filtered_trajectory.joints = filtered_state_means
    else:
        filtered_trajectory.xyz = filtered_state_means

    return filtered_trajectory


# 실행 코드
if __name__ == "__main__":

    base_dir = r"C:\Users\박수민\Documents\neoDMP" # base 경로 (알맞게 수정)
    load_path = os.path.join(base_dir, "data", "pumped_sumin_a.csv") # CSV 로드 파일 경로
    
    # CSV로부터 Trajectory 객체 생성
    traj = Trajectory.load_csv(load_path)

    # 궤적에 칼만 필터 적용
    traj_kalman = kalman_filter(traj, process_var=0.0001, measurement_var=1.0)

    # 시각화
    Trajectory.show(traj, traj_kalman)

    # 궤적 저장
    save_path = os.path.join(base_dir, "data", "processed_sumin_a.csv") # CSV 저장 파일 경로
    traj_kalman.save_csv(save_path)

