import numpy as np
import os
from pykalman import KalmanFilter

try:
    from trajectory import Trajectory  # 실행 코드일 경우
except ImportError:
    from src.trajectory import Trajectory  # 모듈 코드인 경우

# 칼만 필터 적용 함수
# :param: Trajectory, 강도 조절 패러미터 2개
# :return: Trajectory(Filtered)
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

    # CSV로부터 Trajectory 객체 생성
    traj = Trajectory.load_csv("pumped_sumin_a")

    # 궤적에 칼만 필터 적용
    traj_kalman = kalman_filter(traj, process_var=0.0001, measurement_var=1.0)

    # 시각화
    Trajectory.show(traj, traj_kalman)

    # 궤적 저장
    traj_kalman.save_csv("processed_sumin_a.csv")

