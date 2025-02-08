# 전처리 및 후처리를 위한 기능들이 모여 있습니다.

import numpy as np
from pykalman import KalmanFilter
from scipy.interpolate import interp1d

try:
    from trajectory import Trajectory  # 실행 코드일 경우
    from utils import Visualize
except ImportError:
    from src.trajectory import Trajectory  # 모듈 코드인 경우


class Process:
    def __init__(self):
        pass

    # CSV 데이터를 펌핑합니다.
    # :param: 목표율(기본값은 초당 360개)
    # :return: trajectoryectory 객체
    def pumping_data(trajectory, target_rate=360):
        interval_ms = 1000 / target_rate
        new_timestamp = np.arange(trajectory.timestamp[0], trajectory.timestamp[-1], interval_ms)
        
        interpolated_xyz = np.array([
            np.interp(new_timestamp, trajectory.timestamp, trajectory.xyz[:, i]) for i in range(3)
        ]).T
        interpolated_euler = np.array([
            np.interp(new_timestamp, trajectory.timestamp, trajectory.euler_angles[:, i]) for i in range(3)
        ]).T
        interpolated_joints = np.array([
            np.interp(new_timestamp, trajectory.timestamp, trajectory.joints[:, i]) for i in range(trajectory.joints.shape[1])
        ]).T
        interpolated_gripper = np.round(np.interp(new_timestamp, trajectory.timestamp, trajectory.gripper)).astype(int)        
        
        return Trajectory(new_timestamp, interpolated_xyz, interpolated_euler, interpolated_joints, interpolated_gripper)
    

    # 칼만 필터 적용 함수
    # :param: Trajectory, 강도 조절 인자 2개
    # :return: Filtered Trajectory
    @staticmethod
    def kalman_filter(trajectory, process_var=1e-4, measurement_var=1e-1):
        if trajectory.target not in ['C', 'E', 'J']:
            raise Exception("Trajectory의 Target은 C(cartesian), E(euler), J(joint) 중 하나여야 합니다.")

        # target에 맞게 대상 데이터 및 차원 설정
        data, state_dim = {'C':[trajectory.xyz, 3], 'E':[trajectory.euler_angles, 3], 'J':[trajectory.joints, 6]}[trajectory.target]

        kf = KalmanFilter(
            transition_matrices=np.eye(state_dim),  # 상태 전이 행렬 (항등 행렬)
            observation_matrices=np.eye(state_dim),  # 관측 행렬 (항등 행렬)
            transition_covariance=process_var * np.eye(state_dim),  # 프로세스 노이즈
            observation_covariance=measurement_var * np.eye(state_dim),  # 측정 노이즈
            initial_state_mean=data[0],  # 초기 상태값 (첫 번째 데이터)
            initial_state_covariance=np.eye(state_dim)  # 초기 상태 공분산
        )

        filtered_state_means, _ = kf.filter(data)  # 전체 범위 필터링

        # 새 궤적을 복사해 변경 사항 덮어씌우기
        filtered_trajectory = trajectory.copy()
        if trajectory.target == 'C':
            filtered_trajectory.xyz = filtered_state_means
        elif trajectory.target == 'E':
            filtered_trajectory.euler_angles = filtered_state_means
        elif trajectory.target == 'J':
            filtered_trajectory.joints = filtered_state_means

        return filtered_trajectory

    # 특정 구간에 대해 보간하는 함수
    @staticmethod
    def apply_interpolation_segmented(trajectory, start_idx, end_idx, ref_start, ref_end):
        if trajectory.target not in ['C', 'E', 'J']:
            raise Exception("Trajectory의 Target은 C(cartesian), E(euler), J(joint) 중 하나여야 합니다.")

        # target에 따른 대상 데이터 설정
        data = {'C':trajectory.xyz, 'E':trajectory.euler_angles, 'J':trajectory.joints}[trajectory.target]

        # timestamp 및 데이터 배열 절편들을 연결
        timestamps = np.concatenate((trajectory.timestamp[ref_start:start_idx], trajectory.timestamp[end_idx+1:ref_end+1]))
        reference_data = np.concatenate((data[ref_start:start_idx], data[end_idx+1:ref_end+1]))
        
        # 보간
        interpolator = interp1d(timestamps, reference_data, axis=0, kind='cubic', fill_value='extrapolate')
        interpolated_values = interpolator(trajectory.timestamp[start_idx:end_idx+1])
        
        # 새 궤적을 복사해 변경 사항 덮어씌우기
        new_trajectory = trajectory.copy()
        if trajectory.target == 'C':
            new_trajectory.xyz[start_idx:end_idx+1] = interpolated_values
        if trajectory.target == 'E':
            new_trajectory.euler_angles[start_idx:end_idx+1] = interpolated_values
        elif trajectory.target == 'J':
            new_trajectory.joints[start_idx:end_idx+1] = interpolated_values

        return new_trajectory
    
    # 사용자 입력에 따라 보간하는 함수 
    # :param: Trajectory
    # :return: Interpolated Trajectory
    def interpolate(trajectory):

        # 수정 범위 입력받기
        while True:
            print(f"Timestamp 범위: 0 - {len(trajectory.timestamp) - 1}")
            try:
                start_idx = int(input("수정할 영역 시작 인덱스 입력: "))
                end_idx = int(input("수정할 영역 끝 인덱스 입력: "))
                if start_idx < 0 or end_idx >= len(trajectory.timestamp) or start_idx > end_idx:
                    raise ValueError
            except ValueError:
                print("잘못된 입력입니다. 다시 입력해주세요.")
                continue

            # 확인하기
            Trajectory.show(trajectory, trajectory[start_idx:end_idx+1])   
            confirm = input("이 범위로 보간을 적용할까요? (y/n): ")
            if confirm.lower() == 'y':
                break
        
        # 참조 범위 입력받기
        while True:
            try:
                ref_start = int(input("참조할 영역 시작 인덱스 입력: "))
                ref_end = int(input("참조할 영역 끝 인덱스 입력: "))
                if ref_start < 0 or ref_end >= len(trajectory.timestamp) or ref_start > ref_end:
                    raise ValueError
            except ValueError:
                print("잘못된 입력입니다. 다시 입력해주세요.")
                continue
            
            # 확인하기
            Trajectory.show(trajectory, trajectory[start_idx:end_idx+1], trajectory[ref_start:ref_end+1])
            confirm = input("이 참조 영역으로 진행할까요? (y/n): ")
            if confirm.lower() == 'y':
                break
        
        # 선정 구간에 대해 필터 적용
        interpolated_trajectory = Process.apply_interpolation_segmented(trajectory, start_idx, end_idx, ref_start, ref_end)
    
        # Joint 대상일 때는 시뮬레이션도 겸함
        if trajectory.target == 'j':
            Trajectory.show(interpolated_trajectory)
            Visualize.simulate(trajectory)
        else:
            Trajectory.show(trajectory, interpolated_trajectory)
        
        confirm = input("이 결과를 사용할까요? (y/n): ")
        if confirm.lower() == 'y':
            return interpolated_trajectory
        else:
            return trajectory
        


# 실행 코드
if __name__ == "__main__":

    # CSV로부터 Trajectory 객체 생성
    traj = Trajectory.load_csv("test_sumin_a")

    # 궤적 펌핑하기
    traj_pumped = Process.pumping_data(traj)
    print(traj_pumped.target)

    # 궤적의 XYZ 성분에 칼만 필터 적용
    traj_kalman = Process.kalman_filter(traj_pumped)
    Trajectory.show(traj_pumped, traj_kalman)

    # 궤적의 joint 성분에 제거-후-보간 적용
    traj_kalman.target = 'J'
    traj_interpolate = Process.interpolate(traj_kalman)
    Trajectory.show(traj_interpolate, traj_kalman)

