import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pykalman import KalmanFilter
from mpl_toolkits.mplot3d import Axes3D  # 3D plot을 위한 모듈
import sys

# 파일 경로
sys.path.append('..\LfD_DMPs')  # 상위 디렉토리로 이동 후 LfD_DMPs 폴더로 경로 지정
filepath = './test.csv'

def load_data(filepath):
    """ CSV 파일로부터 데이터 로드 """
    return pd.read_csv(filepath)

def interpolate_data(data, num_points=None):
    """ 데이터 인터폴레이션 """
    if num_points is None:
        num_points = len(data)  # 데이터의 전체 길이를 사용
    t_new = np.linspace(data.index[0], data.index[-1], num_points)
    x_interpolated = interp1d(data.index, data['x'], kind='cubic')(t_new)
    y_interpolated = interp1d(data.index, data['y'], kind='cubic')(t_new)
    z_interpolated = interp1d(data.index, data['z'], kind='cubic')(t_new)
    return pd.DataFrame({'x': x_interpolated, 'y': y_interpolated, 'z': z_interpolated}, index=np.round(t_new).astype(int))

def apply_kalman_filter(data):
    """XYZ 데이터에 대해 하나의 칼만 필터 적용"""
    initial_state = data.iloc[0].values  # 초기 상태 추정값
    observation_covariance = np.diag([0.005, 0.003, 0.005]) ** 2  # 예: 측정 노이즈
    transition_covariance = np.diag([0.01, 0.01, 0.01]) ** 2  # 예: 추정 노이즈
    transition = np.eye(3)  # 상태 전이 행렬

    kf = KalmanFilter(
        initial_state_mean=initial_state,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition
    )

    states_pred, _ = kf.smooth(data[['x', 'y', 'z']])
    return pd.DataFrame(states_pred, columns=['x', 'y', 'z'])

def process_and_save_data(filepath, output_filepath):
    """ 데이터 처리 및 저장 """
    data = load_data(filepath)
    interpolated_data = interpolate_data(data[['x', 'y', 'z']])
    filtered_xyz = apply_kalman_filter(interpolated_data)

    # 기존의 조인트 및 roll, pitch, yaw 데이터를 인덱스 기반으로 맞추고 병합
    joint_and_rpy_data = data[['roll', 'pitch', 'yaw', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']]
    full_data = pd.concat([filtered_xyz, joint_and_rpy_data.reset_index(drop=True)], axis=1)
    full_data.to_csv(output_filepath, index=False)
    return full_data



def plot_data_3d(original_filepath, filtered_filepath):
    # 원본 데이터 로드
    original_data = pd.read_csv(original_filepath)
    # 필터링된 데이터 로드
    filtered_data = pd.read_csv(filtered_filepath)

    # 데이터 플로팅
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 원본 데이터 플로팅
    ax.plot(original_data['x'], original_data['y'], original_data['z'], label='Original Data', color='blue', linestyle='--')

    # 필터링된 데이터 플로팅
    ax.plot(filtered_data['x'], filtered_data['y'], filtered_data['z'], label='Filtered Data', color='red')

    ax.set_title('3D Comparison of Original and Kalman Filtered Data')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Z coordinate')
    ax.legend()
    ax.grid(True)

    plt.show()


if __name__ == "__main__":
    filepath = './test.csv'
    output_filepath = './neokalman2.csv'
    processed_data = process_and_save_data(filepath, output_filepath)

    # 처리된 데이터 출력
    print(processed_data.head())

    # csv로 저장
    processed_data.to_csv(output_filepath, index=True)  # 인덱스 포함 저장
    print(f"Processed data saved to {output_filepath}")

    # 데이터 시각화
    plot_data_3d(filepath, output_filepath)
