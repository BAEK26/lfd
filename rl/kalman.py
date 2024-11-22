import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pykalman import KalmanFilter
from trajectory_processing import process_trajectory

# 파일 경로
filepath = 'D:\\Jeong-eun\\LfD_DMPs\\test.csv'
processed_data = process_trajectory(filepath)
print(processed_data)


def load_data(filepath):
    """ CSV 파일로부터 데이터 로드 """
    return pd.read_csv(filepath)

def interpolate_data(data, num_points=500):
    """ 데이터 인터폴레이션 """
    t_new = np.linspace(data.index[0], data.index[-1], num_points)
    x_interpolated = interp1d(data.index, data['x'], kind='cubic')(t_new)
    y_interpolated = interp1d(data.index, data['y'], kind='cubic')(t_new)
    z_interpolated = interp1d(data.index, data['z'], kind='cubic')(t_new)
    return pd.DataFrame({'x': x_interpolated, 'y': y_interpolated, 'z': z_interpolated}, index=t_new)

def apply_kalman_filter(data):
    """ 칼만 필터 적용 """
    initial_state = data.iloc[0]
    observation_covariance = np.diag([1, 1, 1]) ** 2  # 예: 측정 노이즈
    transition_covariance = np.diag([0.1, 0.1, 0.1]) ** 2  # 예: 추정 노이즈
    transition = np.eye(3)  # 상태 전이 행렬

    kf = KalmanFilter(initial_state_mean=initial_state,
                      initial_state_covariance=observation_covariance,
                      observation_covariance=observation_covariance,
                      transition_covariance=transition_covariance,
                      transition_matrices=transition)

    states_pred = kf.smooth(data)[0]
    return pd.DataFrame(states_pred, columns=['x', 'y', 'z'], index=data.index)

def process_trajectory(filepath):
    """ 궤적 데이터를 로드, 인터폴레이션 및 칼만 필터 적용 """
    data = load_data(filepath)
    data_interpolated = interpolate_data(data)
    data_smoothed = apply_kalman_filter(data_interpolated)
    return data_smoothed

if __name__ == "__main__":
    filepath = './test.csv'
    processed_data = process_trajectory(filepath)
    print(processed_data.head())
