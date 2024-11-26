import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pykalman import KalmanFilter
import sys

# 파일 경로
sys.path.append('..\LfD_DMPs')  # 상위 디렉토리로 이동 후 LfD_DMPs 폴더로 경로 지정
filepath = './test.csv'

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
    initial_state = data.iloc[0] # 초기 상태 추정값. 여기선 데이터 첫번째 값을 초기상태로 설정함.
    observation_covariance = np.diag([1, 1, 1]) ** 2  # 예: 측정 노이즈
    transition_covariance = np.diag([0.1, 0.1, 0.1]) ** 2  # 예: 추정 노이즈 / 전이 노이즈 공분산 / 필터가 예측값에만 의존
    transition = np.eye(3)  # 상태 전이 행렬

    kf = KalmanFilter(initial_state_mean=initial_state,
                      initial_state_covariance=observation_covariance, #초기 상태 공분산 (초기상태 불확실성을 의미/ 클수록 덜신뢰)
                      observation_covariance=observation_covariance, # 데이터 노이즈 수준에 맞게 조정하기 (높이면 데이터 신뢰도 떨어짐, 예측에 의존)
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

    # 처리된 데이터 출력
    print(processed_data.head())
    
    # kalman1.csv로 저장
    output_filepath = './kalman1.csv'
    processed_data.to_csv(output_filepath, index=True)  # 인덱스 포함 저장
    print(f"Processed data saved to {output_filepath}")
