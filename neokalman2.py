import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plot을 위한 모듈


class KalmanDataProcessor:
    def __init__(self, input_folder='./data', output_file='./data/neokalman2_processed.csv'):
        self.input_folder = input_folder
        self.output_file = output_file
        self.input_file = self.find_latest_file()

    def find_latest_file(self):
        """ 가장 최근에 생성된 test%d.csv 파일 찾기 """
        files = [f for f in os.listdir(self.input_folder) if f.startswith('test') and f.endswith('.csv')]
        files.sort()
        return os.path.join(self.input_folder, files[-1]) if files else None

    def load_data(self, filepath):
        """ CSV 파일로부터 데이터 로드 """
        return pd.read_csv(filepath)

    def interpolate_data(self, data, num_points=None):
        """ 데이터 인터폴레이션 """
        if num_points is None:
            num_points = len(data)
        t_new = np.linspace(data.index[0], data.index[-1], num_points)
        x_interpolated = interp1d(data.index, data['x'], kind='cubic')(t_new)
        y_interpolated = interp1d(data.index, data['y'], kind='cubic')(t_new)
        z_interpolated = interp1d(data.index, data['z'], kind='cubic')(t_new)
        return pd.DataFrame({'x': x_interpolated, 'y': y_interpolated, 'z': z_interpolated}, index=np.round(t_new).astype(int))

    def apply_kalman_filter(self, data):
        """ XYZ 데이터에 대해 칼만 필터 적용 """
        initial_state = data.iloc[0].values
        observation_covariance = np.diag([0.005, 0.003, 0.005]) ** 2
        transition_covariance = np.diag([0.01, 0.01, 0.01]) ** 2
        transition = np.eye(3)

        kf = KalmanFilter(
            initial_state_mean=initial_state,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition
        )

        states_pred, _ = kf.smooth(data[['x', 'y', 'z']])
        return pd.DataFrame(states_pred, columns=['x', 'y', 'z'])

    def process_and_save_data(self):
        """ 데이터 처리 및 저장 """
        if not self.input_file:
            print("No input file found.")
            return None

        print(f"Processing file: {self.input_file}")
        data = self.load_data(self.input_file)

        # Timestamp 가져오기
        timestamp = data['timestamp']

        # 데이터 처리
        interpolated_data = self.interpolate_data(data[['x', 'y', 'z']])
        filtered_xyz = self.apply_kalman_filter(interpolated_data)

        # 기존 데이터와 병합 (timestamp 포함)
        joint_and_rpy_data = data[['roll', 'pitch', 'yaw', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']]
        full_data = pd.concat([timestamp.reset_index(drop=True), filtered_xyz, joint_and_rpy_data.reset_index(drop=True)], axis=1)
        full_data.to_csv(self.output_file, index=False)

        print(f"Processed data saved to: {self.output_file}")
        return full_data

    def plot_data_3d(self):
        """ 3D 데이터 시각화 """
        original_data = pd.read_csv(self.input_file)
        filtered_data = pd.read_csv(self.output_file)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(original_data['x'], original_data['y'], original_data['z'], label='Original Data', color='blue', linestyle='--')
        ax.plot(filtered_data['x'], filtered_data['y'], filtered_data['z'], label='Filtered Data', color='red')

        ax.set_title('3D Comparison of Original and Kalman Filtered Data')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_zlabel('Z coordinate')
        ax.legend()
        plt.show()

    def run(self):
        """ 전체 프로세스 실행 """
        processed_data = self.process_and_save_data()
        if processed_data is not None:
            print(processed_data.head())
            self.plot_data_3d()


if __name__ == "__main__":
    processor = KalmanDataProcessor()
    processor.run()
