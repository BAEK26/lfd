import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# 1. 데이터 로드 (상대경로 사용)
data = pd.read_csv('./data/relative_sampled_neo_show_scenario.csv')  # 상대경로로 파일 로드

# 2. 중복된 timestamp 처리 (평균값 사용)
data_grouped = data.groupby('timestamp').mean().reset_index()

# 3. 데이터 추출
timestamps_clean = data_grouped['timestamp'].to_numpy()
positions_clean = data_grouped[['x', 'y', 'z', 'roll', 'pitch', 'yaw']].to_numpy()
joints_clean = data_grouped[['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']].to_numpy()

# 4. 세밀한 시간 스케일 정의 (예: 100ms 간격)
fine_timestamps_clean = np.arange(timestamps_clean.min(), timestamps_clean.max(), 100)

# 5. 보간 (Cubic Spline 사용)
interpolated_positions_clean = interp1d(timestamps_clean, positions_clean, axis=0, kind='cubic')(fine_timestamps_clean)
interpolated_joints_clean = interp1d(timestamps_clean, joints_clean, axis=0, kind='cubic')(fine_timestamps_clean)

# 6. 보간된 데이터 합치기
interpolated_data_clean = pd.DataFrame(
    np.hstack([fine_timestamps_clean.reshape(-1, 1), interpolated_positions_clean, interpolated_joints_clean]),
    columns=['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
)

# 7. 시각화
plt.figure(figsize=(12, 8))

# 위치 시각화 (x, y, z)
plt.subplot(2, 1, 1)
plt.plot(fine_timestamps_clean, interpolated_positions_clean[:, 0], label='x')
plt.plot(fine_timestamps_clean, interpolated_positions_clean[:, 1], label='y')
plt.plot(fine_timestamps_clean, interpolated_positions_clean[:, 2], label='z')
plt.title('Position Trajectory')
plt.xlabel('Time (ms)')
plt.ylabel('Position (mm)')
plt.legend()

# 관절 각도 시각화 (joint1 ~ joint6)
plt.subplot(2, 1, 2)
for i in range(6):
    plt.plot(fine_timestamps_clean, interpolated_joints_clean[:, i], label=f'joint{i+1}')
plt.title('Joint Angles Trajectory')
plt.xlabel('Time (ms)')
plt.ylabel('Angle (degrees)')
plt.legend()

plt.tight_layout()
plt.show()

# 8. 결과 저장
interpolated_data_clean.to_csv('interpolated_trajectory_clean.csv', index=False)
print("Interpolated trajectory saved as 'interpolated_trajectory_clean.csv'")
