import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# 1. 데이터 로드 (상대경로 사용)
data = pd.read_csv('./data/test.csv')  # 상대경로로 파일 로드

# 2. 중복된 timestamp 처리 (평균값 사용)
data_grouped = data.groupby('timestamp').mean().reset_index()

# 3. 데이터 추출
timestamps_clean = data_grouped['timestamp'].to_numpy()
positions_clean = data_grouped[['x', 'y', 'z', 'roll', 'pitch', 'yaw']].to_numpy()
joints_clean = data_grouped[['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']].to_numpy()

# 4. 세밀한 시간 스케일 정의 (예: 100ms 간격)
fine_timestamps_clean = np.arange(timestamps_clean.min(), timestamps_clean.max(), 200)

# 5. 보간 (Cubic Spline 사용)
interpolated_positions_clean = interp1d(timestamps_clean, positions_clean, axis=0, kind='cubic')(fine_timestamps_clean)
interpolated_joints_clean = interp1d(timestamps_clean, joints_clean, axis=0, kind='cubic')(fine_timestamps_clean)

# 6. 보간된 데이터 합치기
interpolated_data_clean = pd.DataFrame(
    np.hstack([fine_timestamps_clean.reshape(-1, 1), interpolated_positions_clean, interpolated_joints_clean]),
    columns=['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
)

# 6. 3D 시각화
fig = plt.figure(figsize=(10, 8))

# 3D 궤적 플롯
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions_clean[:, 0], positions_clean[:, 1], positions_clean[:, 2], 'r-', label='Original Trajectory')
ax.plot(interpolated_positions_clean[:, 0], interpolated_positions_clean[:, 1], interpolated_positions_clean[:, 2],
        '--', label='Interpolated Trajectory')
ax.set_xlabel('X Position (mm)')
ax.set_ylabel('Y Position (mm)')
ax.set_zlabel('Z Position (mm)')
ax.legend()
plt.title("Interpolated 3D Trajectory")
plt.show()

"""
# 7. 3D + 시간 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(fine_timestamps_clean, interpolated_positions_clean[:, 0], interpolated_positions_clean[:, 1], label='X vs Time')
ax.plot(fine_timestamps_clean, interpolated_positions_clean[:, 1], interpolated_positions_clean[:, 2], label='Y vs Time')
ax.plot(fine_timestamps_clean, interpolated_positions_clean[:, 2], interpolated_positions_clean[:, 0], label='Z vs Time')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Position 1 (mm)')
ax.set_zlabel('Position 2 (mm)')
ax.legend()
plt.title("Time-based Interpolated Trajectory")
plt.show()

"""
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
interpolated_data_clean.to_csv('test_trajectory.csv', index=False)
print("Interpolated trajectory saved as 'interpolated_trajectory_clean.csv'")

# .traj 파일 저장
def save_as_traj_file(filename, timestamps, positions, joints):
    with open(filename, 'w') as f:
        f.write('# Timestamp, X, Y, Z, Roll, Pitch, Yaw, Joint1, Joint2, Joint3, Joint4, Joint5, Joint6\n')
        for t, pos, joint in zip(timestamps, positions, joints):
            pos_str = ', '.join(f'{p:.6f}' for p in pos)  # 포지션 데이터 포맷
            joint_str = ', '.join(f'{j:.6f}' for j in joint)  # 조인트 데이터 포맷
            f.write(f'{t}, {pos_str}, {joint_str}\n')

# 파일 저장
traj_filename = './data/interpolated_trajectory.traj'
save_as_traj_file(
    traj_filename, 
    fine_timestamps_clean, 
    interpolated_positions_clean, 
    interpolated_joints_clean
)

print(f"Trajectory data saved as '{traj_filename}'")

