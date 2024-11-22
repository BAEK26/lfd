import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# 파일 경로 설정
file_path = 'D:\\Jeong-eun\\LfD_DMPs\\test.csv'

# CSV 파일 읽기
data = pd.read_csv(file_path)

# 데이터 정제: 필요한 x, y, z 데이터 선택
x_demo = data['x'].values
y_demo = data['y'].values
z_demo = data['z'].values

# 데이터 스케일 및 범위 확인
print("X min:", np.min(x_demo), "X max:", np.max(x_demo))
print("Y min:", np.min(y_demo), "Y max:", np.max(y_demo))
print("Z min:", np.min(z_demo), "Z max:", np.max(z_demo))

# 시간 벡터 생성
timesteps = len(x_demo)
t = np.linspace(0, 1, timesteps)

# 매개변수 설정
alpha = 25.0
beta = alpha / 4
num_basis = 1000
#h = np.ones(num_basis) * num_basis # 기저 함수 너비 조절
h = np.ones(num_basis) * 20

# 기저 함수 중심
c = np.linspace(0, 1, num_basis)

# 가중치 학습 함수
def learn_weights(demo, t, c, h):
    phi = np.exp(-h * (t[:, None] - c)**2)
    weights = np.linalg.lstsq(phi, demo, rcond=None)[0]
    return weights

# 각 축에 대한 가중치 학습
weights_x = learn_weights(x_demo,t, c, h)
weights_y = learn_weights(y_demo,t, c, h)
weights_z = learn_weights(z_demo,t, c, h)

weights = np.column_stack((weights_x, weights_y, weights_z))

# 강제항 함수 정의
def forcing_term(time, weights, c, h, scaling_factor=12000.0):
    return scaling_factor * np.dot(weights, np.exp(-h * (time - c)**2))


# DMP 궤적 생성 함수
def generate_dmp(timesteps, start_pos, target, alpha, beta, c, h, weights_x, weights_y, weights_z):
    trajectory = np.zeros((timesteps, 3))
    velocity = np.zeros((timesteps, 3))
    trajectory[0] = start_pos
    for i in range(1, timesteps):
        time = i / float(timesteps)
        x = trajectory[i-1]
        force_x = forcing_term(time, weights_x, c, h)
        force_y = forcing_term(time, weights_y, c, h)
        force_z = forcing_term(time, weights_z, c, h)
        velocity[i] = alpha * (beta * (target - x) - velocity[i-1]) + np.array([force_x, force_y, force_z])
        trajectory[i] = trajectory[i-1] + velocity[i] * (1.0 / timesteps)
        if i > timesteps // 2:
            target = start_pos
    return trajectory

# 시작점과 목표점 설정
start_pos = np.array([x_demo[0], y_demo[0], z_demo[0]])
target = np.array([x_demo[-1], y_demo[-1], z_demo[-1]])

# DMP 궤적 생성
trajectory = generate_dmp(timesteps, start_pos, target, alpha, beta, c, h, weights_x, weights_y, weights_z)

# 결과 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_demo, y_demo, z_demo, 'r-', label='3D Demonstration')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], '--', label='DMP Generated')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')

# 축 범위 조정
ax.set_xlim([np.min(x_demo), np.max(x_demo)])
ax.set_ylim([np.min(y_demo), np.max(y_demo)])
ax.set_zlim([np.min(z_demo), np.max(z_demo)])

ax.legend()
ax.set_title("3D DMP-based Trajectory Learning")
plt.show()
