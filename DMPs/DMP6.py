import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

#6버전은 조인트만 타겟으로

############################
# Hyperparameters & Settings
############################
alpha = 4.0  # phase decay parameter for canonical system
hidden_dim = 128
learning_rate = 0.0018
epochs = 2000
model_save_path = 'dmp_model.pt'
tensorboard_log_dir = './runs/dmp_experiment'

# 파일 경로
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_file_path = os.path.join(base_dir, 'data', 'pumping_interpolated_trajectory.csv')

############################
# Data Loading & Preprocessing
############################
data = pd.read_csv(data_file_path)
x_demo = data['x'].values
y_demo = data['y'].values
z_demo = data['z'].values

scaler = MinMaxScaler()
xyz_demo = scaler.fit_transform(np.vstack((x_demo, y_demo, z_demo)).T)
x_demo, y_demo, z_demo = xyz_demo[:, 0], xyz_demo[:, 1], xyz_demo[:, 2]

start_pos = np.array([x_demo[0], y_demo[0], z_demo[0]])
target_pos = np.array([x_demo[-1], y_demo[-1], z_demo[-1]])

timesteps = len(x_demo)
t = np.cumsum(np.sqrt(np.diff(x_demo)**2 + np.diff(y_demo)**2 + np.diff(z_demo)**2))
t = np.insert(t, 0, 0)
t = t / t[-1]

############################
# Canonical System (Phase)
############################
# DMP canonical system: s(t) = exp(-alpha * t)
# t는 [0,1]로 정규화 되어있으므로 t에 대해 exponent를 취함.
s = np.exp(-alpha * t)

############################
# Model Definition
############################
class DMPNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DMPNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softsign(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


############################
# Training Setup
############################
# 입력 특징: t, s(t), start_x, start_y, start_z, target_x, target_y, target_z
# 출력: x, y, z trajectory
train_input = []
for i in range(timesteps):
    # 하나의 데모 밖에 없으므로 start/target은 동일
    train_input.append([t[i], s[i], start_pos[0], start_pos[1], start_pos[2],
                        target_pos[0], target_pos[1], target_pos[2]])

train_input = torch.tensor(train_input, dtype=torch.float32)
train_output = torch.tensor(np.vstack((x_demo, y_demo, z_demo)).T, dtype=torch.float32)

model = DMPNet(input_dim=8, output_dim=3, hidden_dim=hidden_dim)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

writer = SummaryWriter(log_dir=tensorboard_log_dir)

############################
# Training Loop
############################
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    predictions = model(train_input)
    loss = criterion(predictions, train_output)
    loss.backward()
    optimizer.step()

    # Log metrics
    writer.add_scalar('Loss/train', loss.item(), epoch)
    mse = torch.mean((predictions - train_output)**2).item()
    writer.add_scalar('MSE/train', mse, epoch)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, MSE: {mse}")

writer.close()

############################
# Inference (Generate Trajectory)
############################
with torch.no_grad():
    # 기본적으로는 학습 시의 target pos을 그대로 사용
    generated_trajectory = model(train_input).numpy()

# 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_demo, y_demo, z_demo, 'r-', label='Demonstration Trajectory')
ax.plot(generated_trajectory[:, 0], generated_trajectory[:, 1], generated_trajectory[:, 2], '--', label='Generated Trajectory')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.legend()
plt.title("DMP-based Trajectory with Deep Learning (Phase + Target Generalization)")
plt.show()

# 역정규화
xyz_new = scaler.inverse_transform(generated_trajectory)

def find_closest_row(x, y, z, data_df):
    distances = np.sqrt((data_df['x'] - x)**2 + (data_df['y'] - y)**2 + (data_df['z'] - z)**2)
    closest_idx = distances.idxmin()
    return data.loc[closest_idx, ['roll', 'pitch', 'yaw']]

df = pd.DataFrame({
    'timestamp' : data.timestamp,
    'x' : xyz_new[:, 0],
    'y' : xyz_new[:, 1],
    'z' : xyz_new[:, 2]
})

df[['roll', 'pitch', 'yaw']] = df.apply(lambda row: find_closest_row(row['x'], row['y'], row['z'], data), axis=1)
output = pd.concat([df, data[['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']]], axis=1)
output.to_csv('DMPs_exp_file.csv', index=False)

# 모델 저장
torch.save(model.state_dict(), model_save_path)

############################
# Zero-Shot to a new target
############################
# 향후 새로운 목표지점으로 Trajectory를 생성할때
# 예: 새로운 목표점 new_target
new_target = np.array([0.5, 0.8, 0.2])  # 임의의 new target in [0,1] normalized space

# start는 동일하다고 가정 (추후 start도 변경 가능)
new_input = []
for i in range(timesteps):
    new_input.append([t[i], s[i], start_pos[0], start_pos[1], start_pos[2],
                      new_target[0], new_target[1], new_target[2]])
new_input = torch.tensor(new_input, dtype=torch.float32)

with torch.no_grad():
    new_generated_trajectory = model(new_input).numpy()

# Inverse transform
new_xyz = scaler.inverse_transform(new_generated_trajectory)

# 원하는 경우 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xyz_new[:,0], xyz_new[:,1], xyz_new[:,2], 'r-', label='Original Generated Trajectory')
ax.plot(new_xyz[:,0], new_xyz[:,1], new_xyz[:,2], '--', label='New Target Generated Trajectory')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.legend()
plt.title("Zero-Shot Generalization to New Target")
plt.show()

############################
# 주석:
# 현재는 하나의 demonstration으로 학습했기 때문에 실제로 "제로샷"으로 잘 
# 일반화할지는 미지수입니다. 하지만 코드 구조상 start/target을 입력으로 받고
# phase 변수를 도입했으므로, 여러 다양한 start/target과 trajectory로 학습한다면
# 새로운 목표점에 대한 trajectory 생성이 더 현실화될 수 있습니다.
############################
