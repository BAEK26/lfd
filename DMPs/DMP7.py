import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter

#7버전은 조인트, xyz 모두 보는 버전전

############################
# Hyperparameters & Settings
############################
alpha = 4.0  # canonical system decay parameter
hidden_dim = 128
learning_rate = 0.0018
epochs = 2000
model_save_path = 'dmp_joints_xyz_model.pt'
tensorboard_log_dir = './runs/dmp_joints_xyz_experiment'

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_file_path = os.path.join(base_dir, 'data', 'pumping_interpolated_trajectory.csv')

############################
# Load Data
############################
data = pd.read_csv(data_file_path)

# Joint angles: joint1~joint6
joint_cols = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
joints_demo = data[joint_cols].values  # (T,6)

# xyz data
xyz_demo = data[['x','y','z']].values  # (T,3)

timesteps = len(data)

# 시간 벡터 t 생성 (xyz 기반)
x_demo = xyz_demo[:,0]
y_demo = xyz_demo[:,1]
z_demo = xyz_demo[:,2]

t = np.cumsum(np.sqrt(np.diff(x_demo)**2 + np.diff(y_demo)**2 + np.diff(z_demo)**2))
t = np.insert(t, 0, 0)
t = t / t[-1]

# Canonical system: s(t) = exp(-alpha * t)
s = np.exp(-alpha * t)

# 스케일링
joint_scaler = MinMaxScaler()
joints_demo_scaled = joint_scaler.fit_transform(joints_demo)  # (T,6)

xyz_scaler = MinMaxScaler()
xyz_demo_scaled = xyz_scaler.fit_transform(xyz_demo)  # (T,3)

# Start / Target (joint, xyz)
start_joints = joints_demo_scaled[0, :]    # 6차원
target_joints = joints_demo_scaled[-1, :]  # 6차원

start_xyz = xyz_demo_scaled[0, :]    # 3차원
target_xyz = xyz_demo_scaled[-1, :]  # 3차원

############################
# Model Definition
############################
# 입력: t(1), s(1), start_joints(6), start_xyz(3), target_joints(6), target_xyz(3)
# 총 입력 차원: 1+1+6+3+6+3 = 20
# 출력: joint(6) + xyz(3) = 9
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
# Prepare Training Data
############################
train_input = []
for i in range(timesteps):
    inp = [t[i], s[i]]
    inp.extend(start_joints.tolist())   # 6
    inp.extend(start_xyz.tolist())      # 3
    inp.extend(target_joints.tolist())  # 6
    inp.extend(target_xyz.tolist())     # 3
    train_input.append(inp)

train_input = torch.tensor(train_input, dtype=torch.float32)

# 출력: joints_demo_scaled (T,6), xyz_demo_scaled (T,3) => 합쳐서 (T,9)
train_output = np.concatenate([joints_demo_scaled, xyz_demo_scaled], axis=1)
train_output = torch.tensor(train_output, dtype=torch.float32)

model = DMPNet(input_dim=20, output_dim=9, hidden_dim=hidden_dim)
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

    mse = torch.mean((predictions - train_output)**2).item()
    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_scalar('MSE/train', mse, epoch)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, MSE: {mse}")

writer.close()

############################
# Inference on Trained Target
############################
with torch.no_grad():
    generated_scaled = model(train_input).numpy()

# 분리
gen_joints_scaled = generated_scaled[:, :6]
gen_xyz_scaled = generated_scaled[:, 6:]

gen_joints = joint_scaler.inverse_transform(gen_joints_scaled)
gen_xyz = xyz_scaler.inverse_transform(gen_xyz_scaled)

# 결과 저장
output_df = pd.DataFrame(gen_joints, columns=joint_cols)
output_df['x'] = gen_xyz[:,0]
output_df['y'] = gen_xyz[:,1]
output_df['z'] = gen_xyz[:,2]
output_df['timestamp'] = data['timestamp'].values
output_df.to_csv('DMPs_joints_xyz_exp_file.csv', index=False)

# 모델 저장
torch.save(model.state_dict(), model_save_path)

############################
# Zero-Shot to a New Target
############################
# 새로운 target joint 및 xyz 설정
# 여기서는 demonstration 범위 내에서 임의로 새로운 목표 설정
# 실제 제로샷 성능은 다양한 데이터로 학습해야 향상됨
new_target_joints_raw = np.array([30, 45, 60, 20, 10, 5])  # 가상의 각도 값
new_target_xyz_raw = np.array([0.7, 0.3, 0.5])  # 가상의 xyz 위치 (원래 데이터 범위 고려 필요)

# 스케일링 (joint, xyz)
# joint_scaler, xyz_scaler는 데모 데이터 기준
new_target_joints = joint_scaler.transform(new_target_joints_raw.reshape(1,-1))[0]
new_target_xyz = xyz_scaler.transform(new_target_xyz_raw.reshape(1,-1))[0]

new_input = []
for i in range(timesteps):
    inp = [t[i], s[i]]
    inp.extend(start_joints.tolist())   # start joints 유지
    inp.extend(start_xyz.tolist())      # start xyz 유지
    inp.extend(new_target_joints.tolist())
    inp.extend(new_target_xyz.tolist())
    new_input.append(inp)

new_input = torch.tensor(new_input, dtype=torch.float32)

with torch.no_grad():
    new_generated_scaled = model(new_input).numpy()

new_gen_joints_scaled = new_generated_scaled[:, :6]
new_gen_xyz_scaled = new_generated_scaled[:, 6:]

new_gen_joints = joint_scaler.inverse_transform(new_gen_joints_scaled)
new_gen_xyz = xyz_scaler.inverse_transform(new_gen_xyz_scaled)

# 시각화 (joint)
plt.figure(figsize=(10,6))
for i in range(6):
    plt.plot(new_gen_joints[:,i], label=f'Joint{i+1}')
plt.title("Zero-Shot Generated Joint Angles for New Target")
plt.xlabel("Time step")
plt.ylabel("Joint Angle")
plt.legend()
plt.grid(True)
plt.show()

# 시각화 (xyz)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(gen_xyz[:,0], gen_xyz[:,1], gen_xyz[:,2], 'r-', label='Original Generated XYZ')
ax.plot(new_gen_xyz[:,0], new_gen_xyz[:,1], new_gen_xyz[:,2], '--', label='New Target Generated XYZ')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title("Zero-Shot Generated XYZ for New Target")
plt.show()

############################
# 주석:
# 이 예제에서는 조인트 각도와 xyz를 모두 예측하도록 설정.
# start/target에 joint, xyz 모두 포함하므로, 다양한 목표를 제시했을 때 zero-shot 시도 가능.
# 실제로는 다양한 start/target pair로 학습해야 진정한 일반화 성능이 나올 것.
############################
