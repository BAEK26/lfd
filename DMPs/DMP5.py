import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # TensorBoard writer

# 5 버전은 xyz만 타겟으로

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_file_path = os.path.join(base_dir, 'data', 'pumping_interpolated_trajectory.csv')

data = pd.read_csv(data_file_path)
x_demo = data['x'].values
y_demo = data['y'].values
z_demo = data['z'].values

scaler = MinMaxScaler()
xyz_demo = scaler.fit_transform(np.vstack((x_demo, y_demo, z_demo)).T)

x_demo, y_demo, z_demo = xyz_demo[:, 0], xyz_demo[:, 1], xyz_demo[:, 2]

# 시작점과 목표점 설정
start_pos = np.array([x_demo[0], y_demo[0], z_demo[0]])
target = np.array([x_demo[-1], y_demo[-1], z_demo[-1]])

# 시간 벡터 t 생성
timesteps = len(x_demo)
t = np.cumsum(np.sqrt(np.diff(x_demo)**2 + np.diff(y_demo)**2 + np.diff(z_demo)**2))
t = np.insert(t, 0, 0)
t = t / t[-1]

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

# 모델 초기화
# 여기서는 입력 t 만을 사용하지만, 추후 다른 목표점에 대한 생성을 위해 
# start_pos, target을 함께 input으로 넣는 것도 가능.
model = DMPNet(input_dim=1, output_dim=3)
criterion = nn.SmoothL1Loss()  
optimizer = optim.Adam(model.parameters(), lr=0.0018)

train_input = torch.tensor(t[:, None], dtype=torch.float32)
train_output = torch.tensor(np.vstack((x_demo, y_demo, z_demo)).T, dtype=torch.float32)

# TensorBoard 설정
writer = SummaryWriter(log_dir='./runs/dmp_experiment')

epochs = 2000  # Epoch 늘리기
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    predictions = model(train_input)
    loss = criterion(predictions, train_output)
    loss.backward()
    optimizer.step()

    # TensorBoard에 Loss 기록
    writer.add_scalar('Loss/train', loss.item(), epoch)

    # 회귀 문제에서 accuracy 대신 MSE를 추가적으로 기록해볼 수 있음
    mse = torch.mean((predictions - train_output)**2).item()
    writer.add_scalar('MSE/train', mse, epoch)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, MSE: {mse}")

writer.close()

# DMP Trajectory 생성
with torch.no_grad():
    generated_trajectory = model(train_input).numpy()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_demo, y_demo, z_demo, 'r-', label='Demonstration Trajectory')
ax.plot(generated_trajectory[:, 0], generated_trajectory[:, 1], generated_trajectory[:, 2], '--', label='Generated Trajectory')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.legend()
plt.title("DMP-based Trajectory with Deep Learning")
plt.show()

xyz_new = np.vstack((generated_trajectory[:, 0], generated_trajectory[:, 1], generated_trajectory[:, 2])).T
xyz_new = scaler.inverse_transform(xyz_new)

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

# roll, pitch, yaw를 찾아 DF에 추가
df[['roll', 'pitch', 'yaw']] = df.apply(lambda row: find_closest_row(row['x'], row['y'], row['z'], data), axis=1)
output = pd.concat([df, data[['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']]], axis=1)
output.to_csv('DMPs_exp_file.csv', index=False)

# 학습된 모델 저장
# 추후 다른 목표점으로 trajectory를 그릴 때 이 모델을 로드하여 
# 새로운 목표점 조건하에 inference를 진행할 수 있음.
torch.save(model.state_dict(), 'dmp_model.pt')

####################################################
# 추가 아이디어:
#  - 다른 목표지점을 사용한 trajectory 생성:
#    t는 동일하지만 target을 다르게 설정한 뒤,
#    모델 입력에 목표위치를 추가(예: t, x_start, y_start, z_start, x_target, y_target, z_target)를 넣는 형태로 모델 변경.
#    학습 시에도 해당 정보를 주어준다면, 추론 시 target을 바꾸어 trajectory 생성 가능.
#
#  - canonical system output:
#    DMP에서는 phase 변수 s(t) = exp(-alpha * t) 등으로 정의되는 canonical system을 사용.
#    여기서는 단순히 정규화 시간 t를 사용했지만, phase 변수를 별도로 정의해서
#    model input에 포함하면, DMP의 특징을 더 살릴 수 있음.
####################################################
