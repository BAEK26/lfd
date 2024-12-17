import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

# 파일 경로 설정
file_path = './data/pumping_interpolated_trajectory.csv'

# CSV 파일 읽기
data = pd.read_csv(file_path)

# 데이터 정제: 필요한 x, y, z 데이터 선택
x_demo = data['x'].values
y_demo = data['y'].values
z_demo = data['z'].values

# 데이터 정규화
scaler = MinMaxScaler()
xyz_demo = scaler.fit_transform(np.vstack((x_demo, y_demo, z_demo)).T)

x_demo, y_demo, z_demo = xyz_demo[:, 0], xyz_demo[:, 1], xyz_demo[:, 2]

# 데이터 스케일 및 범위 확인
print("X min:", np.min(x_demo), "X max:", np.max(x_demo))
print("Y min:", np.min(y_demo), "Y max:", np.max(y_demo))
print("Z min:", np.min(z_demo), "Z max:", np.max(z_demo))

# 시작점과 목표점 설정
start_pos = np.array([x_demo[0], y_demo[0], z_demo[0]])
target = np.array([x_demo[-1], y_demo[-1], z_demo[-1]])

# 시간 벡터 생성
timesteps = len(x_demo)
t = np.cumsum(np.sqrt(np.diff(x_demo)**2 + np.diff(y_demo)**2 + np.diff(z_demo)**2))
t = np.insert(t, 0, 0)
t = t / t[-1]  # [0, 1]로 정규화

# 강제항 학습을 위한 딥러닝 기반 모델
import torch
import torch.nn as nn
import torch.optim as optim

class DMPNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DMPNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# 모델 초기화
model = DMPNet(input_dim=1, output_dim=3)  # 입력: 타임스텝, 출력: x, y, z
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 데이터 준비
train_input = torch.tensor(t[:, None], dtype=torch.float32)
train_output = torch.tensor(np.vstack((x_demo, y_demo, z_demo)).T, dtype=torch.float32)

# 모델 학습
epochs = 1000
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    predictions = model(train_input)
    loss = criterion(predictions, train_output)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

# DMP 생성
with torch.no_grad():
    generated_trajectory = model(train_input).numpy()

# 결과 시각화
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
