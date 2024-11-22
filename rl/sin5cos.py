import numpy as np
import matplotlib.pyplot as plt

# DMP 매개변수 설정
alpha = 20.0  # 목표 위치로 수렴할 때의 감쇠 계수
beta = alpha / 4  # 임계 감쇠를 위한 계수
num_basis = 50  # 기저 함수 개수
h = np.ones(num_basis) * 25  # 기저 함수 너비 (수정)

# 예제 2D 시연 궤적 생성 (물결 궤적 - 근데 왜 동그라미..?)
timesteps = 200  # 시간 스텝 수
t = np.linspace(0, 2 * np.pi, timesteps)
x_demo = np.sin(5*t)  # x 좌표 (5주기)
y_demo = np.cos(5*t)  # y 좌표 (5주기)

# 가우시안 기저 함수의 중심 설정
c = np.linspace(0, 1, num_basis)

# 가중치 학습 함수 정의 (각 축에 대해 별도로 학습)
def learn_weights(demo, x_demo, c, h):
    target = demo - demo[0]  # 상대적 위치 계산
    phi = np.exp(-h * (x_demo[:, None] - c)**2)  # 기저 함수 계산
    weights = np.linalg.lstsq(phi, target, rcond=None)[0]  # 가중치 계산
    return weights

# 강제항 함수 정의 (scaling factor 5.0으로 늘려봄/ 원래 2.0)
def forcing_term(x, weights, c, h, scaling_factor=115.0):
    return scaling_factor * np.dot(weights, np.exp(-h * (x - c)**2))

# DMP 궤적 생성 함수
def generate_dmp(timesteps, demo, x_demo, alpha, beta, c, h, weights):
    dmp = np.zeros(timesteps)  # 위치
    dot = np.zeros(timesteps)  # 속도
    ddot = np.zeros(timesteps)  # 가속도
    y = demo[0]  # 시작 위치
    dy = 0.0  # 초기 속도
    target = demo[-1]  # 목표 위치

    # DMP 궤적 생성 루프
    for i in range(1, timesteps):
        f = forcing_term(x_demo[i], weights, c, h)  # 강제항 계산
        ddot[i] = alpha * (beta * (target - y) - dy) + f  # 가속도 계산
        dy += ddot[i] * (x_demo[1] - x_demo[0])  # 속도 업데이트
        y += dy * (x_demo[1] - x_demo[0])  # 위치 업데이트
        dmp[i] = y  # 생성된 위치 저장
    return dmp

# 각 축에 대해 가중치 학습
weights_x = learn_weights(x_demo, t, c, h)
weights_y = learn_weights(y_demo, t, c, h)

# 각 축에 대해 DMP 궤적 생성
x_dmp = generate_dmp(timesteps, x_demo, t, alpha, beta, c, h, weights_x)
y_dmp = generate_dmp(timesteps, y_demo, t, alpha, beta, c, h, weights_y)

# 결과 시각화
plt.figure(figsize=(8, 6))
plt.plot(x_demo, y_demo, label='Demonstration (Wave)')  # 시연 궤적
plt.plot(x_dmp, y_dmp, '--', label='DMP Generated')  # DMP로 생성한 궤적
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.title("2D DMP-based Trajectory Learning")
plt.axis('equal')  # 비율을 동일하게 설정하여 원형을 유지
plt.show()

#RL
# 원형궤적 스케일링 함수랑 / 기저함수 50개로 만들어봄
# actor critic // dmp > 함수... 
# 리얼월드 궤적 모방 
# 가져와서 쏟기 // 경로 // 실린더 1층 잡았다! 임의로 // 가져와서 틸트
# 퓨처 : 시퀀스 하나 더 / 