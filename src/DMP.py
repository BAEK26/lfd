#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DMP.py

입력: 'sumin_test_a.csv'
  - 컬럼: timestamp, x, y, z, roll, pitch, yaw, joint1, joint2, joint3, joint4, joint5, joint6, gripper
DMP 처리: xyz 좌표에 대해 뉴럴 네트워크를 통한 forcing term 예측 후 DMP 동역학 적분
출력: xyz가 DMP 처리된 새로운 경로를 포함한 CSV 파일과, 시뮬레이터를 통한 궤적 비교
"""

##################################
# 0. 라이브러리
##################################

import os
import numpy as np
import pandas as pd
import matplotlib
import torch.optim.adamw
import torch.optim.adamw
matplotlib.use('Qt5Agg')  # 또는 환경에 맞게 변경
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import datetime

# 현재 파일이 src 폴더 안에 있을 때, 상위 폴더를 sys.path에 추가
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
from src.trajectory import Trajectory


##################################
# Forcing Network 클래스
##################################
class ForcingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        input_dim: forcing term 예측에 사용할 입력 차원 (예: 시작 xyz와 목표 xyz → 3+3=6)
        output_dim: 예측할 가중치 차원 = num_basis * movement_dim (예: num_basis * 3)
        """
        super(ForcingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

##################################
# Neural DMP 클래스
##################################
class NeuralDMP:
    def __init__(self, num_basis, alpha_z, beta_z_ratio, alpha_x, tau, width, forcing_input_dim=6, movement_dim=3):
        """
        num_basis: 사용될 기저 함수 개수
        alpha_z: 변환 시스템의 gain 계수
        beta_z_ratio: beta_z = alpha_z / beta_z_ratio (예: 일반적으로 beta_z = alpha_z/4)
        alpha_x: 캐노니컬 시스템 계수
        tau: 시간 상수
        width: basis function의 폭
        forcing_input_dim: forcing 네트워크 입력 차원 (보통 시작과 목표 xyz → 6)
        movement_dim: 움직임 차원 (XYZ → 3)
        """
        self.num_basis = num_basis
        self.alpha_z = alpha_z
        self.beta_z = alpha_z / beta_z_ratio
        self.alpha_x = alpha_x
        self.tau = tau
        self.width = width
        self.movement_dim = movement_dim
        
        # Forcing term 예측을 위한 뉴럴 네트워크 초기화
        self.forcing_net = ForcingNetwork(forcing_input_dim, num_basis * movement_dim)
    
    def __canonical_system(self, timesteps):
        """
        캐노니컬 시스템 계산:
          - t_demo: 0부터 1까지의 선형 구간 (timesteps 길이)
          - s: canonical variable, s = exp(-alpha_x * t_demo / tau)
        """
        t_demo = np.linspace(0, 1, timesteps)
        s = np.exp(-self.alpha_x * t_demo / self.tau)
        return t_demo, s
    
    def __compute_basis_functions(self, s):
        """
        Gaussian basis functions 계산:
          - s: canonical variable (timesteps,)
          - centers: [0, 1] 구간에 균등하게 num_basis 개 배치
          - widths: 모든 basis에 대해 동일한 폭
        반환: (timesteps, num_basis) 크기의 basis 행렬
        """
        centers = np.linspace(0, 1, self.num_basis)
        widths = np.ones(self.num_basis) * self.width
        basis = np.exp(-0.5 * ((s[:, None] - centers) ** 2) / widths)
        return basis

    def generate_tensor(self, timesteps, start_xyz, goal_xyz):
        """
        differentiable simulation (PyTorch) – training 용.
        입력:
          timesteps: 궤적 길이
          start_xyz: 시작 위치 (3,)
          goal_xyz: 목표 위치 (3,)
        반환:
          sim_xyz: (timesteps, 3) 텐서, 시뮬레이션으로 생성한 xyz 궤적
        """
        # 캐노니컬 시스템 (torch tensor)
        t_demo = torch.linspace(0, 1, timesteps)
        s = torch.exp(-self.alpha_x * t_demo / self.tau)  # (timesteps,)
        
        # basis functions 계산 (torch)
        centers = torch.linspace(0, 1, self.num_basis)
        widths = torch.ones(self.num_basis) * self.width
        # s.unsqueeze(1): (timesteps, 1), centers.unsqueeze(0): (1, num_basis)
        basis = torch.exp(-0.5 * ((s.unsqueeze(1) - centers.unsqueeze(0))**2) / widths.unsqueeze(0))  # (timesteps, num_basis)
        
        # forcing 네트워크 입력: 시작과 목표 xyz 연결 (6,)
        forcing_input = torch.tensor(np.concatenate([start_xyz, goal_xyz]), dtype=torch.float32).unsqueeze(0)  # (1,6)
        weights_flat = self.forcing_net(forcing_input)  # (1, num_basis * movement_dim)
        weights = weights_flat.view(self.num_basis, self.movement_dim)  # (num_basis, 3)
        
        dt = 1.0 / (timesteps - 1)
        sim_xyz = torch.zeros((timesteps, self.movement_dim), dtype=torch.float32)
        v = torch.zeros((timesteps, self.movement_dim), dtype=torch.float32)
        sim_xyz[0] = torch.tensor(start_xyz, dtype=torch.float32)
        goal_tensor = torch.tensor(goal_xyz, dtype=torch.float32)
        
        for t in range(1, timesteps):
            # f(t) = (basis[t] dot weights) * s[t]
            f_t = (torch.matmul(basis[t:t+1], weights)).squeeze(0) * s[t]
            # Euler integration:
            #   tau * v_dot = alpha_z*(beta_z*(goal - x) - v) + f(t)
            #   tau * x_dot = v
            a = (self.alpha_z * (self.beta_z * (goal_tensor - sim_xyz[t-1]) - v[t-1]) + f_t) / self.tau
            v[t] = v[t-1] + a * dt
            sim_xyz[t] = sim_xyz[t-1] + (v[t] / self.tau) * dt
        return sim_xyz  # (timesteps, 3)
    
    def train(self, demo_trajectory, log_dir, num_epochs=2000, lr=0.001):
        optimizer = torch.optim.AdamW(self.forcing_net.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        demo_xyz_tensor = torch.tensor(demo_trajectory.xyz, dtype=torch.float32)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=os.path.join(log_dir, str(current_time)))

        timesteps = demo_trajectory.len()
        start_xyz = demo_trajectory.xyz[0]
        goal_xyz = demo_trajectory.xyz[-1]
        demo_xyz_tensor = torch.tensor(demo_trajectory.xyz, dtype=torch.float32)
        
        for epoch in tqdm(range(1, num_epochs + 1), desc="Training epochs", unit="epoch"):
            optimizer.zero_grad()
            sim_xyz_tensor = self.generate_tensor(timesteps, start_xyz, goal_xyz)
            loss = criterion(sim_xyz_tensor, demo_xyz_tensor)
            loss.backward()
            optimizer.step()
            
            # 스칼라 로깅: Training Loss
            writer.add_scalar("Loss/Train", loss.item(), epoch)
            
            # 스칼라 로깅: Position RMSE (전체 xyz 차원)
            position_rmse = torch.sqrt(torch.mean((sim_xyz_tensor - demo_xyz_tensor)**2))
            writer.add_scalar("RMSE/Position", position_rmse.item(), epoch)
            
            # 스칼라 로깅: Final Goal Error (마지막 시간 스텝의 오차)
            final_goal_error = torch.abs(sim_xyz_tensor[-1] - demo_xyz_tensor[-1]).mean()
            writer.add_scalar("FinalGoalError", final_goal_error.item(), epoch)

            # 스칼라 및 히스토그램: Forcing term weights 통계 (학습 직후 네트워크 입력에 대한 예측)            with torch.no_grad():
            forcing_input_tensor = torch.tensor(np.concatenate([start_xyz, goal_xyz]), dtype=torch.float32).unsqueeze(0)
            forcing_weights = self.forcing_net(forcing_input_tensor).view(self.num_basis, self.movement_dim)
            writer.add_histogram("ForcingWeights", forcing_weights, epoch)
            writer.add_scalar("ForcingWeights_Mean", forcing_weights.mean().item(), epoch)
            writer.add_scalar("ForcingWeights_Var", forcing_weights.var().item(), epoch)

            # 히스토그램: 각 시간 스텝의 궤적 오차 (시뮬레이션과 데모 궤적 차이)
            traj_error = sim_xyz_tensor - demo_xyz_tensor
            writer.add_histogram("TrajectoryError", traj_error, epoch)
            
            if epoch % 1000 == 0:
                sim_current = demo_trajectory.copy()
                sim_current.xyz = sim_xyz_tensor.detach().numpy()
                demo_trajectory.show(sim_current)
                tqdm.write(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}")
                print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}")
        
        writer.close()
        print("Training complete.")

    def generate(self, trajectory, start=None, end=None):
        """
        numpy 버전 simulation – 학습 후 최종 출력용.
        입력 DataFrame의 xyz만 DMP 처리하며 나머지 컬럼은 그대로 유지.
        
        인자:
          trajectory: 입력 DataFrame (컬럼: timestamp, x, y, z, roll, pitch, yaw, joint1~joint6, gripper)
          start: 새로운 시작 xyz (array-like, (3,)); 제공하지 않으면 원본 첫 행 사용
          end: 새로운 목표 xyz (array-like, (3,)); 제공하지 않으면 원본 마지막 행 사용
        
        반환:
          output: DMP 처리된 결과 DataFrame
        """
        timesteps = trajectory.len()
        t_demo, s_np = self.__canonical_system(timesteps)
        basis = self.__compute_basis_functions(s_np)  # (timesteps, num_basis)
        
        orig_xyz = trajectory.xyz  # (timesteps, 3)

        start_xyz = start if start is not None else trajectory.xyz[0]
        goal_xyz = end if end is not None else trajectory.xyz[-1]

        # forcing 네트워크의 입력 준비 (numpy → tensor)
        forcing_input = np.concatenate([start_xyz, goal_xyz])
        forcing_input_tensor = torch.tensor(forcing_input, dtype=torch.float32).unsqueeze(0)
        weights_flat = self.forcing_net(forcing_input_tensor)
        weights = weights_flat.view(self.num_basis, self.movement_dim).detach().numpy()  # (num_basis, 3)
        
        # numpy 기반 forcing term 계산
        f = (basis @ weights) * s_np[:, None]  # (timesteps, 3)
        
        dt = 1.0 / (timesteps - 1)
        dmp_xyz = np.zeros_like(orig_xyz)
        v = np.zeros_like(orig_xyz)
        dmp_xyz[0] = start_xyz
        
        for t in range(1, timesteps):
            a = (self.alpha_z * (self.beta_z * (goal_xyz - dmp_xyz[t-1]) - v[t-1]) + f[t]) / self.tau
            v[t] = v[t-1] + a * dt
            dmp_xyz[t] = dmp_xyz[t-1] + (v[t] / self.tau) * dt
        
        trajectory_dmp = trajectory.copy()
        trajectory_dmp.xyz = dmp_xyz

        return trajectory_dmp



##################################
# Main Execution
##################################
if __name__ == '__main__':

    base_dir = r"C:\Users\박수민\Documents\neoDMP" # base 경로 (알맞게 수정)
    input_csv = os.path.join(base_dir, "data", "processed_sumin_a.csv") # CSV 로드 파일 경로

    # CSV 불러와 Trajectory 객체 생성
    traj = Trajectory.load_csv(input_csv)

    # 신경망 DMP 객체 생성
    dmp = NeuralDMP(num_basis=20, alpha_z=25, beta_z_ratio=4, alpha_x=4, tau=1, 
                    width=0.05, forcing_input_dim=6, movement_dim=3)
    
    # 필터링된 Trajectory에 DMP 적용
    log_dir = os.path.join(base_dir, 'runs', 'dmp_training') # 학습 로그 저장 경로 
    dmp.train(traj, log_dir, num_epochs=1000)

    # 학습된 가중치에 기반한 경로 생성 및 시각화
    traj_DMPed = dmp.generate(traj)
    Trajectory.show(traj, traj_DMPed)

    # 일반화 능력 보기
    from src.utils import random_near_endpoints

    # 예시는 5개 정도
    for i in range(5):  

      # end 위치를 30% 이내에서 적절히 변화 주기
      new_goal = random_near_endpoints(traj_DMPed, option='end', random_rate=0.3)
      traj_DMP_goal_changed = dmp.generate(traj_DMPed, end=new_goal)
      Trajectory.show(traj_DMPed, traj_DMP_goal_changed)
