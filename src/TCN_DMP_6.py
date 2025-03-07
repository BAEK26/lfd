#!/usr/bin/env python
# -*- coding: utf-8 -*-
#3차원 인풋, rpy 사용 안하고 수납만. 
# “xyz+오일러 각 6차원 포즈”는 pose_demo로 가지고 있고, 
# #DMP 쪽에서 orientation을 어떻게 처리할지는 후속 설계(“6차원 DMP로 갈지, xyz만 DMP로 할지”).

"""
TCN Performance Metrics: Loss, dimension-wise error, Forcing term 분포, 파라미터 히스토그램
DMP Performance Metrics: Final Goal Error, trajectory error(각 축 RMSE), etc.
"""

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# --------------------------------------------------------
# 상위 폴더 경로 추가 & Trajectory import
# --------------------------------------------------------
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # 또는 환경에 맞게 변경
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.animation as animation
try:
    from src.trajectory import Trajectory
except:
    from trajectory import Trajectory

#from src.dataset_6d import DMPDataset6D  # (예) 6자유도 Dataset (pose_demo: (T,6))
#from src.model_tcn_6d import DMP6D  # (예) 위에서 정의한 TCN + 6D DMP
#from src.trainer_6d import Trainer6D
# from src.utils import random_near_endpoints   # 필요하면 사용

#######################################################
# 1) Dataset 구성: a/b/c/d CSV를 시계열 샘플로 로드
#######################################################
class DMPDataset6D(Dataset):
    """
    '6차원 DMP'를 위한 데이터셋 클래스:
      - pose_demo: (T,6) = [x, y, z, roll, pitch, yaw]
      - multi_in:  (T,8) = [s(t), t_lin, (goal - start) 6D]
      - dt_array: (T-1,) or None
      - start, goal: (6,)
    """
    def __init__(self, csv_files, alpha_x=4.0, tau=1.0):
        """
        csv_files : ['processed_a.csv', 'processed_b.csv', ...]
        alpha_x, tau : DMP canonical system params
        """
        self.samples = []
        for csv_path in csv_files:
            traj = Trajectory.load_csv(csv_path)
            
            xyz = traj.xyz  # (T,3)
            rpy = traj.euler_angles  # (T,3)
            
            T = traj.len()

            # timestamps -> dt_array
            dt_array = np.diff(traj.timestamp) / 1000.0  # 초 단위 변환
            
            # 1) 6차원 pose_demo
            pose_demo = np.concatenate([xyz, rpy], axis=1)  # (T,6)

            # 2) start, goal: (6,)
            start_pose = pose_demo[0]   # (6,)
            goal_pose  = pose_demo[-1]  # (6,)

            # 3) canonical s(t), t_lin
            t_lin = np.linspace(0, 1, T)    # (T,)
            s = np.exp(-alpha_x * t_lin / tau)  # (T,)

            # 4) (goal - start) 6D
            gs_6d = goal_pose - start_pose  # (6,)
            s_col = s.reshape(T,1)          # (T,1)
            t_col = t_lin.reshape(T,1)      # (T,1)
            repeated_gs = np.tile(gs_6d, (T,1))  # (T,6)
            multi_in = np.concatenate([s_col, t_col, repeated_gs], axis=1)  # (T,8)

            # 5) 샘플 딕셔너리 생성
            sample = {
                'pose_demo': pose_demo,  # (T,6)
                'start'    : start_pose, # (6,)
                'goal'     : goal_pose,  # (6,)
                'multi_in' : multi_in,   # (T,8)
                'dt_array' : dt_array    # (T-1,) or None
            }
            self.samples.append(sample)

    def __len__(self):
        """ 전체 유효 CSV(샘플) 개수 반환 """
        return len(self.samples)

    def __getitem__(self, idx):
        """ idx번째 샘플(딕셔너리) 반환 """
        return self.samples[idx]

#######################################################
# 2) TCN 정의
#######################################################
class TCNBlock(nn.Module):
    """
    Dilated Causal Conv 1D 블록(Residual) WaveNet/TCN 스타일
    - 2번의 dilated conv 후, 한번에 slice를 해서
      입력과 동일한 길이를 맞춘다.
    # in_channels -> out_channels -> out_channels
    시간축에 대해 dilated convolution을 적용하므로, receptive field가 지수적으로 증가 
    → 장기 의존성(Long-range dependency)을 포착 가능.
    두 번의 convolution + residual → CNN 기반 시계열 처리의 일반적이고 안정적인 아키텍처.
    설계는  TCN이 “(s(t), t_lin, goal-start(6D))” = 8채널 입력 → “6차원” 출력을 매 시점 얻도록
    """
    def __init__(self, in_channels, out_channels, 
                 kernel_size=2, dilation=1, dropout=0.1, causal=True):
        super(TCNBlock, self).__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout

        #conv 1개당 쓸 padding
        #kernel_size = 2 -> (2-1)=1 -> 패딩 = 1*팽창 만큼
        #conv가 2번이니까 총 2*padding 추가.
        self.single_pad = (kernel_size - 1) * dilation
        # 2번 conv 하고, 최종 slice 할 크기 (2474개 받아야지.)
        self.total_pad = 2 * self.single_pad

        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               dilation=dilation,
                               padding=self.single_pad)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               dilation=dilation,
                               padding=self.single_pad)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # residual downsample if channel size changes
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: (N, in_channels, T)
        """
        # conv1
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.dropout1(y)
        
        # conv2
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.dropout2(y)

        # 인과적으로 넣었으니까 (causal = True) 길이 맞춰야함

        if self.causal:
            # total_pad = 2*(kernel_size-1)*dialation
            if self.total_pad > 0:
                y = y[:, :, :-self.total_pad] #오른편에 2개 느는거 없애기 2팽창해서 그럼

        if self.downsample is not None:
            x = self.downsample(x)

        # 타임 슬라이스 적절하게 잘 하려면
        # x가 (N, 채널 인풋, T) 인거 반영해서 채널 아웃풋 도 똑같아야해
        # 지금 팽창 1보다 크게, 긍께 2정도 했잖아, x 패딩 안했으니까 길이가 걍 T야.
        # downsample 하고나면 (N, 아웃풋 채널, T) -> 길이가 T+2가아니라 T임

        if self.causal and self.total_pad > 0 :
            #지금 x 길이 T니까 자동으로 맞다 생각하고 이거 pass때릴게
            pass

        out = x + y # shape (N, out_channels, T)

        return out

class TCN(nn.Module):
    """
    (in_channels, T) -> (3, T) 형태를 출력하도록 설계
    여기서는 Forcing term f(t)를 3차원(XYZ)으로 내놓기
        TCN for 6D DMP:
    - Input shape : (batch, 8, T)  ->  ( (s, t_lin, goal-start(6D)) => 8 channels )
    - Output shape: (batch, 6, T)  ->  (6D forcing term)
    - in_ch=8: -> 채널 0 = s(t), 채널 1 = t_lin, 채널 2~7 = (goal - start) (6D).
    - out_ch=6: 매 시점 6차원 forcing term.
    - causal=True 시, 오른쪽 패딩을 잘라 causal conv를 흉내냅니다.
    """
    def __init__(self, in_ch=8, out_ch=6, num_channels=32,
                 levels=4, kernel_size=2, dropout=0.1, causal=True):
        super(TCN, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_channels = num_channels
        self.levels = levels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.causal = causal

        layers = []
        current_in = in_ch # 기본 1채널(= s(t)만 입력)이라 가정
        # 필요시 in_ch를 늘리면(예: 2,3...) start/goal 등도 추가 채널로 넣을 수 있음

        for i in range(levels):
            block = TCNBlock(
                in_channels=current_in,
                out_channels=num_channels,
                kernel_size=kernel_size,
                dilation=2**i,
                dropout=dropout,
                causal=causal
            )
            layers.append(block)
            current_in = num_channels

        self.network = nn.Sequential(*layers)
        # 최종으로 out_ch (num_channels, T) -> (3, T) 로 변환하기 위한 1x1 conv
        self.final_conv = nn.Conv1d(num_channels, out_ch, kernel_size=1)

    def forward(self, x):
        """
        x: shape (N, in_ch=8, T)
        return: shape (N, out_ch=6, T)
        """
        out = self.network(x)
        out = self.final_conv(out)  # (batch, 3, T)
        return out


#######################################################
# 3) End-to-End DMP + TCN 모델 (수정 버전)
#######################################################
class DMP6D(nn.Module): #초기속도 어떻게 수정한담...?
    """
    TCN으로부터 6D forcing term f(t)을 예측 -> 
    DMP 방정식(6차원)으로 적분 -> x(t) in R^6 (x,y,z,roll,pitch,yaw)
    """
    def __init__(self,
                 alpha_z=25.0,
                 beta_z=25.0/4.0,
                 tau=1.0,
                 in_ch=8,     # s(t)+ t_lin + (goal-start)6D => 8채널
                 out_ch=6,    # 6D forcing
                 num_channels=32,
                 levels=4,
                 kernel_size=2,
                 dropout=0.1,
                 causal=True):
        super(DMP6D, self).__init__()
        self.alpha_z = alpha_z
        self.beta_z  = beta_z
        self.tau     = tau

        # TCN: (N,8,T)->(N,6,T)
        self.tcn = TCN(in_ch=in_ch,
                       out_ch=out_ch,
                       num_channels=num_channels,
                       levels=levels,
                       kernel_size=kernel_size,
                       dropout=dropout,
                       causal=causal)

    def forward(self, net_input, start_pose, goal_pose, dt_array=None):
        """
        net_input: (N, 8, T)
          - channel0: s(t)
          - channel1: t_lin
          - channel2..7: (goal-start) 6D
        start_pose: (N,6)
        goal_pose : (N,6)
        dt_array  : None or shape (T-1,) or shape (N, T-1)
          - batch_size=1 가정 시 (T-1,)이 일반적
        return: x_out: (N,6,T) -> 6D DMP 적분 결과
        """
        # 1) TCN -> f_pred(t): shape (N,6,T)
        f_pred = self.tcn(net_input)  # (N,6,T)

        N, _, T = f_pred.shape
        device = f_pred.device

        # 2) 준비: x_out,v_out
        x_out = torch.zeros(N, 6, T, device=device)
        v_out = torch.zeros(N, 6, T, device=device)

        x_out[:, :, 0] = start_pose  # (N,6)
        # (선택) v_out[:, :, 0] = ?

        # 3) dt 설정
        if dt_array is None:
            # 일정 dt
            dt_list = [1.0/(T-1)]*(T-1)
        else:
            dt_list = dt_array  # shape (T-1,) or (N,T-1)

        # 4) DMP 적분 (for문)
        for t in range(1, T):
            # s(t) = net_input[:,0,t] (N,) -> unsqueeze(-1)->(N,1)
            s_t = net_input[:, 0, t].unsqueeze(-1)  # (N,1)

            # forcing term
            f_t = f_pred[:, :, t]  # (N,6)

            # a(t) = alpha_z [ beta_z (goal-x) - v ] + f*s
            a = (self.alpha_z
                 * (self.beta_z*(goal_pose - x_out[:, :, t-1])
                    - v_out[:, :, t-1])
                 + f_t*s_t) / self.tau

            # dt_i
            if isinstance(dt_list, list):
                # => [float, float, ...] length T-1
                dt_i = dt_list[t-1]
            elif len(dt_list.shape) == 1:
                # => shape (T-1,)
                dt_i = dt_list[t-1]
            else:
                # => shape (N, T-1)
                dt_i = dt_list[:, t-1]  # (N,)

            v_out[:, :, t] = v_out[:, :, t-1] + a * dt_i
            x_out[:, :, t] = x_out[:, :, t-1] + (v_out[:, :, t]/self.tau)*dt_i

        return x_out


#######################################################
# 4) Trainer 함수
#######################################################
class Trainer6D:
    """
    - 1) Dataset & DataLoader
    - 2) Model (TCN+DMP 6D)
    - 3) Training Loop
    - 4) Generate/Inference
    """
    def __init__(self,
                 csv_list,
                 alpha_x=4.0, tau=1.0,
                 alpha_z=25.0, beta_z=25.0/4.0,
                 lr=1e-3, num_channels=32, levels=4,
                 kernel_size=2, dropout=0.1, causal=True,
                 num_epochs=50,
                 log_dir="./runs_tcn_dmp6d"):
        """
        csv_list:  예) [r"data/a.csv", r"data/b.csv", ...]
        alpha_x, tau : canonical system
        alpha_z, beta_z: DMP dynamical params
        num_channels, levels: TCN hparams
        log_dir: tensorboard 로그 디렉토리
        """
        self.csv_list = csv_list
        self.alpha_x = alpha_x
        self.tau = tau
        self.alpha_z = alpha_z
        self.beta_z = beta_z
        self.lr = lr
        self.num_channels = num_channels
        self.levels = levels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.causal = causal
        self.num_epochs = num_epochs
        self.log_dir = log_dir

        # 1) Dataset & DataLoader
        self.dataset = DMPDataset6D(
            csv_files=self.csv_list,
            alpha_x=self.alpha_x,
            tau=self.tau,
        )
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        # 2) Model 준비
        self.model = DMP6D(
            alpha_z=self.alpha_z, beta_z=self.beta_z, tau=self.tau,
            in_ch=8,    # s(t), t_lin, (goal-start)6D => 8
            out_ch=6,   # 6D forcing
            num_channels=self.num_channels,
            levels=self.levels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            causal=self.causal
        )
        self.model = self.model.float()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 3) Optimizer / Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # 4) TensorBoard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, str(current_time)), flush_secs=30)

    def train(self):
        """
        메인 훈련 루프
        step(Epoch*Batch) 
        step으로 매기는 이유는 RL이랑도 비교해야해서. epoch로 안매김
        """
        total_steps = self.num_epochs * len(self.loader)
        pbar = tqdm(total = total_steps, desc="Training", unit="batch")
        global_step = 0  # for step-based logging

        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            for batch_idx, sample in enumerate(self.loader):
                # sample: dict keys -> ['pose_demo','start','goal','multi_in','dt_array']
                pose_demo = sample['pose_demo'][0].float().to(self.device)  # (T,6)
                start_6d  = sample['start'][0].float().to(self.device)      # (6,)
                goal_6d   = sample['goal'][0].float().to(self.device)       # (6,)
                multi_in  = sample['multi_in'][0].float().to(self.device)   # (T,8)

                dt_array_obj = sample['dt_array'][0]  # shape(T-1,) or None
                if dt_array_obj is not None:
                    dt_array_obj = dt_array_obj.numpy()  # or .float() => shape(T-1,)
                    dt_array_obj = torch.tensor(dt_array_obj, dtype=torch.float32, device=self.device)

                # reshape for TCN: (1,8,T)
                T = pose_demo.shape[0]
                net_input = multi_in.permute(1,0).unsqueeze(0)  # (batch=1, in_ch=8, T)
                start_6d  = start_6d.unsqueeze(0)   # (1,6)
                goal_6d   = goal_6d.unsqueeze(0)    # (1,6)

                # === 2) Forward ===
                self.optimizer.zero_grad()
                # forward => x_out: (1,6,T)
                x_out = self.model(net_input, start_6d, goal_6d, dt_array=dt_array_obj)
                # pose_demo -> (1,6,T)
                pose_demo_3d = pose_demo.permute(1,0).unsqueeze(0)  # (1,6,T)
                loss = self.criterion(x_out, pose_demo_3d)
                
                # === 3) Backward ===
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # (A) Per-step TB logging
                self.writer.add_scalar("Loss/Train_step", loss.item(), global_step)

                # (B) Forcing term histogram (optional)
                #    f_pred is the TCN output before DMP integration, can be accessed if we
                #    separate out self.tcn call. For example:
                #    f_pred = self.model.tcn(net_input)
                #    self.writer.add_histogram("ForcingTerm", f_pred.cpu().data, global_step)

                # (C) Model weight histogram (optional, every N steps to reduce log size)
                # if global_step % 100 == 0:
                #     for name, param in self.model.named_parameters():
                #         self.writer.add_histogram(f"Weights/{name}", param, global_step)

                global_step += 1
                pbar.update(1)

            # === 4) epoch-based logging (avg loss, final goal error, dimension-wise RMSE etc.)
            avg_loss = epoch_loss / len(self.loader)
            self.writer.add_scalar("Loss/Train", avg_loss, epoch)

            # Final Goal Error
            with torch.no_grad():
                final_err = (x_out[0, :, -1] - pose_demo_3d[0, :, -1]).abs().mean()
            self.writer.add_scalar("Metric/FinalGoalError", final_err.item(), epoch)

            # Dimension-wise RMSE (x,y,z,roll,pitch,yaw)
            # compute RMSE on entire trajectory for each dimension
            with torch.no_grad():
                diff = x_out - pose_demo_3d  # (1,6,T)
                diff_sq = diff**2
                mse_dim = diff_sq.mean(dim=(0,2))  # (6,) => each dimension
                rmse_dim = torch.sqrt(mse_dim)     # (6,)
            dim_names = ["x","y","z","roll","pitch","yaw"]
            for i, dim_name in enumerate(dim_names):
                self.writer.add_scalar(f"DimRMSE/{dim_name}", rmse_dim[i].item(), epoch)

            if epoch % 100 == 0:
                print(f"[Epoch {epoch}/{self.num_epochs}] avg_loss={avg_loss:.6f}, final_err={final_err:.6f}")

        self.writer.close()
        pbar.close()
        print("Training complete.")

    def generate(self, csv_path, start_6d_override=None):
        """
        모델 학습 후, 'csv_path' 파일(길이 T)에서
        (6D) DMP를 재생성.
        - pose_demo, multi_in, etc.를 재활용하거나
          또는 간단히 'start','goal'만 새로 세팅 가능
        - start_6d_override: 만약 None이 아니면, CSV로부터 읽은 start 대신 이 값을 사용
        """
        traj = Trajectory.load_csv(csv_path)

        pose_6d = np.concatenate([traj.xyz, traj.euler_angles], axis=1)  # (T,6)
        T = pose_6d.shape[0]

        # CSV에서 읽은 start, goal
        default_start_6d = pose_6d[0]
        default_goal_6d  = pose_6d[-1]

        # 2) start_6d_override가 있으면, 그걸 우선사용
        start_6d = start_6d_override if start_6d_override is not None else default_start_6d
        goal_6d = default_goal_6d
    
        # canonical s(t)
        t_lin = np.linspace(0,1,T)
        s = np.exp(-self.alpha_x * t_lin / self.tau)  # (T,)

        # (goal-start) 6D
        gs_6d = goal_6d - start_6d
        repeated_gs = np.tile(gs_6d, (T,1))  # (T,6)

        s_col = s.reshape(T,1)
        t_col = t_lin.reshape(T,1)
        net_input_np = np.concatenate([s_col, t_col, repeated_gs], axis=1)  # (T,8)

        # Convert to torch
        device = next(self.model.parameters()).device
        net_input_torch = torch.tensor(net_input_np, dtype=torch.float32, device=device).permute(1,0).unsqueeze(0)
        # shape (1,8,T)

        start_6d_torch = torch.tensor(start_6d, dtype=torch.float32, device=device).unsqueeze(0) # (1,6)
        goal_6d_torch  = torch.tensor(goal_6d, dtype=torch.float32, device=device).unsqueeze(0)  # (1,6)

        self.model.eval()
        with torch.no_grad():
            x_out = self.model(net_input_torch, start_6d_torch, goal_6d_torch) # (1,6,T)

        x_out_np = x_out.squeeze(0).permute(1,0).cpu().numpy()  # (T,6)
        # x_out_np[:, :3] => (x,y,z)
        # x_out_np[:, 3:] => (roll,pitch,yaw)

        # 새 Trajectory
        new_traj = traj.copy()
        new_traj.xyz = x_out_np[:, :3]  # (T,3)

        # rpy도 갱신하고 싶다면...
        # new_traj.rpy = x_out_np[:, 3:]  # (T,3)

        return new_traj
    
    def generate_sequence(self, csv_paths):
        """
        예) csv_paths = ['processed_a.csv', 'processed_b.csv', 'processed_c.csv', 'processed_d.csv']
        a->b->c->d 순서로 DMP를 체이닝하여 하나의 big trajectory를 생성
        """
        all_xyz = []
        all_euler_angles = []
        all_joints = []
        all_timestamps = []
        all_grippers = []
        all_weights = []
        current_time_offset = 0.0

        sub_traj = None
        for i, csv_path in enumerate(csv_paths):
            if i == 0:
                sub_traj = self.generate(csv_path, start_6d_override=None)
            else:
                last_xyz = sub_traj.xyz[-1]  # (3,)
                last_rpy = sub_traj.euler_angles[-1]  # (3,)
                last_6d = np.concatenate([last_xyz, last_rpy])
                sub_traj = self.generate(csv_path, start_6d_override=last_6d)
            
            if sub_traj is None:
                print(f"Failed to generate sub-traj for {csv_path}")
                return None
            
            all_xyz.append(sub_traj.xyz)
            all_euler_angles.append(sub_traj.euler_angles)
            all_timestamps.append(sub_traj.timestamp + current_time_offset)
            current_time_offset = all_timestamps[-1][-1]
            
            all_joints.append(sub_traj.joints)
            
            if sub_traj.gripper is not None:
                all_grippers.append(sub_traj.gripper)
            if sub_traj.weight is not None:
                all_weights.append(sub_traj.weight)

        return Trajectory(all_timestamps[0], all_xyz[0], all_euler_angles[0], all_joints[0], all_grippers[0], all_weights[0])


#######################################################
# 5) Main
#######################################################
if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")

    # 0) CSV 파일 리스트
    csv_list = [
        os.path.join(base_dir, "data", "processed_a.csv"),
        os.path.join(base_dir, "data", "processed_b.csv"),
        os.path.join(base_dir, "data", "processed_c.csv"),
        os.path.join(base_dir, "data", "processed_d.csv"),
    ]

    # 1) Dataset 확인
    dataset = DMPDataset6D(
        csv_files=csv_list,
        alpha_x=3.5,
        tau=1.0,
    )
    print("len(dataset) =", len(dataset))
    if len(dataset) > 0:
        sample0 = dataset[0]
        pose_demo = sample0['pose_demo']   # (T,6)
        multi_in  = sample0['multi_in']    # (T,8)
        print("pose_demo shape:", pose_demo.shape)
        print("multi_in shape:", multi_in.shape)
        print("start shape:", sample0['start'].shape)  # (6,)
        print("goal shape:", sample0['goal'].shape)    # (6,)
        dt_arr = sample0['dt_array']
        if dt_arr is not None:
            print("dt_array shape:", dt_arr.shape)
        else:
            print("dt_array is None")

    # 2) Trainer 생성 (TCN+DMP 6D) 
    # 배치 수(len(self.loader))는 len(self.dataset)(=4)÷batch_size(=1)→ 4 
    # 에포크 수(self.num_epochs)는 main에서 num_epochs=100 → 100

    trainer = Trainer6D(
        csv_list=csv_list,
        alpha_x=3.5,
        tau=1.0,
        alpha_z=30.0,
        beta_z=3.0,
        lr=1e-3,
        num_channels=32,
        levels=4,
        kernel_size=2,
        dropout=0.25,
        causal=True,
        num_epochs=12000,
        log_dir=os.path.join(base_dir, "runs", "tcn_dmp_6d")
    )

    # 3) 학습
    trainer.train()

     # 4) 모델 checkpoint (.pt) 저장
    pt_save_path = os.path.join(base_dir, "model_origin.pt")
    torch.save(trainer.model.state_dict(), pt_save_path)
    print("Model checkpoint saved to", pt_save_path)

    # 5) 추론(Generate): 예) 첫 번째 CSV로부터 재생성
    test_csv = csv_list[0]
    new_traj = trainer.generate_sequence(csv_list)
    if new_traj is not None:
        # 추가된 부분: 데모 궤적 시각화
        sub_csv_list = [
            os.path.join(base_dir, "data", "processed_a.csv"),
            os.path.join(base_dir, "data", "processed_b.csv"),
            os.path.join(base_dir, "data", "processed_c.csv"),
            os.path.join(base_dir, "data", "processed_d.csv"),
        ]

        # 1) 각각의 데모 궤적 로드
        demo_trajs = []
        for i, csv_path in enumerate(sub_csv_list):
            demo_traj = Trajectory.load_csv(csv_path)
            demo_trajs.append(demo_traj)

        # 2) 체이닝으로 최종 궤적 생성
        final_traj = trainer.generate_sequence(sub_csv_list)


        # 3) 플롯
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
 
 
        # 각 데모(a/b/c/d) 그리기
        colors = ['blue', 'green', 'red', 'orange']
        labels = ['demo_a', 'demo_b', 'demo_c', 'demo_d']
        for i, dtraj in enumerate(demo_trajs):
            ax.plot(dtraj.xyz[:,0], dtraj.xyz[:,1], dtraj.xyz[:,2],
                    label=labels[i], color=colors[i])

        # 체이닝된 최종 궤적
        ax.plot(final_traj.xyz[:,0], final_traj.xyz[:,1], final_traj.xyz[:,2],
                label="Chained DMP", color='magenta')

        ax.legend()
        ax.set_title("Demos (a/b/c/d) vs. Chained DMP")
        plt.show()

        # 시각화 결과 파일 저장 (jpg & gif)
        figs_dir = os.path.join(base_dir, "figs")
        os.makedirs(figs_dir, exist_ok=True)

        final_traj.save_csv("generated_trajectory_origin.csv")
        
        # 정적 이미지 (jpg) 저장
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(final_traj.xyz[:,0], final_traj.xyz[:,1], final_traj.xyz[:,2], label='Generated Trajectory', color='orange')
        ax.legend()
        plt.title("Generated Trajectory")
        jpg_path = os.path.join(figs_dir, "trajectory.jpg")
        plt.savefig(jpg_path)
        print("Saved jpg to", jpg_path)
        plt.close(fig)
        
        # 애니메이션 (gif) 생성 및 저장
        fig_anim = plt.figure()
        ax_anim = fig_anim.add_subplot(111, projection='3d')
        def update(num):
            ax_anim.clear()
            ax_anim.plot(final_traj.xyz[:num,0], final_traj.xyz[:num,1], final_traj.xyz[:num,2],
                         label='Generated Trajectory', color='orange')
            # 축 범위 설정 (여기서는 데이터 전체 범위를 사용)
            ax_anim.set_xlim([min(np.min(final_traj.xyz[:,0]), np.min(final_traj.xyz[:,0])) - 1,
                              max(np.max(final_traj.xyz[:,0]), np.max(final_traj.xyz[:,0])) + 1])
            ax_anim.set_ylim([min(np.min(final_traj.xyz[:,1]), np.min(final_traj.xyz[:,1])) - 1,
                              max(np.max(final_traj.xyz[:,1]), np.max(final_traj.xyz[:,1])) + 1])
            ax_anim.set_zlim([min(np.min(final_traj.xyz[:,2]), np.min(final_traj.xyz[:,2])) - 1,
                              max(np.max(final_traj.xyz[:,2]), np.max(final_traj.xyz[:,2])) + 1])
            ax_anim.legend()
            plt.title("Trajectory Animation")
            return ax_anim,
        frames = final_traj.xyz.shape[0]
        ani = animation.FuncAnimation(fig_anim, update, frames=frames, interval=100, blit=False)
        gif_path = os.path.join(figs_dir, "trajectory.gif")
        ani.save(gif_path, writer='pillow')
        print("Saved gif to", gif_path)
        plt.close(fig_anim)
