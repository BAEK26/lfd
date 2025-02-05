import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import os
from copy import deepcopy


# 향후 쓰이게 될 궤적들의 기본 단위를 정의하는 클래스.
# 데이터 저장 및 시각화를 다룹니다.
class Trajectory:

    # 기본 생성자 : 각 값들에 대한 배열을 인자로 받습니다(잘 사용 안 함).
    def __init__(self, timestamp, xyz, euler_angles, joints, gripper, target='C'):
        self.timestamp = np.array(timestamp)
        self.xyz = np.array(xyz)
        self.euler_angles = np.array(euler_angles)
        self.joints = np.array(joints)
        self.gripper = np.array(gripper)
        self.target = target  

        """
        self.target을 어떻게 설정하느냐에 따라 향후 시각화, DMP, 필터링 등에서 적용 대상이 달라집니다.
        C, E, J (각각 Cartesian, Euler, Joint) 중 하나로 설정하시면 됩니다.


        """
    
    # CSV로부터 Trajectory 인스턴스를 반환합니다(주로 사용).
    # :param file_path: CSV 로드 파일 경로
    # :return: Trajectory 객체
    @classmethod
    def load_csv(cls, file_path):
        data = pd.read_csv(file_path)
        xyz = np.vstack((data['x'].values, data['y'].values, data['z'].values)).T
        euler_angles = np.vstack((data['roll'].values, data['pitch'].values, data['yaw'].values)).T
        joints = np.vstack((data['joint1'].values, data['joint2'].values, data['joint3'].values, 
                            data['joint4'].values, data['joint5'].values, data['joint6'].values)).T
        return cls(data['timestamp'].values, xyz, euler_angles, joints, data['gripper'].values)

    # Trajectory 인스턴스를 CSV로 저장합니다.
    # :param file_path: CSV 저장 파일 경로
    def save_csv(self, file_path):
        data = pd.DataFrame({
            'timestamp': self.timestamp,
            'x': self.xyz[:, 0], 'y': self.xyz[:, 1], 'z': self.xyz[:, 2],
            'roll': self.euler_angles[:, 0], 'pitch': self.euler_angles[:, 1], 'yaw': self.euler_angles[:, 2],
            'joint1': self.joints[:, 0], 'joint2': self.joints[:, 1], 'joint3': self.joints[:, 2],
            'joint4': self.joints[:, 3], 'joint5': self.joints[:, 4], 'joint6': self.joints[:, 5],
            'gripper': self.gripper
        })
        data.to_csv(file_path, index=False, header=True)
    
    # traj2 = traj1.copy()와 같은 문법을 지원합니다.
    def copy(self):
        
        return deepcopy(self)
    
    # print(traj)와 같은 문법을 지원합니다.
    def __str__(self):
        data = pd.DataFrame({
            'timestamp': self.timestamp,
            'x': self.xyz[:, 0], 'y': self.xyz[:, 1], 'z': self.xyz[:, 2],
            'roll': self.euler_angles[:, 0], 'pitch': self.euler_angles[:, 1], 'yaw': self.euler_angles[:, 2],
            'joint1': self.joints[:, 0], 'joint2': self.joints[:, 1], 'joint3': self.joints[:, 2],
            'joint4': self.joints[:, 3], 'joint5': self.joints[:, 4], 'joint6': self.joints[:, 5],
            'gripper': self.gripper
        })
        return str(data)

    # traj.len()과 같은 문법을 지원합니다.
    def len(self):
        return len(self.timestamp)

    # traj[0:500]과 같은 문법을 지원합니다.
    def __getitem__(self, key):
        if isinstance(key, slice):
            return Trajectory(
                self.timestamp[key],
                self.xyz[key],
                self.euler_angles[key],
                self.joints[key],
                self.gripper[key],
                self.target
            )
        else:
            raise TypeError("Indexing must be done using slices (e.g., trajectory[start:end])")
         
            
    # Trajectory 데이터를 시각화하는 함수입니다. 
    # target에 따라 시각화 대상은 달라집니다.

    # 다음의 문법을 지원합니다.
    # traj.show(): 해당 Trajectory만 출력
    # Trajectory.show(traj1, traj2, ...): 여러 Trajectory를 동시에 출력
    @staticmethod
    def plot(*trajectories):
        if len(trajectories) == 1 and isinstance(trajectories[0], list):
            trajectories = trajectories[0]

        # 둘 이상의 궤적이 들어오면 첫 번째 궤적 기준으로 목표 설정
        plot_target = trajectories[0].target 

        # Joint 그래프 출력
        if plot_target == 'J':
            num_joints = trajectories[0].joints.shape[1]
            fig, axes = plt.subplots(num_joints, 1, figsize=(6, 6), sharex=True)

            for i, traj in enumerate(trajectories):
                color = plt.cm.jet(i / max(1, len(trajectories)-1)) # Trajectory별 색상 지정
                for j in range(num_joints): # 각 조인트(6개)에 대해 반복
                    axes[j].plot(traj.timestamp, traj.joints[:, j], color=color, alpha=0.8, label=f'Traj {i+1}')
                    axes[j].set_ylabel(f'Joint {j+1}')
                    axes[j].grid(True)

            axes[-1].set_xlabel("Time") # 마지막 축의 x-label 설정
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')

            plt.suptitle("Joint Trajectories Comparison")
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            plt.show()

        # 3D 그래프 출력 (Cartesian 또는 Euler)
        else: 
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
        
            for i, traj in enumerate(trajectories):
                color = plt.cm.jet(i / max(1, len(trajectories)-1)) # Trajectory별 색상 지정
                data = traj.euler_angles if plot_target == 'E' else traj.xyz

                # Trajectory 궤적 출력
                ax.plot(data[:, 0], data[:, 1], data[:, 2], color=color, label=f'Trajectory {i+1}')
                
                # 시작점, 종료점 표시
                ax.scatter(data[0, 0], data[0, 1], data[0, 2], color=color, marker='^', s=100, label=f'Start {i+1}')
                ax.scatter(data[-1, 0], data[-1, 1], data[-1, 2], color=color, marker='o', s=100, label=f'End {i+1}')
    
            ax.set_xlabel('Roll' if plot_target == 'E' else 'X')
            ax.set_ylabel('Pitch' if plot_target == 'E' else 'Y')
            ax.set_zlabel('Yaw' if plot_target == 'E' else 'Z')
            ax.legend()
            ax.set_title("Trajectories")
            plt.show()
    
    def show(self, *args):
        if not args:
            self.plot(self)
        else:
            self.plot(self, *args)


# 사용 예제
if __name__ == "__main__":

    # base 경로 (알맞게 수정)
    base_dir = r"C:\Users\박수민\Documents\neoDMP"

    # 실제 CSV 파일 경로
    path1 = os.path.join(base_dir, "data", "test_sumin_a.csv")
    path2 = os.path.join(base_dir, "data", "test_sumin_b.csv")
    
    # CSV로부터 객체 생성
    traj1 = Trajectory.load_csv(path1)
    traj2 = Trajectory.load_csv(path2)

    # traj1 시각화(기본값은 C - cartesian)
    traj1.show()  
    
    # __getitem__ 연산자 사용 예시
    
    traj1_half = traj1[0 : traj1.len() // 2] # '궤적 길이'가 아닌 '시간' 기준으로 절반입니다.
    Trajectory.show(traj1, traj1_half) 

    # 오일러 각 시각화
    traj1.target = 'E'
    Trajectory.show(traj1, traj2) # 먼저 오는 궤적의 target 기준으로 시각화합니다.

    # 조인트 시각화
    traj2.target = 'J'
    Trajectory.show(traj2, traj1) 

