import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import os
from copy import deepcopy

base_dir = os.path.join(os.path.dirname(__file__), '..')
data_dir = os.path.join(base_dir, 'data')


# 향후 쓰이게 될 궤적들의 기본 단위를 정의하는 클래스.
# 데이터 저장 및 시각화를 다룹니다.
class Trajectory:

    # 기본 생성자 : 각 값들에 대한 배열을 인자로 받습니다(잘 사용 안 함).
    def __init__(self, timestamp, xyz, euler_angles, joints, gripper, target='C'):
        """
        self.timestamp - self.joints : Manual mode 기반 Demo로부터(csv에 존재)
        self.gripper : Demo 교시 과정에서 수동으로 설정한 데이터(역시 csv에 존재)
        self.target : 시각화, DMP, 필터링 등에서의 피적용 대상
    
        target은 C, E, J (각각 Cartesian, Euler, Joint) 중 하나로 설정하시면 됩니다.
        """
        self.timestamp = np.array(timestamp)
        self.xyz = np.array(xyz)
        self.euler_angles = np.array(euler_angles)
        self.joints = np.array(joints)
        self.gripper = np.array(gripper)
        self.target = target  


    # CSV로부터 Trajectory 인스턴스를 반환합니다(주로 사용).
    # :param file_name: CSV 로드 파일 이름
    # :return: Trajectory 객체
    @classmethod
    def load_csv(cls, file_name=None):

        # 파일 이름이 지정되지 않았다면 가장 최근 csv 파일을 가져옵니다.
        if file_name is None:
            existing_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
            if not existing_files:
                print("No CSV files found in the directory.")
                return
            existing_files.sort(key=lambda f: os.path.getctime(os.path.join(data_dir, f)), reverse=True)  # 최신 파일 찾기
            file_name = existing_files[0]
        
        # 파일 이름이 제시되었다면
        else: 
            file_path = os.path.join(data_dir, file_name) # 파일 경로 설정

            # 파일명에 csv 미포함 시 추가
            if ".csv" not in file_name:
                file_path += '.csv'

            # 해당 파일 존재 여부 확인
            if not os.path.exists(file_path):
                print("File not found.")
                return
            
            data = pd.read_csv(file_path) # 데이터 불러오기

        # 각 데이터 넘파이 배열화하기
        xyz = np.vstack((data['x'].values, data['y'].values, data['z'].values)).T
        euler_angles = np.vstack((data['roll'].values, data['pitch'].values, data['yaw'].values)).T
        joints = np.vstack((data['joint1'].values, data['joint2'].values, data['joint3'].values, 
                            data['joint4'].values, data['joint5'].values, data['joint6'].values)).T
        
        # Trajectory 클래스 반환
        return cls(data['timestamp'].values, xyz, euler_angles, joints, data['gripper'].values)


    # Trajectory 인스턴스를 CSV로 저장합니다.
    # :param file_name: CSV 저장 파일 이름
    def save_csv(self, file_name):

        # 파일 경로 설정
        file_path = os.path.join(data_dir, file_name)

        # 파일명에 csv 미포함 시 추가
        if ".csv" not in file_name:
            file_path += '.csv'

        # Trajectory 객체를 csv로 변환    
        data = pd.DataFrame({
            'timestamp': self.timestamp,
            'x': self.xyz[:, 0], 'y': self.xyz[:, 1], 'z': self.xyz[:, 2],
            'roll': self.euler_angles[:, 0], 'pitch': self.euler_angles[:, 1], 'yaw': self.euler_angles[:, 2],
            'joint1': self.joints[:, 0], 'joint2': self.joints[:, 1], 'joint3': self.joints[:, 2],
            'joint4': self.joints[:, 3], 'joint5': self.joints[:, 4], 'joint6': self.joints[:, 5],
            'gripper': self.gripper
        })

        # 파일 경로에 저장
        data.to_csv(file_path, index=False, header=True)
    

    # traj2 = traj1.copy()와 같은 문법을 지원합니다.
    def copy(self): return deepcopy(self)
    

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
         

    # Trajectory 데이터를 시각화하는 내부 함수입니다. 
    # target에 따라 시각화 대상은 달라집니다.
    @staticmethod
    def __plot(*trajectories):

        # 첫 번째 궤적 기준으로 목표 설정
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

    # 외부적으로 시각화는 이 함수를 사용합니다.
    # traj.show(): 해당 Trajectory만 출력
    # Trajectory.show(traj1, traj2, ...): 여러 Trajectory를 동시에 출력
    def show(self, *args):
        if not args:
            self.__plot(self)
        else:
            self.__plot(self, *args) 
