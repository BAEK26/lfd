from trajectory import Trajectory

import os
import numpy as np

# CSV 데이터를 펌핑합니다.
# :param: 목표율(기본값은 초당 360개)
# :return: trajectoryectory 객체
def pumping(trajectory, target_rate=360):
    interval_ms = 1000 / target_rate
    new_timestamp = np.arange(trajectory.timestamp[0], trajectory.timestamp[-1], interval_ms)
    
    interpolated_xyz = np.array([
        np.interp(new_timestamp, trajectory.timestamp, trajectory.xyz[:, i]) for i in range(3)
    ]).T
    interpolated_euler = np.array([
        np.interp(new_timestamp, trajectory.timestamp, trajectory.euler_angles[:, i]) for i in range(3)
    ]).T
    interpolated_joints = np.array([
        np.interp(new_timestamp, trajectory.timestamp, trajectory.joints[:, i]) for i in range(trajectory.joints.shape[1])
    ]).T
    interpolated_gripper = np.round(np.interp(new_timestamp, trajectory.timestamp, trajectory.gripper)).astype(int)        
    
    return Trajectory(new_timestamp, interpolated_xyz, interpolated_euler, interpolated_joints, interpolated_gripper)


# 실행 코드
if __name__ == "__main__":

    base_dir = r"C:\Users\박수민\Documents\neoDMP" # base 경로 (알맞게 수정)
    load_path = os.path.join(base_dir, "data", "test_sumin_a.csv") # CSV 로드 파일 경로
    
    # CSV로부터 trajectoryectory 객체 생성
    traj = Trajectory.load_csv(load_path)

    # 궤적 펌핑
    pumped_traj = pumping(traj, target_rate=360)

    # 궤적 저장
    save_path = os.path.join(base_dir, "data", "pumped_sumin_a.csv") # CSV 저장 파일 경로
    pumped_traj.save_csv(save_path)
    

