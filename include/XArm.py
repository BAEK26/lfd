from xarm.wrapper import XArmAPI

try:
    from trajectory import Trajectory  # 실행 코드일 경우
except ImportError:
    from src.trajectory import Trajectory  # 모듈 코드인 경우

class XArm:
    def __init__(self, ip = "127.0.0.1"):
        self.arm = XArmAPI(ip)
        self.arm.connect()

    def inverse_kinematics(self, xyz, euler_angles, input_is_radian=None, return_is_radian=None):
        pose = list(xyz) + list(euler_angles)
        joint_angles = self.arm.get_inverse_kinematics(pose, input_is_radian, return_is_radian)[1][0:6]
        return joint_angles

    def forward_kinematics(self, joints, input_is_radian=None, return_is_radian=None):
        x, y, z, roll, pitch, yaw = self.arm.get_forward_kinematics(joints, input_is_radian, return_is_radian)[1]
        return [x, y, z], [roll, pitch, yaw]

    def trajectory_transform(self, trajectory, using='IK'):
        # 새로운 Trajectory 객체 생성
        new_trajectory = Trajectory(
            trajectory.timestamp,
            trajectory.xyz.copy(),
            trajectory.euler_angles.copy(),
            trajectory.joints.copy(),
            trajectory.gripper.copy(),
            trajectory.target
        )
        
        if using == 'IK':
            for i in range(len(trajectory.timestamp)):
                # 각 시점의 xyz와 euler_angles를 사용하여 joint_angles를 구함
                xyz = trajectory.xyz[i]
                euler_angles = trajectory.euler_angles[i]
                joint_angles = self.inverse_kinematics(xyz, euler_angles)
                new_trajectory.joints[i] = joint_angles
        elif using == 'FK':
            for i in range(len(trajectory.timestamp)):
                # 각 시점의 joint_angles를 사용하여 xyz와 euler_angles를 구함
                joints = trajectory.joints[i]
                xyz, euler_angles = self.forward_kinematics(joints)
                new_trajectory.xyz[i] = xyz
                new_trajectory.euler_angles[i] = euler_angles

        return new_trajectory
    

# 사용 예제
if __name__ == "__main__":

    # CSV로부터 객체 생성
    traj = Trajectory.load_csv("processed_sumin_a.csv")

    # XArm 객체 생성
    xarm = XArm()

    # traj1 시각화(기본값은 C - cartesian)
    path1_DMPed_IK = xarm.trajectory_transform(traj)
    path1_DMPed_IK.target = 'J'
    path1_DMPed_IK.show()