from xarm.wrapper import XArmAPI

class UFactoryLite6:
    def __init__(self, ip='192.168.1.194'):
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

    def move_joints(self, joint_positions):
        self.arm.set_servo_angle(angle=joint_positions, is_radian=False)

    def get_joint_positions(self):
        return self.arm.get_servo_angle()

    def disconnect(self):
        self.arm.disconnect()

# 로봇 암 초기화
robot_arm = UFactoryLite6()
