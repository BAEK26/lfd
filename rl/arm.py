from UFactoryLite6 import UFactoryLite6

class RobotArmController:
    def __init__(self):
        self.robot = UFactoryLite6()

    def move_arm(self, joint_positions):
        self.robot.move_joints(joint_positions)

    def get_state(self):
        return self.robot.get_joint_positions()
