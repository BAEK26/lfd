import csv
from UFactoryLite6 import UFactoryLite6
import time

class KinestheticTeaching:
    def __init__(self):
        self.robot = UFactoryLite6()
        self.recorded_path = []

    def record_trajectory(self, duration=10):
        start_time = time.time()
        while time.time() - start_time < duration:
            joints = self.robot.get_joint_positions()
            self.recorded_path.append(joints)
            time.sleep(0.1)  # 0.1초 간격으로 기록

        # 기록된 경로 저장
        with open('kinesthetic_trajectory.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.recorded_path)

    def replay_trajectory(self, path='kinesthetic_trajectory.csv'):
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                joint_positions = [float(x) for x in row]
                self.robot.move_joints(joint_positions)
                time.sleep(0.1)

# 시연을 위해 호출
kinesthetic_teaching = KinestheticTeaching()
kinesthetic_teaching.record_trajectory()
kinesthetic_teaching.replay_trajectory()
