import os
import csv
import sys
import time
from xarm.wrapper import XArmAPI

# 파일 이름 자동 생성: test%d.csv 형태
data_folder = 'data'
os.makedirs(data_folder, exist_ok=True)

# 기존 test 파일 중 마지막 번호 확인
existing_files = [f for f in os.listdir(data_folder) if f.startswith('test') and f.endswith('.csv')]
file_number = len(existing_files) + 1
datafile_path = os.path.join(data_folder, f'test{file_number}.csv')

print(f"Recording data to: {datafile_path}")

ip = "192.168.1.194"
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# Function to get current coordinates and angles
def get_robot_state():
    coordinates = arm.get_position(is_radian=False)
    angles = arm.get_servo_angle()
    return coordinates[1], angles[1]

# Turn on manual mode before recording
arm.set_mode(2)
arm.set_state(0)
arm.set_suction_cup(False)

# Open the CSV file to save the data
with open(datafile_path, 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    try:
        start_time = time.time()
        while True:
            coordinates, angles = get_robot_state()
            timestamp = round((time.time() - start_time) * 1000, 0)
            
            writer.writerow({
                'timestamp': timestamp,
                'x': coordinates[0],
                'y': coordinates[1],
                'z': coordinates[2],
                'roll': coordinates[3],
                'pitch': coordinates[4],
                'yaw': coordinates[5],
                'joint1': angles[0],
                'joint2': angles[1],
                'joint3': angles[2],
                'joint4': angles[3],
                'joint5': angles[4],
                'joint6': angles[5]
            })
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Data recording stopped.")

arm.set_mode(0)
arm.disconnect()
