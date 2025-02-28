import os
import csv
import time
import keyboard  # 키보드 입력 감지 모듈
from xarm.wrapper import XArmAPI
import serial  # 시리얼 통신을 위한 모듈

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

# 시리얼 포트 설정 (아두이노의 포트와 일치해야 합니다)
serial_port = 'COM8'  # Windows에서는 'COM3'과 같은 형태일 수 있음
baud_rate = 9600
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# 현재 좌표 및 각도 가져오는 함수
def get_robot_state():
    coordinates = arm.get_position(is_radian=False)
    angles = arm.get_servo_angle()
    return coordinates[1], angles[1]

# 시리얼을 통해 scale 값을 읽어오는 함수
def get_scale_value():
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        try:
            weight = float(line.split(" ")[-2])  # "측정된 무게: xx.xx g" 형태에서 무게 값 추출
            return weight
        except ValueError:
            return None
    return None

# 수동 모드로 변경
arm.set_mode(2)
arm.set_state(0)

gripper_start_state = 0
arm.set_suction_cup(gripper_start_state)
is_gripper_open = gripper_start_state

# CSV 파일 작성
with open(datafile_path, 'w', newline='') as csvfile:
    fieldnames = [
        'timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw',
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
        'gripper', 'scale'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    try:
        start_time = time.time()
        while True:
            if keyboard.is_pressed('up'):
                arm.set_suction_cup(True)
                is_gripper_open = 1
            elif keyboard.is_pressed('down'):
                arm.set_suction_cup(False)
                is_gripper_open = 0
            
            coordinates, angles = get_robot_state()
            timestamp = round((time.time() - start_time) * 1000, 0)
            
            scale_value = get_scale_value()
            if scale_value is None:
                scale_value = 0  # 시리얼 데이터를 못 읽었을 경우 기본값 설정
            
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
                'joint6': angles[5],
                'gripper': is_gripper_open,
                'scale': scale_value
            })
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("Data recording stopped.")

arm.set_mode(0)
arm.disconnect()
ser.close()

# keyboard 모듈이 없을 경우 설치 방법
# pip install keyboard pyserial
