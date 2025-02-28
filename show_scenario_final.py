import os
import csv
import time
import keyboard
import serial
from xarm.wrapper import XArmAPI

# -------------------------------
# 1) 설정: 로봇 및 시리얼 포트 초기화
# -------------------------------
data_folder = 'data'
os.makedirs(data_folder, exist_ok=True)

existing_files = [f for f in os.listdir(data_folder) if f.startswith('test') and f.endswith('.csv')]
file_number = len(existing_files) + 1
datafile_path = os.path.join(data_folder, f'test{file_number}.csv')

print(f"Recording data to: {datafile_path}")

# 로봇 설정
ip = "192.168.1.194"
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# 시리얼 포트 설정
serial_port = 'COM4'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate, timeout=0)  # timeout=0으로 즉시 응답

# -------------------------------
# 2) 데이터 읽기 함수 정의
# -------------------------------
def get_robot_state():
    """
    로봇의 현재 좌표와 관절 각도를 반환.
    """
    coordinates = arm.get_position(is_radian=False)
    angles = arm.get_servo_angle()
    return coordinates[1], angles[1]

last_valid_value = 0  # 이전 무게 값을 저장하는 변수

def get_scale_value():
    """
    HX711에서 시리얼 데이터를 읽어 가장 최근 값을 반환.
    버퍼가 비어 있으면 이전 값을 유지.
    """
    global last_valid_value  # 이전 값을 저장하는 변수 사용

    if ser.in_waiting > 0:  # 버퍼에 데이터가 있을 때만 읽기
        try:
            data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore').strip()  # 모든 데이터 읽기
            lines = data.splitlines()

            if lines:  # 데이터가 있으면 가장 마지막 값 사용
                latest_value = lines[-1]
                parts = latest_value.split()
                if len(parts) >= 2 and parts[-1] == 'g':
                    last_valid_value = float(parts[-2])  # 새로운 값 저장
                    return last_valid_value  #
        except (UnicodeDecodeError, ValueError):
            pass  # 오류 발생 시 무시

    return last_valid_value  # 새로운 값이 없으면 이전 값 반환



# -------------------------------
# 3) 메인 루프: CSV로 데이터 기록
# -------------------------------
arm.set_mode(2)
arm.set_state(0)


gripper_start_state = 1
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
    time.sleep(4)
    print("이제하세요~~~~")

    try:
        start_time = time.time()
        while True:
            loop_start = time.time()  # 루프 실행 시간 측정

            # 그리퍼 상태 업데이트
            if keyboard.is_pressed('up'):
                arm.set_suction_cup(True)
                is_gripper_open = 1
            elif keyboard.is_pressed('down'):
                arm.set_suction_cup(False)
                is_gripper_open = 0

            # 로봇 상태 읽기
            coordinates, angles = get_robot_state()
            timestamp = round((time.time() - start_time) * 1000, 3)  # 밀리초 단위

            # 무게 데이터 읽기
            scale_value = get_scale_value()
            if scale_value is None:
                scale_value = 0  # 기본값 설정
            
            # 데이터 기록
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

            # 루프 실행 시간 확인
            loop_duration = (time.time() - loop_start) * 1000  # ms 단위
            print(f"[{timestamp} ms] Loop: {loop_duration:.2f} ms, Scale: {scale_value} g")

            # 루프 실행 속도 조절 (5ms 주기)
            time.sleep(max(0, 0.005 - (time.time() - loop_start)))

    except KeyboardInterrupt:
        print("Data recording stopped.")

arm.set_mode(0)
arm.disconnect()
ser.close()
