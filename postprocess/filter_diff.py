
#종은아 박스카필터 말고 percentile 필터 어떤가. 쫌 간단한거임 월요일 미팅때 한번 보고드려보겠니
# csv파일에서 데이터 읽고 핵심 좌표를 추출하는건데
#일단 상위 5% 변화량 적용시켜보는거임

import pandas as pd
import numpy as np
from time import time

# # CSV 파일 읽기
# file_path = 'scenarios/one_scenario.csv'
# df = pd.read_csv(file_path)

# # 변화량 계산
# diff = np.linalg.norm(np.diff(df[['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']], axis=0), axis=1)

# # 변화량이 임계값 이상인 지점의 인덱스를 찾음
# threshold = np.percentile(diff, 95)  # 상위 5% 변화량을 기준으로 필터링
# key_indices = np.where(diff > threshold)[0]

# # 핵심 좌표 선택
# key_points = df.iloc[key_indices]
# key_points = key_points.append(df.iloc[-1])  # 마지막 좌표 추가
# key_points

"""
이거 밑에처럼 넣으면 됨. run_scenarioFile.py있잖아
"""

import pandas as pd
import numpy as np
# from xarm.wrapper import XArmAPI

# class RobotMain(object):
#     def __init__(self, robot, key_points, **kwargs):
#         self.alive = True
#         self._arm = robot
#         self._key_points = key_points
#         self._tcp_speed = 100
#         self._tcp_acc = 2000
#         self._angle_speed = 20
#         self._angle_acc = 500
#         self._vars = {}
#         self._funcs = {}
#         self._robot_init()

#     def _robot_init(self):
#         self._arm.clean_warn()
#         self._arm.clean_error()
#         self._arm.motion_enable(True)
#         self._arm.set_mode(0)
#         self._arm.set_state(0)
#         time.sleep(1)
#         self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
#         self._arm.register_state_changed_callback(self._state_changed_callback)
#         if hasattr(self._arm, 'register_count_changed_callback'):
#             self._arm.register_count_changed_callback(self._count_changed_callback)

#     def _error_warn_changed_callback(self, data):
#         if data and data['error_code'] != 0:
#             self.alive = False
#             self.pprint('err={}, quit'.format(data['error_code']))
#             self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

#     def _state_changed_callback(self, data):
#         if data and data['state'] == 4:
#             self.alive = False
#             self.pprint('state=4, quit')
#             self._arm.release_state_changed_callback(self._state_changed_callback)

#     def _count_changed_callback(self, data):
#         if self.is_alive:
#             self.pprint('counter val: {}'.format(data['count']))

#     def _check_code(self, code, label):
#         if not self.is_alive or code != 0:
#             self.alive = False
#             ret1 = self._arm.get_state()
#             ret2 = self._arm.get_err_warn_code()
#             self.pprint('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(label, code, self._arm.connected, self._arm.state, self._arm.error_code, ret1, ret2))
#         return self.is_alive

#     @staticmethod
#     def pprint(*args, **kwargs):
#         try:
#             stack_tuple = traceback.extract_stack(limit=2)[0]
#             print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1], ' '.join(map(str, args))))
#         except:
#             print(*args, **kwargs)

#     @property
#     def arm(self):
#         return self._arm

#     @property
#     def VARS(self):
#         return self._vars

#     @property
#     def FUNCS(self):
#         return self._funcs

#     @property
#     def is_alive(self):
#         if self.alive and self._arm.connected and self._arm.error_code == 0:
#             if self._arm.state == 5:
#                 cnt = 0
#                 while self._arm.state == 5 and cnt < 5:
#                     cnt += 1
#                     time.sleep(0.1)
#             return self._arm.state < 4
#         else:
#             return False

#     def run(self):
#         try:
#             self._tcp_speed = 33
#             self._tcp_acc = 2000
#             self._angle_speed = 12
#             self._angle_acc = 200
#             for index, row in self._key_points.iterrows():
#                 angles = row[['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']].tolist()
#                 code = self._arm.set_servo_angle(angle=angles, speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=50.0)
#                 if not self._check_code(code, 'set_servo_angle'):
#                     return
#             code = self._arm.set_servo_angle(angle=angles, speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
#             if not self._check_code(code, 'set_servo_angle'):
#                 return
#         except Exception as e:
#             self.pprint('MainException: {}'.format(e))
#         self.alive = False
#         self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
#         self._arm.release_state_changed_callback(self._state_changed_callback)
#         if hasattr(self._arm, 'release_count_changed_callback'):
#             self._arm.release_count_changed_callback(self._count_changed_callback)

#여기서부터 진짜 핵심!

if __name__ == '__main__':
    # RobotMain.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    # arm = XArmAPI('192.168.1.194', baud_checkset=False)
    
    # CSV 파일에서 핵심 좌표 추출
    file_path = 'scenarios/one_scenario.csv'
    df = pd.read_csv(file_path)
    print(df)
    diff = np.linalg.norm(np.diff(df[['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']], axis=0), axis=1)
    threshold = np.percentile(diff, 95)
    key_indices = np.where(diff > threshold)[0]
    key_points = df.iloc[key_indices]
    # key_points = key_points.append(df.iloc[-1])
    print(key_points)
    # robot_main = RobotMain(arm, key_points)
    # robot_main.run()
