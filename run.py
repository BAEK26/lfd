
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: Interface for obtaining information
"""

import os
import csv
import sys
import time
import params

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI


#######################################################
"""
Just for test example
"""
# if len(sys.argv) >= 2:
#     ip = "192.168.1.194" #sys.argv[1] #
# else:
#     try:
#         from configparser import ConfigParser
#         parser = ConfigParser()
#         parser.read('../robot.conf')
#         ip = parser.get('xArm', 'ip')
#     except:
#         ip = input('Please input the xArm ip address:')
#         if not ip:
#             print('input error, exit')
#             sys.exit(1)
########################################################

ip = "192.168.1.194"
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

print('=' * 50)
print('version:', arm.get_version())
print('state:', arm.get_state())
print('cmdnum:', arm.get_cmdnum())
print('err_warn_code:', arm.get_err_warn_code())
print('position(°):', arm.get_position(is_radian=False))
print('position(radian):', arm.get_position(is_radian=True))
print('angles(°):', arm.get_servo_angle(is_radian=False))
print('angles(radian):', arm.get_servo_angle(is_radian=True))
print('angles(°)(servo_id=1):', arm.get_servo_angle(servo_id=1, is_radian=False))
print('angles(radian)(servo_id=1):', arm.get_servo_angle(servo_id=1, is_radian=True))

# Function to get current coordinates and angles
def get_robot_state():
    coordinates = arm.get_position(is_radian=False)  # Replace with actual method to get coordinates
    angles = arm.get_servo_angle()      # Replace with actual method to get joint angles
    return coordinates[1], angles[1]
coordinates, angles = get_robot_state()
# Open a CSV file to save the data
arm.set_servo_angle(angle=angles, speed=params.angle_speed, mvacc=params.angle_acc, wait=False, radius=0.0)
print(arm.get_position(), arm.get_position(is_radian=False))
angles[0] -= 1
arm.set_servo_angle(angle=angles, speed=params.angle_speed, mvacc=params.angle_acc, wait=False, radius=0.0)
print(arm.get_position(), arm.get_position(is_radian=False))
angles[0] -= 1
arm.set_servo_angle(angle=angles, speed=params.angle_speed, mvacc=params.angle_acc, wait=False, radius=0.0)
print(arm.get_position(), arm.get_position(is_radian=False))
angles[0] -= 1
arm.set_servo_angle(angle=angles, speed=params.angle_speed, mvacc=params.angle_acc, wait=False, radius=0.0)
angles[0] -= 1
arm.set_servo_angle(angle=angles, speed=params.angle_speed, mvacc=params.angle_acc, wait=False, radius=0.0)
angles[0] -= 1
arm.set_servo_angle(angle=angles, speed=params.angle_speed, mvacc=params.angle_acc, wait=False, radius=0.0)
angles[0] -= 1
arm.set_servo_angle(angle=angles, speed=params.angle_speed, mvacc=params.angle_acc, wait=False, radius=0.0)

arm.disconnect()
