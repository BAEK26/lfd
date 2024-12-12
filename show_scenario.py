
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


# Turn on manual mode before recording
arm.set_mode(2)
arm.set_state(0)
#arm.set_suction_cup(True) #열림 open gripper
#time.sleep(3)
arm.set_suction_cup(False) #닫힘 close gripper

# Open a CSV file to save the data
for i in range(1):
    arm.set_mode(2)
    datafile_path = os.path.join('data', f'neo_show_scenario.csv')
    with open(datafile_path, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        try:
            start_time = time.time()
            while True:
                # Get current robot state
                coordinates, angles = get_robot_state()
                
                # Record the timestamp
                timestamp = round((time.time() - start_time)*1000, 0)
                
                # Write the data to the CSV file
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
                
                # Wait for a short period before polling again
                time.sleep(0.01)  # Adjust the interval as needed

        except KeyboardInterrupt:
            # Stop recording on user interrupt
            print(f"{i} Data recording stopped.")

arm.set_mode(0)
arm.disconnect()
