#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2022, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
# Notice
#   1. Changes to this file on Studio will not be preserved
#   2. The next conversion will overwrite the file with the same name
#
# xArm-Python-SDK: https://github.com/xArm-Developer/xArm-Python-SDK
#   1. git clone git@github.com:xArm-Developer/xArm-Python-SDK.git
#   2. cd xArm-Python-SDK
#   3. python setup.py install
"""
import sys
import math
import time
import queue
import datetime
import random
import traceback
import threading
from xarm import version
from xarm.wrapper import XArmAPI
import csv


class RobotMain(object):
    """Robot Main Class"""
    def __init__(self, robot, **kwargs):
        self.alive = True
        self._arm = robot
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 20
        self._angle_acc = 500
        self._vars = {'x': 0, 'y': 0, 'z': 0, 'count': 0, 'repeat': 0, 'bottle_from':206, 'bottle_to':206, 'pour_angle' : 115}
        self._funcs = {}
        self._robot_init()

    # Robot init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'register_count_changed_callback'):
            self._arm.register_count_changed_callback(self._count_changed_callback)

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint('err={}, quit'.format(data['error_code']))
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    # Register state changed callback
    def _state_changed_callback(self, data):
        if data and data['state'] == 4:
            self.alive = False
            self.pprint('state=4, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)

    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint('counter val: {}'.format(data['count']))

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(label, code, self._arm.connected, self._arm.state, self._arm.error_code, ret1, ret2))
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1], ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)

    @property
    def arm(self):
        return self._arm

    @property
    def VARS(self):
        return self._vars

    @property
    def FUNCS(self):
        return self._funcs

    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False

    def _initial_position(self):
        # initial position
        print("initial_position")
        code, pos = self._arm.get_position(is_radian=True)
        #print(f"code is {code}")
        #print(f"pos is {pos}")
        if pos[0] > 140 and pos[1] < -180 and pos[2] > 170:
            pass
        elif pos[0] > 350 and pos[2] > 470 or pos[2]>510:
            pass
        elif pos[1] > 175 or (pos[1] < -130  and pos[0] != 110 and pos[1] != -146.300003)or pos[2] > 450:
            coordinates = self._get_coordinates(0)
            pos[0] = 110
            pos[3] = coordinates[3]
            pos[4] = coordinates[4]
            pos[5] = coordinates[5]
            code = self._arm.set_position(*pos, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            pos[2] = 180
            code = self._arm.set_position(*pos, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            if pos[1] > 175:
                code = self._arm.set_servo_angle(angle=[59.3, 51.2, 45.7, 62.8, -89.5, -7.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'):
                    return
            if pos[1] < -130:
                code = self._arm.set_servo_angle(angle=[-48.4, 43.6, 40.1, -49.9, -93.1, -44.3], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'):
                    return
        code = self._arm.set_servo_angle(angle=[1.8, -6.8, 7.3, 0.7, -71.1, 0.0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        time.sleep(1)
        # code = self._arm.open_lite6_gripper()
        time.sleep(1)

    def _get_coordinates(self, bottle):
        x= 257.0
        y= 167.0
        z= 310.0
        p= -179.3
        q= -83.9
        r= -0.4 #Benchmark(1st floor 6th bottle)
        data = [
            [0,x,y,z,p,q,r],
            [102,x-74.5,y+227.9,z,p,q,r],
            [103,x-52,y+172.2,z,p,q,r],
            [104,x-30.5,y+116.2,z,p,q,r],
            [105,x-13.59,y+58.7,z,p,q,r],
            [106,x,y,z,p,q,r],
            [107,x+2,y-59.4,z,p,q,r],
            [108,x+6,y-119.2,z,p,q,r],
            [109,x+6,y-179.2,z,p,q,r],
            [110,x+6,y-239.1,z,p,q,r],
            [111,x+2,y-298.6,z,p,q,r],
            [112,x-12.1,y-357.9,z+2,p,q,r],
            [113,x-27.9,y-415.7,z+2,p,q,r],
            [114,x-48.3,y-472.1,z+2,p,q,r],
            [115,x-73.7,y-526.4,z+2,p,q,r],
            [204,x+7.5,y+157.1,z+97,p,q,r],
            [205,x+26.4,y+100.2,z+97,p,q,r],
            [206,x+41,y+42,z+97,p,q,r],
            [207,x+51.6,y-17,z+97,p,q,r],
            [208,x+58.4,y-76.7,z+97,p,q,r],
            [209,x+61.4,y-135.0,z+97,p,q,r],
            [210,x+60.7,y-196.6,z+97,p,q,r],
            [211,x+56.3,y-256.4,z+97,p,q,r],
            [212,x+48.1,y-315.8,z+97,p,q,r],
            [213,x+36,y-371.0,z+97,p,q,r],
            [214,x+19.9,y-432.3,z+97,p,q,r],
            [215,x-0.6,y-488.7,z+97,p,q,r],
            [309,x+108,y-37.5,z+202,p,q,r],
            [310,x+111,y-97.5,z+202,p,q,r],
            [311,x+113,y-160.2,z+202,p,q,r],
            [312,x+108,y-217.7,z+202,p,q,r]
            ]

        position = []
        for i in range(len(data)):
            if(bottle==data[i][0]):
                for j in range(1,7):
                    value = data[i][j]
                    position.append(value)
        return position

    # Robot Main Run
    def initialSetting(self):
        # setting speed
        print("initial setting")
        self._tcp_speed = 150
        self._tcp_acc = 1800
        self._angle_speed = 30
        self._angle_acc = 225
        self._vars['x'] = None
        self._vars['y'] = 0
        self._vars['z'] = None
        self._vars['count'] = 0
        self._vars['repeat'] = 0

        #set robot arm to initial position
        self._initial_position()
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        #put the number of bottles you want to reach
        #bottle_from = 206
        #bottle_to = 206
        self._vars['bottle_from'] = 206
        self._vars['bottle_to'] = 206

        #put the degree of angle you want to pour coffee beans
        #pour_angle = 115
        self._vars['pour_angle'] = 115
        self._vars['waiting_time'] = 5
        self.waiting_time = 5

    def processLid(self, i):
        print("PROCESS LID")

        coordinates = self._get_coordinates(i)
        coordinates = self.set_coordinates(i, coordinates)

        self.grabLid(i, coordinates)
        coordinates = self.liftLid(i)
        self.putDownLid(i, coordinates)

        return coordinates

    def set_coordinates(self,i, coordinates):
        print("-----set_coordinates")
        if 100<i<200:
            coordinates[0] += -80
        elif 200<i<300:
            coordinates[0] += -120
        elif 300<i<400:
            coordinates[0] += -200
        return coordinates

    def reset_coordinates(self,i, coordinates):
        print("-----reset_coordinates")
        if 100<i<200:
            coordinates[0] += 80
        elif 200<i<300:
            coordinates[0] += 120
        elif 300<i<400:
            coordinates[0] += 200
        return coordinates

    def grabLid(self, i, coordinates, dummy=0):
        print("-----grabLid")
        coordinates[2] += dummy
        self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        coordinates = self._get_coordinates(i)
        coordinates[2] += dummy
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(1)

    def liftLid(self,i, dummy=0):
        print("-----liftLid")
        coordinates = self._get_coordinates(i)
        coordinates[2] += 15
        coordinates[2] += dummy
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        return coordinates

    def putDownLid(self,i,coordinates, dummy=0):
        print("-----put down lid")
        coordinates = self.set_coordinates(i, coordinates)
        coordinates[2] += dummy
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return

        print("----------1: side")
        if i > 300:
            code = self._arm.set_servo_angle(angle=[56.2, -0.6, 37, 65.9, -64.4, -34.1], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
        else:
            coordinates = self._get_coordinates(0)
            coordinates[0] += -80
            code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'):
                return

        # exit()
        print("----------2: down")
        if dummy != 0:
            code = self._arm.set_servo_angle(angle=[82.1, 67.8, 56.7, 81.6, -86.7, 12.4], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        else:
            code = self._arm.set_servo_angle(angle=[84.6, 76.1, 81.4, 84.7, -87.9, -7.4], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        time.sleep(2)
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(4)

        print("----------3: slightly up")
        if dummy != 0:
            coordinates = [95, 222.9, 115.3,-162.3, -85.3, -17.6]
        else:
            coordinates = [92, 299.6, 112.7,120.7, -87.7, 59.8]
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return

    def processCup(self,i):
        print("PROCESS CUP")
        print("set coord")
        coordinates = self._get_coordinates(i)
        coordinates = self.set_coordinates(i, coordinates)
        print("process cup")
        coordinates = self.grabBottle(i, coordinates)
        coordinates = self.liftBottle(i, coordinates)
        coordinates = self.pourBeans(i, coordinates)
        #self._initial_position() #subsitute >> self.goBackBottle() go back to bottle postion
        coordinates = self.putDownBottle(i, coordinates)

        return coordinates

    def grabBottle(self, i, coordinates):
        print("-----grabBottle")
        self._initial_position()
        coordinates = self._get_coordinates(i)
        coordinates = self.set_coordinates(i,coordinates)
        coordinates[2] += -40

        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return

        print("----------grab ")
        coordinates = self._get_coordinates(i)
        coordinates[2] += -40
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(1)

        return coordinates

    def liftBottle(self,i,coordinates):
        print("-----liftBottle")
        coordinates = self._get_coordinates(i)
        coordinates[2] += -10
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)

        return coordinates

    def pourBeans(self, i, coordinates):
        print("-----pourBeans")
        coordinates = self.set_coordinates(i, coordinates)

        print(" ")
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return

        print("----------get and set coord 0 ")
        coordinates = self._get_coordinates(0)
        coordinates[0] += -80
        coordinates[1] += -170
        coordinates[2] += -100
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return

        self.pouring()

        return coordinates

    def pouring(self):
        print("----------pouring ")
        code = self._arm.set_servo_angle(servo_id=1, angle=90, speed=self._angle_speed, mvacc=self._angle_acc, relative=True, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return

        code = self._arm.set_servo_angle(angle=[118.7, 40.8, 44.3, 6.8, -81.2, 0], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return

        print("servoid 6 , angle pourangle")
        code = self._arm.set_servo_angle(servo_id=6, angle=-self._vars['pour_angle'], speed=self._angle_speed/2, mvacc=self._angle_acc, relative=True, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        time.sleep(self.waiting_time)
        code = self._arm.set_servo_angle(servo_id=6, angle=self._vars['pour_angle'], speed=self._angle_speed, mvacc=self._angle_acc, relative=True, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return


        code = self._arm.set_servo_angle(servo_id=1, angle=-60, speed=self._angle_speed, mvacc=self._angle_acc, relative=True, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return

    def putDownBottle(self, i, coordinates):
        print("-----putDownBottle")
        coordinates = self._get_coordinates(i)
        coordinates = self.set_coordinates(i, coordinates)
        print("----------putDownBottle 1")
        coordinates[2] = self._get_coordinates(0)[2]-100
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return

        print("----------putDownBottle 2")
        coordinates[2] = self._get_coordinates(i)[2]-10
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return

        print("----------putDownBottle 3")
        coordinates = self.reset_coordinates(i, coordinates)

        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return

        time.sleep(1)
        print("----------putDownBottle 4")
        coordinates = self._get_coordinates(i)
        coordinates[2] += -40
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)

        print("----------putDownBottle last : open gripper")
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(1)

        return coordinates

    def grabLidBack(self, i, coordinates, dummy=0):
        print("-----grabLidBack")
        if dummy == 0:
            coordinates = self.set_coordinates(i,coordinates)
            code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'):
                return
            if i in (114, 115):
                coordinates = self._get_coordinates(0)
                coordinates[0] += -80
                coordinates[1] += -170
                coordinates[2] += -100
                code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
                if not self._check_code(code, 'set_position'):
                    return
            self._initial_position()
        print("----------1: side")
        coordinates = self._get_coordinates(0)
        coordinates[0] += -80
        # coordinates[2] += -218
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return

        angle=[]
        if dummy != 0:
            angle = [82.1, 67.8, 56.7, 81.6, -86.7, 12.4]
        else:
            angle = [84.6, 76.1, 81.4, 84.7, -87.9, -7.4]
        print("----------2: down")
        code = self._arm.set_servo_angle(angle=angle, speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        time.sleep(2)
        print("----------lid back gripper")
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(2)

        print("----------3: slightly up")
        if dummy != 0:
            coordinates = [95, 222.9, 115.3,-162.3, -85.3, -17.6]
        else:
            coordinates = [92, 299.6, 112.7,120.7, -87.7, 59.8]
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return

        return coordinates

    def closeLid(self, i, dummy=0):
        coordinates = self._get_coordinates(i)
        coordinates = self.set_coordinates(i, coordinates)
        coordinates[2] += 15
        coordinates[2] += dummy

        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return

        print("----------1: side")
        coordinates = self.reset_coordinates(i, coordinates)

        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return

        print("----------2: down")
        coordinates[2] += -15
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(2)

        print("----------let lid go gripper")
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(1)

        # back to ready position
        coordinates = self._get_coordinates(i)
        coordinates[2] += dummy
        coordinates = self.set_coordinates(i, coordinates)

        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return

        return coordinates

    def processLidBack(self, i, coordinates):
        print("PROCESS Lid Back")
        coordinates = self.grabLidBack(i,coordinates)
        self._initial_position()
        coordinates = self.closeLid(i)

    def processDummyBottle(self, i=102):
        print("PROCESS DummyBottle")
        coordinates = self.grabDummyBottle(i)
        coordinates = self.liftDummyBottle(i, coordinates)
        coordinates = self.setDummyBottle(i, coordinates)
        self._initial_position()

    def processDummyBottleBack(self, i=102):
        print("PROCESS DummyBottle BACK")
        coordinates = self.grabDummyBottleBack()
        coordinates = self.putDownBottle(i, coordinates)
        self._initial_position()

        return coordinates


    def grabDummyBottle(self, i):
        i = 102
        coordinates = self._get_coordinates(i)
        coordinates[0] += -80
        coordinates[2] += -40

        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        coordinates = self._get_coordinates(i)
        coordinates[2] += -40
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(1)
        return coordinates

    def liftDummyBottle(self, i, coordinates):
        coordinates = self._get_coordinates(i)
        coordinates[2] += -10
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        return coordinates

    def setDummyBottle(self, i, coordinates):
        coordinates = self.set_coordinates(i, coordinates)
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return

        self._initial_position()
        # # TODO 삭제 후 pouringbeans 와 유사하게 동작하도록.
        # coordinates[0] += -80
        # coordinates[1] += -170
        # coordinates[2] += -100
        code = self._arm.set_servo_angle(servo_id=1, angle=45, speed=self._angle_speed, mvacc=self._angle_acc, relative=True, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[123.1, 47.2, 45.9, 6.4, -90.9, 2.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        time.sleep(1)
        print("----down")

        coordinates = [-149.8, 244.7, 75.6, 143.3, -85.3, 154.4] # 저울 좌표
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_servo_angle(angle=[123.1, 68, 57.4, 6, -97, -1.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        time.sleep(2)
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(2)
        print("----up")

        coordinates = [-149.8, 244.7, 200, 143.3, -85.3, 154.4] # 저울 위의 좌표
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_servo_angle(servo_id=1, angle=-45, speed=self._angle_speed, mvacc=self._angle_acc, relative=True, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        return coordinates

    def grabDummyBottleBack(self):
        print("grab_dummy_bottle_back")
        self._initial_position()
        print("-----turning 45")
        # # TODO 삭제 후 pouringbeans 와 유사하게 동작하도록.
        # coordinates[0] += -80
        # coordinates[1] += -170
        # coordinates[2] += -100
        code = self._arm.set_servo_angle(servo_id=1, angle=45, speed=self._angle_speed, mvacc=self._angle_acc, relative=True, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        coordinates = [-149.8, 244.7, 200, 143.3, -85.3, 154.4] # 저울 위의 좌표
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        time.sleep(1)
        print("----down")

        coordinates = [-149.8, 244.7, 75.6, 143.3, -85.3, 154.4] # 저울 좌표
        code = self._arm.set_position(*coordinates, speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=False)
        if not self._check_code(code, 'set_position'):
            return
        code = self._arm.set_servo_angle(angle=[123.1, 68, 57.4, 6, -97, -1.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        time.sleep(2)
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(2)
        print("----up")

        code = self._arm.set_servo_angle(angle=[123.1, 47.2, 45.9, 6.4, -90.9, 2.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(servo_id=1, angle=-45, speed=self._angle_speed, mvacc=self._angle_acc, relative=True, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        self._initial_position()

        return coordinates

    def processDummyLidOpen(self, i=102):
        # grab + 40
        coordinates = self._get_coordinates(i)
        coordinates = self.set_coordinates(i, coordinates)
        self.grabLid(i, coordinates, dummy=40)
        #Lift + 40
        coordinates = self.liftLid(i,dummy=40)
        self.putDownLid(i, coordinates, dummy=40)
        self._initial_position()

    def processDummyLidBack(self, i, coordinates):
        self.grabLidBack(i, coordinates, dummy=40)
        # safe zone
        self._initial_position()
        self.closeLid(i, dummy=40)

    def run(self):
        try:
            # 1 initialSetting()
            self.initialSetting()
            # 2 ProcessDummyLidOpen
            # Robot opens dummybottle lid
                #self.processDummyLidOpen(102) #얘는 주석 문제 없음
            # 2.1 grabLid
            # 2.2 liftLid
            # 2.3 putdownLid

            # 3 ProcessDummyBottle
            # Robot takes dummy bottle from the designated position and put it on the scale.
            self.processDummyBottle(102)
            # 3.1 grabDummyBottle
            # 3.2 liftDummyBottle
            # 3.3 setDummyBottle

            prev_i = 0
            for i in [106]:# 213, 309]:#106, 213, 309]:#105, 210, 302 [106,213,309]:
                print(f"---PROCESS--{i}--CUP---")
                if prev_i != 0 and prev_i // 100 != i // 100 or (prev_i // 100 == i // 100 and abs(prev_i-i) >6):
                    self._initial_position()
                   
                # 4 ProcessLid
                # robot opens the lid and puts it on the table
                    #coordinates = self.processLid(i)
                # 4.1 grabLid
                # 4.2 liftLid
                # 4.3 putdownlid

                # 5 processCup
                # Robot grabs the bottle, pours beans in the cup, and puts the bottle back afterwards.
                coordinates = self.processCup(i)
                # 5.1 grab_bottle
                # 5.2 lift_bottle
                # 5.3 pour_beans
                # 5.4 need to go back to bottle position safely...
                # 5.5 put_down_bottle
                x, y, z, r, p, w = coordinates
                self._arm.set_position(
                    x - 10, y, z, r, p, w ,
                    speed=self._tcp_speed,
                    mvacc=self._tcp_acc,
                    radius=0.0,
                    wait=True
                )
                self._initial_position() 

                # 6 ProcessLidBack
                # robot puts the lid back
                #coordinates = self.processLidBack(i, coordinates)
                # 6.1 grab_lid
                # 6.2 close_lid
                prev_i = i


            # back to initial position
            #self._initial_position()

            # 7 ProcessDummyBottleBack
            # Robot grabs the dummy bottle and puts back on the original possition.
            self.processDummyBottleBack()
            # 7.1 grabDummyBottleBack
            # 7.2 putDownBottle

            # 8 Process dummy lid back
        #self.processDummyLidBack(102, None)
            # 8.1 grabLidBack
            # 8.2 closeLid

            # final safe zone
            self._arm.set_position(
                    x - 10, y, z, r, p, w ,
                    speed=self._tcp_speed,
                    mvacc=self._tcp_acc,
                    radius=0.0,
                    wait=True
                )
            self._initial_position()

        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        self.alive = False
        self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.release_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'release_count_changed_callback'):
            self._arm.release_count_changed_callback(self._count_changed_callback)


if __name__ == '__main__':
    RobotMain.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    arm = XArmAPI('192.168.1.194', baud_checkset=False)
    robot_main = RobotMain(arm)
    robot_main.run()
