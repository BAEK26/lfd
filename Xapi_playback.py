#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: Record trajectory
    1. requires firmware 1.2.0 and above support
"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
my_path = "data\interpolated_trajectory.traj"
from xarm.wrapper import XArmAPI

ip = "192.168.1.194"
arm = XArmAPI(ip, is_radian=True)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)


arm.load_trajectory(my_path)
arm.playback_trajectory()