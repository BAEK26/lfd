from xarm.wrapper import XArmAPI
xarm = XArmAPI('192.168.1.194')  
print(xarm.arm.arm.joint_limit_min, xarm.arm.arm.joint_limit_max)