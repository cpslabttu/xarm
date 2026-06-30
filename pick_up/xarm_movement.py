import xarm

arm = xarm.Controller('USB')

def reset_servos():
    for i in range(6, 0, -1):
        arm.setPosition(i, 500, duration=1000, wait=True)
        
def close_gripper(gripper_strength):
    arm.setPosition(1, gripper_strength, wait=True)

def open_gripper():
    arm.setPosition(1, 0, wait=True)

def move_down(duration):
    arm.setPosition(5, 875, duration, wait=True)
