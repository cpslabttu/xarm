import xarm

arm = xarm.Controller('USB')

def initiate(start_pos):
    arm.setPosition(3, start_pos, wait=True)
    arm.setPosition(2, 430) # Rotate gripper to parallel
    arm.setPosition(6, 600) # Rotate arm to middle
        
def close_gripper(gripper_strength):
    arm.setPosition(1, gripper_strength, wait=True)

def open_gripper():
    arm.setPosition(1, 0, wait=True)

def move_down(duration):
    arm.setPosition(5, 875, duration, wait=True)
