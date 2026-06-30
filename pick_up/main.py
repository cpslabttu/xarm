import xarm
from xarm_movement import reset_servos, move_down, close_gripper, open_gripper

arm = xarm.Controller('USB')

MOVE_DELAY_MS = 2000
ACTION_DELAY_MS = 3000
GRIPPER_STRENGTH = 370

# Orientates Gripper Parallel to Obj
arm.setPosition(4, 100)

open_gripper()

# Extend out
arm.setPosition(3, 655, duration=ACTION_DELAY_MS, wait=True)

move_down(MOVE_DELAY_MS)
close_gripper(GRIPPER_STRENGTH)

# Picks up
arm.setPosition(5, 610,duration=MOVE_DELAY_MS, wait=True)

move_down(MOVE_DELAY_MS)
open_gripper()

reset_servos()