import time
import xarm


GRIP_SERVO_ID = 1
ROTATE_SERVO_ID = 2
TILT_SERVO_ID = 3

GRIP_OPEN = 0
GRIP_CLOSED = 500

ROTATE_MIN = 0
ROTATE_MAX = 1000

TILT_MIN = 0
TILT_MAX = 1000

DURATION_MS = 60
COMMAND_INTERVAL = 0.06
MIN_DELTA = 8


arm = xarm.Controller("USB")

last_sent = {
    GRIP_SERVO_ID: 0.0,
    ROTATE_SERVO_ID: 0.0,
    TILT_SERVO_ID: 0.0,
}

last_value = {
    GRIP_SERVO_ID: None,
    ROTATE_SERVO_ID: None,
    TILT_SERVO_ID: None,
}


def clamp(value, low, high):
    return max(low, min(high, int(round(value))))


def move_servo(servo_id, value, low, high):
    value = clamp(value, low, high)
    now = time.time()

    previous_value = last_value[servo_id]
    if previous_value is not None and abs(value - previous_value) < MIN_DELTA:
        return False

    if now - last_sent[servo_id] < COMMAND_INTERVAL:
        return False

    arm.setPosition(
        servo_id,
        value,
        duration=DURATION_MS,
        wait=False
    )

    last_value[servo_id] = value
    last_sent[servo_id] = now
    return True


def move_gripper(value):
    return move_servo(
        GRIP_SERVO_ID,
        value,
        GRIP_OPEN,
        GRIP_CLOSED
    )

def rotate_gripper(value):
    return move_servo(
        ROTATE_SERVO_ID,
        value,
        ROTATE_MIN,
        ROTATE_MAX
    )

def tilt_gripper(value):
    return move_servo(
        TILT_SERVO_ID,
        value,
        TILT_MIN,
        TILT_MAX
    )