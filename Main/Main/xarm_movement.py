import time
import xarm


# Servo IDs on the xArm (1 = gripper, ... , 6 = base).
SERVO_GRIP = 1
SERVO_ROTATE = 2
SERVO_PITCH = 3
SERVO_ELBOW = 4
SERVO_SHOULDER = 5
SERVO_BASE = 6

ALL_SERVOS = [
    SERVO_GRIP,
    SERVO_ROTATE,
    SERVO_PITCH,
    SERVO_ELBOW,
    SERVO_SHOULDER,
    SERVO_BASE,
]

# Per-servo travel limits, as (value_at_norm_0, value_at_norm_1).
# Wrist/gripper servos use their full range; the big arm joints are
# kept conservative so a gesture can't slam the arm into a hard stop.
# TEST WITH A CLEAR WORKSPACE and widen/flip these once the direction
# and safe range are confirmed on the real arm.
SERVO_RANGES = {
    SERVO_GRIP: (0, 500),        # 0 = open, 500 = closed
    SERVO_ROTATE: (0, 1000),     # 0.5 norm = centered
    SERVO_PITCH: (200, 800),
    SERVO_ELBOW: (250, 750),
    SERVO_SHOULDER: (250, 750),
    SERVO_BASE: (0, 1000),       # 0.5 norm = centered
}

DURATION_MS = 60
COMMAND_INTERVAL = 0.06
MIN_DELTA = 8


arm = xarm.Controller("USB")

last_sent = {servo_id: 0.0 for servo_id in ALL_SERVOS}
last_value = {servo_id: None for servo_id in ALL_SERVOS}


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


def set_servo(servo_id, norm):
    """Drive a servo from a normalized 0..1 gesture value."""
    low, high = SERVO_RANGES[servo_id]
    value = low + norm * (high - low)
    return move_servo(servo_id, value, min(low, high), max(low, high))


def servo_units(servo_id, norm):
    """The raw servo value a given norm maps to (for on-screen display)."""
    low, high = SERVO_RANGES[servo_id]
    return clamp(low + norm * (high - low), min(low, high), max(low, high))
