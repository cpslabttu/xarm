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

COMMAND_INTERVAL = 0.06
MIN_DELTA = 3

# Max servo units a joint may move per command (slew-rate limit). Small
# values = slow/smooth; the heavy lower joints are kept slow so gesture
# jitter and mode switches can't fling the arm. Tune per joint.
SERVO_MAX_STEP = {
    SERVO_GRIP: 40,
    SERVO_ROTATE: 30,
    SERVO_PITCH: 18,
    SERVO_ELBOW: 8,
    SERVO_SHOULDER: 8,
    SERVO_BASE: 10,
}

# How long each servo takes to reach a commanded step. Longer = smoother
# physical motion; the lower joints get more easing.
SERVO_DURATION = {
    SERVO_GRIP: 60,
    SERVO_ROTATE: 70,
    SERVO_PITCH: 120,
    SERVO_ELBOW: 200,
    SERVO_SHOULDER: 200,
    SERVO_BASE: 180,
}


arm = xarm.Controller("USB")

last_sent = {servo_id: 0.0 for servo_id in ALL_SERVOS}
last_value = {servo_id: None for servo_id in ALL_SERVOS}

# Slew-limited target the servo is currently easing toward.
servo_target = {servo_id: None for servo_id in ALL_SERVOS}


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
        duration=SERVO_DURATION[servo_id],
        wait=False
    )

    last_value[servo_id] = value
    last_sent[servo_id] = now
    return True


def set_servo(servo_id, norm):
    """Ease a servo toward a normalized 0..1 gesture value, slew-limited."""
    low, high = SERVO_RANGES[servo_id]
    desired = low + norm * (high - low)

    target = servo_target[servo_id]
    if target is None:
        # First command: snap to avoid slewing from an arbitrary origin.
        target = desired
    else:
        step = SERVO_MAX_STEP[servo_id]
        delta = max(-step, min(step, desired - target))
        target = target + delta

    servo_target[servo_id] = target
    return move_servo(servo_id, target, min(low, high), max(low, high))


def servo_units(servo_id, norm):
    """The raw servo value a given norm maps to (for on-screen display)."""
    low, high = SERVO_RANGES[servo_id]
    return clamp(low + norm * (high - low), min(low, high), max(low, high))
