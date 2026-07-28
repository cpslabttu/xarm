import time
import serial
from xarm_movement import arm, close_gripper, open_gripper

SENSOR_PORT = 'COM4'   

DOWN_SERVO = 3
DOWN_START = 830
DOWN_LIMIT = 500        

# 5: base pivot, swings the whole arm forward/down to close the distance
# once something is spotted.
BASE_SERVO = 5
BASE_START = 500
BASE_LIMIT = 850        # guess - how far the base swings in; tune on the arm

# 4: wrist, tilts only the gripper. The sensor sits above the gripper, so
# once we've found something we tilt the wrist down an extra amount to
# bring the gripper into the sensor's line of sight before closing in -
# otherwise the gripper ends up too high and misses. This is a fixed
# offset: the sensor/gripper gap is a rigid mounting distance, not
# something that scales with how far the scan went.
WRIST_SERVO = 4
WRIST_HOME = 500
WRIST_COMPENSATION = 60  

# Distance that means "something is there." Tune against your sensor's
# resting/background reading once you test on the real setup.
DETECT_DISTANCE_CM = 20.0
GRAB_DISTANCE_CM = 5.0

# The sensor trips before the gripper is actually close enough (same
# sensor/gripper offset problem as the wrist compensation), so push the
# base a bit further in before closing. Tune up if it still falls short,
# down if it starts ramming the object.
GRAB_OVERSHOOT = 25

# 0 = open ... 600 = fully closed on this gripper (see xarm_movement.close_gripper).
GRIP_STRENGTH = 600

# Ignore the sensor for this long right after the arm settles at the start
# of descend_until_detected() - otherwise the first reading is often
# whatever built up in the buffer while the arm was still moving (e.g. the
# table under it), which reads as a false detect.
DETECT_START_DELAY_S = 1.5
#....... 

# How long to hold the arm up at the initial height so someone can place the
# next object underneath it, before it starts scanning down.
PLACEMENT_WAIT_S = 5.0

# getPosition() reads the real servo feedback, which won't always land
# exactly on the commanded value - allow a little slack when checking
# whether the arm is already up at the initial height.
HEIGHT_TOLERANCE = 15

# Small-step, non-blocking motion: a new command lands before the previous
# one finishes easing, so the servo never fully stops between ticks. This
# is what removes the jolt.
STEP = 6
STEP_DURATION_MS = 150
TICK_INTERVAL = 0.08

# Servo 3 carries the whole forearm/wrist/gripper, so it shows the jolt
# more than the base does at the same step rate - smaller steps, more
# ticks, smooths it out.
DOWN_STEP = 3
DOWN_STEP_DURATION_MS = 180
DOWN_TICK_INTERVAL = 0.08

RESET_DURATION_MS = 1200   # long + simultaneous so the return home eases

ser = serial.Serial(SENSOR_PORT, 9600, timeout=0.2)
time.sleep(2)          # Uno reboots when the port opens; let it settle


def get_latest_distance():
    """Drain the buffer, return the newest valid reading (or None)."""
    latest = None
    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line.startswith("DIST:") and "INVALID" not in line:
            try:
                latest = float(line.split(":")[1])
            except ValueError:
                pass
    return latest


def already_at_initial_height():
    return abs(arm.getPosition(DOWN_SERVO) - DOWN_START) <= HEIGHT_TOLERANCE


def descend_until_detected():
    """Returns the servo-3 position where something was detected, or None."""
    pos = DOWN_START
    arm.setPosition(DOWN_SERVO, pos, 500, wait=True)

    time.sleep(DETECT_START_DELAY_S)
    get_latest_distance()   # discard whatever built up while we waited

    while pos != DOWN_LIMIT:
        dist = get_latest_distance()
        if dist is not None and dist < DETECT_DISTANCE_CM:
            print(f"Object detected at {dist:.1f} cm")
            return pos

        pos = min(pos + DOWN_STEP, DOWN_LIMIT) if pos < DOWN_LIMIT else max(pos - DOWN_STEP, DOWN_LIMIT)
        arm.setPosition(DOWN_SERVO, pos, DOWN_STEP_DURATION_MS, wait=False)
        time.sleep(DOWN_TICK_INTERVAL)

    print("Reached full tilt-down without detecting anything")
    return None


def lower_to_ground(target):
    """Ease servo 3 back down to target (the height where the object/table
    was actually found), same step cadence as descend_until_detected."""
    pos = DOWN_START
    while pos != target:
        pos = max(pos - DOWN_STEP, target)
        arm.setPosition(DOWN_SERVO, pos, DOWN_STEP_DURATION_MS, wait=False)
        time.sleep(DOWN_TICK_INTERVAL)


def approach_and_grab(ground_pos):
    """Lower the gripper to match the sensor's sightline, then swing the
    base in until close enough to grab. ground_pos is the servo-3 height
    where the object/table was actually found, used when placing it back
    down so we don't overshoot past the real ground."""
    arm.setPosition(WRIST_SERVO, WRIST_HOME + WRIST_COMPENSATION, 400, wait=True)

    pos = BASE_START
    arm.setPosition(BASE_SERVO, pos, 500, wait=True)

    while pos != BASE_LIMIT:
        dist = get_latest_distance()
        print(f"[approach] pos={pos} dist={dist}")   
        if dist is not None and dist < GRAB_DISTANCE_CM:
            print(f"Object at {dist:.1f} cm - grabbing")

            pos = min(pos + GRAB_OVERSHOOT, BASE_LIMIT) if pos < BASE_LIMIT else max(pos - GRAB_OVERSHOOT, BASE_LIMIT)
            arm.setPosition(BASE_SERVO, pos, 300, wait=True)   # close the last bit of the gap

            close_gripper(GRIP_STRENGTH)
            time.sleep(0.3)

            arm.setPosition(DOWN_SERVO, DOWN_START, 800, wait=False)  # raise back up to the initial height
            arm.setPosition(BASE_SERVO, BASE_START, 800, wait=True)   # pick it all the way up
            time.sleep(0.5)

            arm.setPosition(BASE_SERVO, pos, 800, wait=True)          # swing back over the drop spot
            time.sleep(0.3)

            lower_to_ground(ground_pos)   # slowly ease back down instead of dropping from height
            time.sleep(0.3)
            open_gripper()      # release - let it fall the rest of the way
            time.sleep(0.3)
            return True

        pos = min(pos + STEP, BASE_LIMIT) if pos < BASE_LIMIT else max(pos - STEP, BASE_LIMIT)
        arm.setPosition(BASE_SERVO, pos, STEP_DURATION_MS, wait=False)
        time.sleep(TICK_INTERVAL)

    print("Reached full approach without getting within grab range")
    arm.setPosition(BASE_SERVO, BASE_START, 800, wait=True)
    return False


def pick_and_reset():
    open_gripper()   # start every run with the fingers open and ready
    arm.setPosition(2, 430)   # rotate gripper to parallel
    arm.setPosition(6, 600)   # rotate arm to middle

    if already_at_initial_height():
        print("Already at the initial height - object should already be placed")
    else:
        arm.setPosition(DOWN_SERVO, DOWN_START, wait=True)   # the initial raise
        print(f"Waiting {PLACEMENT_WAIT_S:.0f}s to place the object...")
        time.sleep(PLACEMENT_WAIT_S)

    detected_pos = descend_until_detected()
    if detected_pos is not None:
        approach_and_grab(detected_pos)

    # Reset all servos
    arm.setPosition(1, 0, RESET_DURATION_MS, wait=False)
    arm.setPosition(2, 430, RESET_DURATION_MS, wait=False)
    arm.setPosition(3, DOWN_START, RESET_DURATION_MS, wait=False)
    arm.setPosition(4, WRIST_HOME, RESET_DURATION_MS, wait=False)
    arm.setPosition(5, BASE_START, RESET_DURATION_MS, wait=False)
    arm.setPosition(6, 600, RESET_DURATION_MS, wait=False)
    time.sleep(RESET_DURATION_MS / 1000)


if __name__ == "__main__":
    pick_and_reset()
