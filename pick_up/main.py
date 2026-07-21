import time
import serial
from xarm_movement import arm, initiate, close_gripper, open_gripper

SENSOR_PORT = 'COM4'   # your Arduino port (COMx on Windows)

DOWN_SERVO = 5
DOWN_START = 500
DOWN_LIMIT = 875

TILT_SERVO = 4
TILT_START = 500      # servo 4's position at the top of the descent
TILT_END = 200        # servo 4's position at full-down (gripper pointing down)

STEP = 15
GRAB_DISTANCE_CM = 5.0

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


def descend_until_close():
    pos = DOWN_START
    arm.setPosition(DOWN_SERVO, pos, 500, wait=True)
    arm.setPosition(TILT_SERVO, TILT_START, 500, wait=True)

    while pos < DOWN_LIMIT:
        dist = get_latest_distance()

        if dist is not None and dist < GRAB_DISTANCE_CM:
            print(f"Object at {dist:.1f} cm - stopping descent")
            return True

        pos = min(pos + STEP, DOWN_LIMIT)

        # Servo 4 tracks servo 5's progress proportionally (0.0 -> 1.0)
        progress = (pos - DOWN_START) / (DOWN_LIMIT - DOWN_START)
        tilt = TILT_START + progress * (TILT_END - TILT_START)

        print(f"pos={pos}  tilt={int(round(tilt))}")
        arm.setPosition(TILT_SERVO, int(round(tilt)), 200, wait=False)
        arm.setPosition(DOWN_SERVO, pos, 200, wait=True)
        time.sleep(0.08)

    print("Reached full-down without object inside 5 cm")
    return False


def pick_and_reset():
    open_gripper()
    initiate(start_pos=830)          # your parallel-gripper setup pose
    time.sleep(0.5)

    if descend_until_close():
        close_gripper(GRIP_STRENGTH)
        time.sleep(0.3)
        arm.setPosition(DOWN_SERVO, DOWN_START, 800, wait=True)   # lift
        time.sleep(0.5)

    # reset everything to home
    open_gripper()
    arm.setPosition(2, 500, 500, wait=True)
    arm.setPosition(3, 500, 500, wait=True)
    arm.setPosition(6, 500, 500, wait=True)
    arm.setPosition(DOWN_SERVO, DOWN_START, 800, wait=True)


if __name__ == "__main__":
    pick_and_reset()