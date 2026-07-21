"""Spatial hand control for the xArm.

A more intuitive scheme than the per-finger-gesture version in Main.py:
you move your *positioning hand* through space and the arm follows, while
your *other hand* acts as a dead-man switch and gripper trigger.

  POSITIONING HAND (whichever hand is on the right side of the frame)
    left / right   -> base rotation   (servo 6)
    up / down      -> shoulder        (servo 5)
    toward / away  -> elbow reach      (servo 4)   [via palm apparent size]
    palm roll      -> wrist rotate     (servo 2)
    (its fingers are ignored, so tilting/reaching never fights the grip)

  OTHER HAND (dead-man switch + trigger)
    open palm      -> ENGAGED: the arm follows the positioning hand
    fist (curled)  -> FROZEN: arm holds still so you can reposition your
                      positioning hand freely, like lifting a mouse
    thumb+index pinch -> close the gripper (release to open)

Motion is relative to an anchor captured each time you re-engage, so the
arm never snaps: it resumes from wherever it was left and moves by how far
your hand travels from the anchor. This reuses the smoothing, slew-limiting
and deadzone machinery that made the earlier version steady.

SAFETY: run first with the arm powered down and watch the on-screen servo
readout. The gain/deadzone/direction constants below are conservative
starting points; if an axis moves the wrong way, flip the sign of its GAIN.
Keep the workspace clear the first time servos are actually driven.
"""

import time

import cv2
import mediapipe as mp

from gripper_mapping import (
    get_roll,
    get_pinch,
    get_curl,
    clamp,
    palm_scale,
    map_range,
    OneEuroFilter,
)
from xarm_movement import (
    set_servo,
    servo_units,
    SERVO_GRIP,
    SERVO_ROTATE,
    SERVO_PITCH,
    SERVO_ELBOW,
    SERVO_SHOULDER,
    SERVO_BASE,
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='C:/Users/CPS Lab/Documents/xARM/Main/models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    # Steadier tracking: keep the detector from dropping a hand the moment
    # it tilts, which is what caused the intermittent freezes.
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]

# --- Positioning-hand spatial axes -> servos ---------------------------
# Each axis is measured 0..1 from the camera, then drives a servo as an
# offset from the anchor. GAIN scales how far a servo travels per unit of
# hand motion (higher = a small hand move sweeps more of the range). Flip
# the sign if the arm moves opposite to your hand. DEADZONE ignores tiny
# motion so a still hand yields a still arm.
POS_MCPS = (5, 9, 13, 17)     # knuckles: a stable palm-center reference

# Palm apparent size -> depth. Bigger = hand closer to camera. These are
# the wrist->middle-MCP distances at the near/far ends of your comfortable
# reach; tune to your camera distance.
DEPTH_NEAR = 0.42             # hand close to camera  -> depth 1.0
DEPTH_FAR = 0.14              # hand far from camera  -> depth 0.0

AXES = {
    "x":     {"servo": SERVO_BASE,     "gain":  2.2, "deadzone": 0.02},
    "y":     {"servo": SERVO_SHOULDER, "gain": -2.2, "deadzone": 0.02},
    "depth": {"servo": SERVO_ELBOW,    "gain":  1.6, "deadzone": 0.03},
    "roll":  {"servo": SERVO_ROTATE,   "gain":  1.0, "deadzone": 0.02},
}

# Wrist pitch (servo 3) isn't controlled here; a flat webcam can't read it
# cleanly. It's held at a fixed comfortable angle instead.
WRIST_PITCH_HOLD = 0.5

# --- Other-hand switch thresholds --------------------------------------
# Engaged only while the switch hand is present and open. Curled fingers
# (fist) disengage. Hysteresis keeps it from chattering at the boundary.
ENGAGE_CURL_ON = 0.55         # curl above this -> freeze
ENGAGE_CURL_OFF = 0.40        # curl below this -> engage
# Gripper close on a firm pinch, open when clearly released.
GRIP_CLOSE = 0.65
GRIP_OPEN = 0.40

# One-euro smoothing per spatial axis (carried across freezes so re-engage
# doesn't jump). Lower min_cutoff = smoother but laggier.
AXIS_FILTER = {
    "x":     (0.9, 0.02),
    "y":     (0.9, 0.02),
    "depth": (0.6, 0.02),
    "roll":  (0.8, 0.02),
}

# Frames a switch-hand state must persist before it takes effect.
ENGAGE_FRAMES = 3


def hand_center_x(landmarks):
    return sum(landmarks[i].x for i in POS_MCPS) / len(POS_MCPS)


def read_positioning_hand(landmarks):
    """The four spatial axes, each normalized to 0..1."""
    cx = sum(landmarks[i].x for i in POS_MCPS) / len(POS_MCPS)
    cy = sum(landmarks[i].y for i in POS_MCPS) / len(POS_MCPS)
    depth = map_range(palm_scale(landmarks), DEPTH_FAR, DEPTH_NEAR, 0.0, 1.0)
    roll, _ = get_roll(landmarks)
    return {"x": cx, "y": cy, "depth": depth, "roll": roll}


def apply_deadzone(delta, deadzone):
    if abs(delta) <= deadzone:
        return 0.0
    return delta - deadzone if delta > 0 else delta + deadzone


def split_hands(result):
    """Assign roles by screen position, not by MediaPipe's L/R label.

    The right-most hand positions the arm; the left-most is the switch.
    Position is far more stable than the handedness label, which flips
    when the hands get close or a palm turns over.
    """
    hands = [lm for lm in (result.hand_landmarks or []) if len(lm) == 21]
    if len(hands) < 2:
        return None, None
    hands.sort(key=hand_center_x)          # ascending x: left first
    switch_lm = hands[0]
    positioning_lm = hands[-1]
    return positioning_lm, switch_lm


def draw_hand(frame, landmarks, width, height, color):
    points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], color, 4)
    for point in points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)


cap = cv2.VideoCapture(0)
start_time = time.time()

axis_filter = {name: None for name in AXES}

# Last norm commanded to each spatial servo; survives freezes and is what
# the next anchor builds on.
commanded_norm = {AXES[a]["servo"]: None for a in AXES}
anchor_hand = {}
anchor_servo = {}
rebaseline = True

engaged = False
engage_counter = 0
grip_closed = False


with HandLandmarker.create_from_options(options) as landmarker:
    # Park the uncontrolled wrist pitch at a neutral angle once.
    set_servo(SERVO_PITCH, WRIST_PITCH_HOLD)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Real elapsed milliseconds: VIDEO mode needs true timestamps or its
        # tracker degrades (this was a source of the earlier jitter).
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        positioning_lm, switch_lm = split_hands(result)

        now = time.time()

        # --- Switch hand: engage / freeze + gripper ---
        want_engaged = False
        if switch_lm is not None:
            curl_norm, _ = get_curl(switch_lm)
            pinch_norm, _ = get_pinch(switch_lm)

            if engaged:
                want_engaged = curl_norm < ENGAGE_CURL_ON
            else:
                want_engaged = curl_norm < ENGAGE_CURL_OFF

            # Gripper: discrete with hysteresis, independent of engage state.
            if pinch_norm > GRIP_CLOSE:
                grip_closed = True
            elif pinch_norm < GRIP_OPEN:
                grip_closed = False
            set_servo(SERVO_GRIP, 1.0 if grip_closed else 0.0)

        # Freezing is instant so the arm stops the moment you start to fist;
        # engaging is debounced so a passing gesture can't fling the arm.
        if not want_engaged:
            engaged = False
            engage_counter = 0
        elif not engaged:
            engage_counter += 1
            if engage_counter >= ENGAGE_FRAMES:
                engaged = True
                engage_counter = 0
                rebaseline = True         # re-anchor on (re-)engage
        else:
            engage_counter = 0

        # --- Positioning hand: drive the arm while engaged ---
        if positioning_lm is not None:
            raw = read_positioning_hand(positioning_lm)

            hand = {}
            for name, value in raw.items():
                if axis_filter[name] is None:
                    min_cutoff, beta = AXIS_FILTER[name]
                    axis_filter[name] = OneEuroFilter(now, value, min_cutoff, beta)
                    hand[name] = value
                else:
                    hand[name] = axis_filter[name](now, value)

            if engaged:
                if rebaseline:
                    for name, cfg in AXES.items():
                        servo_id = cfg["servo"]
                        if commanded_norm[servo_id] is None:
                            commanded_norm[servo_id] = 0.5
                        anchor_hand[name] = hand[name]
                        anchor_servo[name] = commanded_norm[servo_id]
                    rebaseline = False

                for name, cfg in AXES.items():
                    servo_id = cfg["servo"]
                    delta = apply_deadzone(hand[name] - anchor_hand[name], cfg["deadzone"])
                    value = clamp(anchor_servo[name] + cfg["gain"] * delta, 0.0, 1.0)
                    commanded_norm[servo_id] = value
                    set_servo(servo_id, value)

            color = (0, 255, 0) if engaged else (0, 165, 255)
            draw_hand(frame, positioning_lm, frame_width, frame_height, color)
        else:
            # Positioning hand gone: re-anchor when it returns, don't snap.
            rebaseline = True

        if switch_lm is not None:
            draw_hand(frame, switch_lm, frame_width, frame_height, (255, 200, 0))

        # --- HUD ---
        status = "ENGAGED" if engaged else "FROZEN"
        status_color = (0, 255, 0) if engaged else (0, 165, 255)
        if positioning_lm is None or switch_lm is None:
            status = "SHOW BOTH HANDS"
            status_color = (0, 0, 255)
        cv2.putText(frame, status, (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        cv2.putText(frame, f"grip: {'CLOSED' if grip_closed else 'open'}",
                    (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        for row, (name, cfg) in enumerate(AXES.items()):
            servo_id = cfg["servo"]
            shown = commanded_norm[servo_id]
            units = "--" if shown is None else servo_units(servo_id, shown)
            cv2.putText(
                frame, f"{name:<5} -> S{servo_id}: {units}",
                (30, 135 + 35 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

        cv2.imshow("xArm Spatial Control", frame)
        if cv2.waitKey(1) == ord(' '):
            break

cap.release()
cv2.destroyAllWindows()
