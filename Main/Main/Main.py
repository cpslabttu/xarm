import cv2
import mediapipe as mp
import time

from gripper_mapping import (
    get_pinch,
    get_roll,
    get_curl,
    count_extended_fingers,
    clamp,
    OneEuroFilter,
)
from xarm_movement import (
    set_servo,
    servo_units,
    ALL_SERVOS,
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
    num_hands=2
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

# Which physical hand does what. If they feel swapped (the camera view is
# mirrored), swap these two labels.
CONTROL_HAND = "Right"
MODE_HAND = "Left"

# Sentinel for the "freeze the arm where it is" state, used alongside the
# mode indices in the debounce logic below.
HOLD = "hold"

# Off-hand finger counts -> what they select.
MODE_FINGERS = {1: 0, 2: 1}
HOLD_FINGERS = 4  # open palm

# Each mode repoints the 3 control gestures at different servos.
MODES = [
    {
        "name": "MODE 1: WRIST / GRIP",
        "pinch": SERVO_GRIP,
        "roll": SERVO_ROTATE,
        "curl": SERVO_PITCH,
    },
    {
        "name": "MODE 2: ARM",
        "pinch": SERVO_ELBOW,
        "roll": SERVO_BASE,
        "curl": SERVO_SHOULDER,
    },
]

GESTURES = ("pinch", "roll", "curl")

# Gestures drive servos as offsets from an anchor, not as absolute
# positions. The anchor is re-captured whenever control re-engages (hold
# released, mode switched, control hand reacquired): the servos keep the
# position they were left at, and the hand pose at that instant becomes
# the new "no change" reference. Opening your hand after a hold therefore
# means "don't move the gripper" rather than "open the gripper".
#
# A gesture must move more than this much before it commands anything, so
# the incidental pinch/curl wobble from tilting your palm doesn't drag the
# other two servos along with it.
GESTURE_DEADZONE = 0.03

# Smoothing per gesture (not per servo), so the filters carry over when
# the mode switches instead of jumping.
PINCH_MIN_CUTOFF = 0.5
PINCH_BETA = 0.02
ROLL_MIN_CUTOFF = 0.8
ROLL_BETA = 0.02
CURL_MIN_CUTOFF = 0.6
CURL_BETA = 0.02

# Require a finger count to hold for a few frames before switching modes.
MODE_SWITCH_FRAMES = 4
# Ignore control-hand gestures briefly after a switch so the twitch of
# reconfiguring both hands doesn't get sent to the new servos.
MODE_SWITCH_FREEZE_FRAMES = 8


def split_hands(result):
    control_lm = None
    mode_lm = None
    if not result.hand_landmarks:
        return control_lm, mode_lm

    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        if len(landmarks) != 21:
            continue
        label = handedness[0].category_name
        if label == CONTROL_HAND and control_lm is None:
            control_lm = landmarks
        elif label == MODE_HAND and mode_lm is None:
            mode_lm = landmarks
    return control_lm, mode_lm


def apply_deadzone(delta):
    if abs(delta) <= GESTURE_DEADZONE:
        return 0.0
    # Subtract the deadzone rather than stepping past it, so motion starts
    # from zero instead of jumping once the threshold is crossed.
    return delta - GESTURE_DEADZONE if delta > 0 else delta + GESTURE_DEADZONE


def draw_hand(frame, landmarks, width, height, color):
    points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], color, 4)
    for point in points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)


cap = cv2.VideoCapture(0)
frame_timestamp_ms = 1

pinch_filter = None
roll_filter = None
curl_filter = None

active_mode = 0
pending_target = 0
mode_counter = 0
freeze_counter = 0
hold_active = False

# Last normalized value commanded to each servo; survives holds and mode
# switches, and is what the next anchor builds on. None = never driven.
commanded_norm = {servo_id: None for servo_id in ALL_SERVOS}

# Anchor: hand pose and servo position captured at the last re-engage.
anchor_hand = {}
anchor_servo = {}
rebaseline = True


with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        control_lm, mode_lm = split_hands(result)

        # --- Mode / hold selection from the off-hand's finger count ---
        finger_count = None
        if mode_lm is not None:
            finger_count = count_extended_fingers(mode_lm)
            target = None
            if finger_count in MODE_FINGERS:
                target = MODE_FINGERS[finger_count]
            elif finger_count == HOLD_FINGERS:
                target = HOLD

            current = HOLD if hold_active else active_mode
            if target is not None and target != current:
                if target == pending_target:
                    mode_counter += 1
                else:
                    pending_target = target
                    mode_counter = 1
                if mode_counter >= MODE_SWITCH_FRAMES:
                    if target == HOLD:
                        hold_active = True
                    else:
                        # Leaving hold also picks the mode that released it.
                        hold_active = False
                        active_mode = target
                    mode_counter = 0
                    freeze_counter = MODE_SWITCH_FREEZE_FRAMES
                    # Re-anchor when control comes back, whichever way we
                    # got here: the servos stay put, the hand pose resets.
                    rebaseline = True
            else:
                pending_target = current
                mode_counter = 0

        if freeze_counter > 0:
            freeze_counter -= 1

        mode = MODES[active_mode]

        # --- Control gestures from the dominant hand ---
        if control_lm is not None:
            pinch_norm, pinch_raw = get_pinch(control_lm)
            roll_norm, roll_raw = get_roll(control_lm)
            curl_norm, curl_raw = get_curl(control_lm)

            now = time.time()
            if pinch_filter is None:
                pinch_filter = OneEuroFilter(now, pinch_norm, PINCH_MIN_CUTOFF, PINCH_BETA)
                roll_filter = OneEuroFilter(now, roll_norm, ROLL_MIN_CUTOFF, ROLL_BETA)
                curl_filter = OneEuroFilter(now, curl_norm, CURL_MIN_CUTOFF, CURL_BETA)
            else:
                pinch_norm = pinch_filter(now, pinch_norm)
                roll_norm = roll_filter(now, roll_norm)
                curl_norm = curl_filter(now, curl_norm)

            hand_norm = {
                "pinch": pinch_norm,
                "roll": roll_norm,
                "curl": curl_norm,
            }

            # The filters keep tracking while held, so the anchor captured on
            # release reflects the hand's real pose rather than a stale one.
            if freeze_counter == 0 and not hold_active:
                if rebaseline:
                    for gesture in GESTURES:
                        servo_id = mode[gesture]
                        if commanded_norm[servo_id] is None:
                            # Never driven: adopt the hand pose so the very
                            # first engage behaves as a plain absolute map.
                            commanded_norm[servo_id] = hand_norm[gesture]
                        anchor_hand[gesture] = hand_norm[gesture]
                        anchor_servo[gesture] = commanded_norm[servo_id]
                    rebaseline = False

                for gesture in GESTURES:
                    servo_id = mode[gesture]
                    delta = apply_deadzone(hand_norm[gesture] - anchor_hand[gesture])
                    value = clamp(anchor_servo[gesture] + delta, 0.0, 1.0)
                    commanded_norm[servo_id] = value
                    set_servo(servo_id, value)

            hand_color = (0, 165, 255) if hold_active else (0, 255, 0)
            draw_hand(frame, control_lm, frame_width, frame_height, hand_color)

            label_colors = ((255, 0, 0), (0, 0, 255), (0, 255, 255))
            for row, (gesture, color) in enumerate(zip(GESTURES, label_colors)):
                servo_id = mode[gesture]
                shown = commanded_norm[servo_id]
                units = "--" if shown is None else servo_units(servo_id, shown)
                cv2.putText(
                    frame,
                    f"{gesture:<5} -> S{servo_id}: {units}",
                    (30, 130 + 40 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                )
        else:
            # Hand left the frame; re-anchor rather than snap when it returns.
            rebaseline = True

        if mode_lm is not None:
            draw_hand(frame, mode_lm, frame_width, frame_height, (255, 200, 0))

        # --- Status header ---
        header = f"HOLD ({mode['name']})" if hold_active else mode["name"]
        header_color = (0, 165, 255) if hold_active else (255, 255, 255)
        cv2.putText(
            frame, header, (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, header_color, 3
        )
        if finger_count is not None:
            cv2.putText(
                frame, f"mode hand fingers: {finger_count}", (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2
            )

        cv2.imshow("xArm Gesture Control", frame)

        frame_timestamp_ms += 1

        if cv2.waitKey(1) == ord(' '):
            break

cap.release()
cv2.destroyAllWindows()
