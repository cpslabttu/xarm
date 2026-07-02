import cv2
import mediapipe as mp
import time

from gripper_mapping import (
    get_pinch,
    get_roll,
    get_curl,
    count_extended_fingers,
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
pending_mode = 0
mode_counter = 0


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

        # --- Mode selection from the off-hand's finger count ---
        finger_count = None
        if mode_lm is not None:
            finger_count = count_extended_fingers(mode_lm)
            target = None
            if finger_count == 1:
                target = 0
            elif finger_count == 2:
                target = 1

            if target is not None and target != active_mode:
                if target == pending_mode:
                    mode_counter += 1
                else:
                    pending_mode = target
                    mode_counter = 1
                if mode_counter >= MODE_SWITCH_FRAMES:
                    active_mode = target
                    mode_counter = 0
            else:
                pending_mode = active_mode
                mode_counter = 0

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

            set_servo(mode["pinch"], pinch_norm)
            set_servo(mode["roll"], roll_norm)
            set_servo(mode["curl"], curl_norm)

            draw_hand(frame, control_lm, frame_width, frame_height, (0, 255, 0))

            cv2.putText(
                frame,
                f"pinch -> S{mode['pinch']}: {servo_units(mode['pinch'], pinch_norm)}",
                (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2
            )
            cv2.putText(
                frame,
                f"roll  -> S{mode['roll']}: {servo_units(mode['roll'], roll_norm)}",
                (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
            )
            cv2.putText(
                frame,
                f"curl  -> S{mode['curl']}: {servo_units(mode['curl'], curl_norm)}",
                (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
            )

        if mode_lm is not None:
            draw_hand(frame, mode_lm, frame_width, frame_height, (255, 200, 0))

        # --- Status header ---
        cv2.putText(
            frame, mode["name"], (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3
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
