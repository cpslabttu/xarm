import cv2
import mediapipe as mp
import time

from gripper_mapping import get_gripper_value, get_rotation_value, get_pitch_value, OneEuroFilter
from xarm_movement import move_gripper, rotate_gripper, pitch_gripper

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='C:/Users/CPS Lab/Documents/xARM/Main/models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

ROT_MIN_CUTOFF = 0.8
ROT_BETA = 0.02

GRIP_MIN_CUTOFF = 0.5
GRIP_BETA = 0.02

PITCH_MIN_CUTOFF = 0.8
PITCH_BETA = 0.02

cap = cv2.VideoCapture(0)
frame_timestamp_ms = 1

rotation_filter = None
gripper_filter = None
pitch_filter = None

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

        if result.hand_landmarks and len(result.hand_landmarks[0]) == 21:
            lm = result.hand_landmarks[0]

            gripper_value, pinch_ratio = get_gripper_value(lm)
            rotation_value, roll = get_rotation_value(lm)
            pitch_value, hand_y = get_pitch_value(lm)

            now = time.time()
            if rotation_filter is None:
                rotation_filter = OneEuroFilter(
                    now,
                    rotation_value,
                    min_cutoff=ROT_MIN_CUTOFF,
                    beta=ROT_BETA
                )
                rotation_smoothed = rotation_value
            else:
                rotation_smoothed = rotation_filter(now, rotation_value)
                
            if gripper_filter is None:
                gripper_smoothed = OneEuroFilter(
                    now,
                    gripper_value,
                    min_cutoff=GRIP_MIN_CUTOFF,
                    beta=GRIP_BETA
                )
                gripper_smoothed = gripper_value
            else:
                gripper_smoothed = gripper_filter(now, gripper_value)

            if pitch_filter is None:
                pitch_filter = OneEuroFilter(
                    now,
                    pitch_value,
                    min_cutoff=PITCH_MIN_CUTOFF,
                    beta=PITCH_BETA
                )
                pitch_smoothed = pitch_value
            else:
                pitch_smoothed = pitch_filter(now, pitch_value)

            grip_sent = move_gripper(gripper_value)
            rotate_sent = rotate_gripper(rotation_smoothed)
            pitch_sent = pitch_gripper(pitch_smoothed)
            sent = grip_sent
            
            # OpenCV Markings
            points = []

            for landmark in lm:
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                points.append((x, y))

            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 4)

            for point in points:
                cv2.circle(frame, point, 5, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"Servo 1: {gripper_value}",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 0, 0),
                3
            )
            
            cv2.putText(
                frame,
                f"Servo 2: {rotation_value}",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 255),
                3
            )
            
            cv2.putText(
                frame,
                f"Servo 3: {pitch_value}",
                (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 255),
                3
            )

            cv2.putText(
                frame,
                f"Pinch Ratio: {pinch_ratio:.2f}",
                (30, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                3
            )
       # elif not result.hand_landmarks and :
            

        cv2.imshow("xArm Gripper Control", frame)

        frame_timestamp_ms += 1

        if cv2.waitKey(1) == ord(' '):
            break

cap.release()
cv2.destroyAllWindows()