import math

# --- Pinch (thumb tip <-> index tip, normalized by palm size) ---
# norm: 0 = fingers apart (open), 1 = fingers together (closed)
PINCH_TOGETHER = 0.17
PINCH_APART = 1.6

# --- Roll (in-plane rotation of the palm around the wrist) ---
# norm: 0.5 = neutral (hand pointing up), 0/1 = rolled fully each way
NEUTRAL_ANGLE = -90.0
ROLL_HALF_RANGE = 120.0

# --- Curl throttle (middle/ring/pinky curled toward the palm) ---
# norm: 0 = fingers extended, 1 = fingers fully curled
CURL_EXTENDED = 1.0
CURL_CURLED = 0.35


class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=0.8, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.t_prev = t0
        self.x_prev = x0
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev

        a_d = self._alpha(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.t_prev = t
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def clamp(value, low, high):
    return max(low, min(high, value))


def lm_dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def map_range(value, in_min, in_max, out_min, out_max):
    value = clamp(value, in_min, in_max)
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def palm_scale(landmarks):
    # Wrist -> middle-finger MCP: a size reference that is roughly
    # invariant to how close the hand is to the camera.
    return lm_dist(landmarks[0], landmarks[9])


def get_pinch(landmarks):
    scale = palm_scale(landmarks)
    if scale == 0:
        return 0.0, 0.0

    dist = lm_dist(landmarks[4], landmarks[8]) / scale
    norm = map_range(dist, PINCH_TOGETHER, PINCH_APART, 1.0, 0.0)
    return norm, dist


def get_roll(landmarks):
    wrist = landmarks[0]

    kx = (landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 4
    ky = (landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 4

    angle = math.degrees(math.atan2(ky - wrist.y, kx - wrist.x))
    offset = (angle - NEUTRAL_ANGLE + 180) % 360 - 180

    norm = clamp(0.5 + offset / (2 * ROLL_HALF_RANGE), 0.0, 1.0)
    return norm, angle


def get_curl(landmarks):
    scale = palm_scale(landmarks)
    if scale == 0:
        return 0.0, 0.0

    # Tip-to-MCP distance for middle/ring/pinky: large when extended,
    # small when curled. Thumb + index are left free for the pinch.
    tips_mcps = [(12, 9), (16, 13), (20, 17)]
    ratio = sum(lm_dist(landmarks[t], landmarks[m]) for t, m in tips_mcps)
    ratio = ratio / (3 * scale)

    norm = map_range(ratio, CURL_CURLED, CURL_EXTENDED, 1.0, 0.0)
    return norm, ratio


def count_extended_fingers(landmarks):
    # Counts index/middle/ring/pinky that are extended, assuming the
    # mode-select hand is held roughly upright (fingertips above knuckles).
    count = 0
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if landmarks[tip].y < landmarks[pip].y:
            count += 1
    return count
