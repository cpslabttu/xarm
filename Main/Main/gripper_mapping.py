import math

DIST_MIN = 0.17
DIST_MAX = 1.6                                                         

GRIP_OPEN = 0
GRIP_CLOSED = 550

ROTATE_MIN = 0
ROTATE_CENTER = 500
ROTATE_MAX = 1000

UNITS_PER_DEGREE = 1000.0 / 240.0
NEUTRAL_ANGLE = -90.0
ROTATION_DIRECTION = 1

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


def get_gripper_value(landmarks):
    pinch = lm_dist(landmarks[4], landmarks[8])
    scale = lm_dist(landmarks[0], landmarks[9])

    if scale == 0:
        return GRIP_OPEN, 0.0

    dist = pinch / scale

    value = map_range(
        dist,
        DIST_MIN,
        DIST_MAX,
        GRIP_CLOSED,
        GRIP_OPEN
    )
    
    return int(round(value)), dist

def get_rotation_value(landmarks):
    wrist = landmarks[0]
    
    kx = (landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 4
    ky = (landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 4
    
    dx = kx - wrist.x
    dy = ky - wrist.y
    angle = math.degrees(math.atan2(dy, dx))
    
    delta = (angle - NEUTRAL_ANGLE + 180) % 360 - 180
    offset_degrees = delta * ROTATION_DIRECTION
    
    value = ROTATE_CENTER + offset_degrees * UNITS_PER_DEGREE
    value = clamp(value, ROTATE_MIN, ROTATE_MAX)
    
    return int(round(value)), angle