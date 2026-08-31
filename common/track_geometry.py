import math

SEGMENT_LENGTH = 0.13

# (centre_x, centre_y, rotation_about_z_rad)
SEGMENTS = [
    (-0.5950, 0.0000,  0.000000),
    (-0.4858, 0.0096,  0.174533),
    (-0.3800, 0.0379,  0.349066),
    (-0.2741, 0.0663,  0.174533),
    (-0.1650, 0.0758,  0.000000),
    (-0.0558, 0.0663, -0.174533),
    ( 0.0500, 0.0379, -0.349066),
    ( 0.1559, 0.0096, -0.174533),
]


def _endpoints():
    out = []
    half = SEGMENT_LENGTH / 2.0
    for cx, cy, th in SEGMENTS:
        dx, dy = math.cos(th) * half, math.sin(th) * half
        out.append(((cx - dx, cy - dy), (cx + dx, cy + dy)))
    return out


_SEGS = _endpoints()

_CUM = [0.0]
for _a, _b in _SEGS:
    _CUM.append(_CUM[-1] + math.hypot(_b[0] - _a[0], _b[1] - _a[1]))
TOTAL_LENGTH = _CUM[-1]


def _point_to_segment(px, py, a, b):
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 <= 1e-12:
        return math.hypot(px - ax, py - ay), 0.0, 0.0
    t = ((px - ax) * vx + (py - ay) * vy) / L2
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * vx, ay + t * vy
    dist = math.hypot(px - qx, py - qy)
    cross = vx * (py - ay) - vy * (px - ax)     # >0 = left of path
    sign = 1.0 if cross >= 0.0 else -1.0
    return dist, sign, t * math.sqrt(L2)


def cross_track_error(x, y):
    """
    Returns (signed_error_metres, progress_fraction_0_to_1).
    Positive error = robot is to the LEFT of the centreline.
    """
    best_d = float("inf")
    best_sign = 1.0
    best_arc = 0.0

    for i, (a, b) in enumerate(_SEGS):
        d, s, along = _point_to_segment(x, y, a, b)
        if d < best_d:
            best_d = d
            best_sign = s
            best_arc = _CUM[i] + along

    progress = best_arc / TOTAL_LENGTH if TOTAL_LENGTH > 0.0 else 0.0
    return best_sign * best_d, max(0.0, min(1.0, progress))