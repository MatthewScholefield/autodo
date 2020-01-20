from math import cos, sin

import numpy as np


def euler_to_rot(yaw, pitch, roll):
    cy = cos(yaw)
    sy = sin(yaw)
    cp = cos(pitch)
    sp = sin(pitch)
    cr = cos(roll)
    sr = sin(roll)

    my = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])
    mp = np.array([
        [1, 0, 0],
        [0, cp, -sp],
        [0, sp, cp]
    ])
    mr = np.array([
        [cr, -sr, 0],
        [sr, cr, 0],
        [0, 0, 1]
    ])
    return my @ mp @ mr
