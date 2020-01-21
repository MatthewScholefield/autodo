import re
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


def extract_image_id(filename):
    m = re.match(r'.*(ID_[0-9a-f]{9}).jpg', filename)
    return m.group(1) if m else None


def iou(box1, box2):
    """Finds overlap of two bounding boxes"""
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])
    if xmin >= xmax or ymin >= ymax:
        return 0
    box1area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    intersection = (xmax - xmin) * (ymax - ymin)
    union = box1area + box2area - intersection
    return intersection / union
