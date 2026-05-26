"""
EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) calculations.
Pure geometry — no ML involved.
Based on Soukupova & Cech (2016) paper.
"""
import numpy as np
from scipy.spatial import distance


def calculate_EAR(eye_points):
    """
    Eye Aspect Ratio.

    Layout of 6 eye landmark points:
         p2  p3
    p1            p4
         p6  p5

    Formula: EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Open eye  → EAR ≈ 0.25–0.30
    Closing   → EAR drops toward 0
    Closed    → EAR < 0.15
    Threshold → alert if EAR < 0.25
    """
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)


def calculate_MAR(mouth_points):
    """
    Mouth Aspect Ratio — same principle as EAR but for mouth.
    High MAR (> 0.70) = open mouth = yawning.
    Yawning is early fatigue signal — often appears before eye closure.
    """
    A = distance.euclidean(mouth_points[2], mouth_points[10])
    B = distance.euclidean(mouth_points[4], mouth_points[8])
    C = distance.euclidean(mouth_points[0], mouth_points[6])
    return (A + B) / (2.0 * C)


def landmarks_to_numpy(shape):
    """Convert dlib 68-point shape object to numpy array."""
    coords = np.zeros((68, 2), dtype=int)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords