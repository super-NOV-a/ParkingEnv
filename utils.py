"""Utility helpers shared by parking‑env project.
================================================
All functions here are **stateless**, pure utilities that can happily be
imported by *vehicle.py* and *parking_env.py* without causing circular
dependencies.

They were extracted from the older environment/vehicle files to avoid
code duplication.
"""

from __future__ import annotations

import math
from typing import List, Tuple

__all__ = [
    "normalize_angle",
    "parking_corners",
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _normalize_angle(angle: float) -> float:
    """Wrap *angle* into the interval ``[-pi, +pi]``.

    Parameters
    ----------
    angle : float
        Angle in **radians**.

    Returns
    -------
    float
        Wrapped angle.
    """
    return math.atan2(math.sin(angle), math.cos(angle))


def parking_corners(
    x: float,
    y: float,
    yaw: float,
    length: float,
    width: float,
) -> List[Tuple[float, float]]:
    """Return the four world‑coordinates corners of a rectangular parking box.

    The rectangle is centred at *(x, y)*, has total *length* (front↔rear) and
    *width* (left↔right) and is rotated by *yaw* (rad, CCW, +x axis is yaw=0).
    """
    half_w = width / 2.0
    half_l = length / 2.0
    # Local coordinates of the rectangle corners (rear‑left, rear‑right, front‑right, front‑left)
    local = [
        (-half_l, -half_w),
        (-half_l, +half_w),
        (+half_l, +half_w),
        (+half_l, -half_w),
    ]
    corners: List[Tuple[float, float]] = []
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    for cx, cy in local:
        rx = cx * cos_y - cy * sin_y
        ry = cx * sin_y + cy * cos_y
        corners.append((x + rx, y + ry))
    return corners
