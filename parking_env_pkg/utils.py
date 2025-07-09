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

Vector = Tuple[float, float]

__all__ = [
    "normalize_angle",
    "parking_corners",
]

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _normalize_angle(angle: float) -> float:
    """将角度归一化到 [-π, π)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def parking_corners(x: float, y: float, yaw: float,
                    length: float, width: float) -> List[Vector]:
    """给定车位中心 pose → 返回 4 角顶点(顺时针)。"""
    half_w = width / 2
    l_f, l_r = length / 2, -length / 2
    base = [(l_r, -half_w), (l_r, half_w), (l_f, half_w), (l_f, -half_w)]
    out = []
    cos_t, sin_t = math.cos(yaw), math.sin(yaw)
    for dx, dy in base:
        out.append((x + dx * cos_t - dy * sin_t,
                    y + dx * sin_t + dy * cos_t))
    return out
