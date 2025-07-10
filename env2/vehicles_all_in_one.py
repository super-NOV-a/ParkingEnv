"""vehicles.py – Unified vehicle dynamics module
================================================
This file **refactors** the previous scattered implementations into **one
cohesive module** with:

* `VehicleBase` — geometry utilities, state storage, common helpers.
* `VehicleContinuous` — continuous actions `[steer_cmd, accel_cmd]`.
* `VehicleDiscAccel` — coarse *steer‑to‑angle* × *target‑speed* grid (3 × 3 = 9
  actions, compatible with the *original* discrete‑acceleration env).
* `VehicleArc` — 2‑D discrete **(steer notch, arc length)** grid (15 × 4 = 60).

All three classes expose the *same* public API:
    ``reset_state(x, y, yaw)`` and ``step(action) -> (state, direction)``
where
    ``state = np.array([x, y, yaw, v, steer])``.

Each subclass offers a pure‑NumPy fallback plus an *optional* Numba kernel (if
``numba`` is installed).  Geometry helpers (Shapely polygon etc.) live in the
base class and reuse pre‑computed offsets, so no duplicate maths.
"""

from __future__ import annotations
import math
from typing import Sequence, Tuple, Union

import numpy as np

try:
    import numba as nb  # type: ignore
    USING_NUMBA = True
except ModuleNotFoundError:
    USING_NUMBA = False

# ---------------------------------------------------------------------------
# Shared helpers -------------------------------------------------------------
# ---------------------------------------------------------------------------

def _normalize_angle(a: float) -> float:  # inline small util – avoids extra import
    """Wrap angle to [-π, π)."""
    return math.atan2(math.sin(a), math.cos(a))

# ---------------------------------------------------------------------------
# 0.  Base geometry / cache ---------------------------------------------------
# ---------------------------------------------------------------------------
class VehicleBase:
    """Base class: *only* geometry + common bookkeeping.

    Sub‑classes implement ``step()``.
    """

    def __init__(
        self,
        wheelbase: float,
        width: float,
        front_hang: float,
        rear_hang: float,
        max_steer: float,
        max_speed: float,
        dt: float,
    ):
        self.wheelbase = wheelbase
        self.width = width
        self.front_hang = front_hang
        self.rear_hang = rear_hang
        self.max_steer = max_steer
        self.max_speed = max_speed
        self.dt = dt

        # state = [x_r, y_r, yaw, v, steer]
        self.state = np.zeros(5, dtype=np.float32)
        self.direction = 1  # 1 fwd, ‑1 rev

        # pre‑compute offset of geometric centre (mid‑length) from rear axle
        self._geom_offset = self.wheelbase / 2 + (front_hang - rear_hang) / 2
        self._geom_cache = np.zeros(3, dtype=np.float32)  # cx, cy, yaw

    # ------------------------------------------------------------------
    # public helpers ----------------------------------------------------
    def reset_state(self, x: float, y: float, yaw: float):
        self.state[:] = (x, y, yaw, 0.0, 0.0)
        self.direction = 1
        self._update_geom_cache()

    def _update_geom_cache(self):
        x_r, y_r, yaw = self.state[:3]
        off = self._geom_offset
        c, s = math.cos(yaw), math.sin(yaw)
        self._geom_cache[:] = (x_r + off * c, y_r + off * s, yaw)

    def get_pose_center(self) -> Tuple[float, float, float]:
        return tuple(self._geom_cache)

    def get_shapely_polygon(self):
        from shapely.geometry import Polygon  # lazy import
        cx, cy, yaw = self.get_pose_center()
        half_w = self.width / 2
        l_f = self.wheelbase / 2 + self.front_hang
        l_r = -self.wheelbase / 2 - self.rear_hang
        # local corners (centre origin)
        loc = [
            (l_r, -half_w), (l_r, half_w), (l_f, half_w), (l_f, -half_w)
        ]
        world = []
        c, s = math.cos(yaw), math.sin(yaw)
        for lx, ly in loc:
            wx = lx * c - ly * s + cx
            wy = lx * s + ly * c + cy
            world.append((wx, wy))
        return Polygon(world)

    # subclasses must implement
    def step(self, action):  # type: ignore[override]
        raise NotImplementedError

# ---------------------------------------------------------------------------
# 1. Continuous actions -------------------------------------------------------
# ---------------------------------------------------------------------------
if USING_NUMBA:

    @nb.njit(fastmath=True, cache=True)
    def _step_cont_core(state, steer_cmd, acc_cmd, wheelbase, max_steer, max_speed, dt):
        x, y, yaw, v, steer = state
        # Limit & apply commands directly (simple bicycle)
        steer = max(-max_steer, min(max_steer, steer_cmd * max_steer))
        v += acc_cmd * max_speed * dt
        v = max(-0.5 * max_speed, min(max_speed, v))
        if abs(steer) < 1e-4:
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
        else:
            R = wheelbase / math.tan(steer)
            d_yaw = v * dt / R
            yaw_n = yaw + d_yaw
            x += R * (math.sin(yaw_n) - math.sin(yaw))
            y += R * (-math.cos(yaw_n) + math.cos(yaw))
            yaw = yaw_n
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        return np.array([x, y, yaw, v, steer], dtype=np.float32)


class VehicleContinuous(VehicleBase):
    """`action = (steer_cmd ∈ [‑1,1], accel_cmd ∈ [‑1,1])`."""

    def step(self, action: Sequence[float]):  # type: ignore[override]
        steer_cmd, acc_cmd = float(action[0]), float(action[1])
        if USING_NUMBA:
            self.state = _step_cont_core(
                self.state, steer_cmd, acc_cmd,
                self.wheelbase, self.max_steer, self.max_speed, self.dt,
            )
        else:
            self._python_step(steer_cmd, acc_cmd)
        self.direction = 1 if self.state[3] >= 0 else -1
        self._update_geom_cache()
        return self.state, self.direction

    def _python_step(self, steer_cmd, acc_cmd):
        steer = max(-self.max_steer, min(self.max_steer, steer_cmd * self.max_steer))
        v = self.state[3] + acc_cmd * self.max_speed * self.dt
        v = max(-0.5 * self.max_speed, min(self.max_speed, v))
        x, y, yaw = self.state[:3]
        if abs(steer) < 1e-4:
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
        else:
            R = self.wheelbase / math.tan(steer)
            d_yaw = v * self.dt / R
            yaw_n = yaw + d_yaw
            x += R * (math.sin(yaw_n) - math.sin(yaw))
            y += R * (-math.cos(yaw_n) + math.cos(yaw))
            yaw = yaw_n
        yaw = _normalize_angle(yaw)
        self.state[:] = (x, y, yaw, v, steer)

# ---------------------------------------------------------------------------
# 2. Discrete accel/steer grid (3×3) -----------------------------------------
# ---------------------------------------------------------------------------
STEER_GRID = (-1.0, 0.0, 1.0)  # scaled × max_steer
SPEED_GRID = (1.0, 0.0, -1.0)  # scaled × max_speed (fwd / coast / rev)
N_DISC = len(STEER_GRID) * len(SPEED_GRID)  # 9

if USING_NUMBA:

    @nb.njit(fastmath=True, cache=True)
    def _step_disc_core(state, aid, wheelbase, max_steer, max_speed, dt):
        steer_cmd = STEER_GRID[aid % 3] * max_steer
        target_v = SPEED_GRID[aid // 3] * max_speed
        # first‑order speed response
        v = state[3]
        dv = target_v - v
        max_dv = max_speed * dt
        if dv > max_dv:
            dv = max_dv
        elif dv < -max_dv:
            dv = -max_dv
        v += dv
        # kinematics
        x, y, yaw = state[:3]
        steer = steer_cmd
        if abs(steer) < 1e-4:
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
        else:
            R = wheelbase / math.tan(steer)
            d_yaw = v * dt / R
            yaw_n = yaw + d_yaw
            x += R * (math.sin(yaw_n) - math.sin(yaw))
            y += R * (-math.cos(yaw_n) + math.cos(yaw))
            yaw = yaw_n
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        return np.array([x, y, yaw, v, steer], dtype=np.float32)


class VehicleDiscAccel(VehicleBase):
    """9 discrete actions → *coarse* steer & target speed grid."""

    N_ACTIONS = N_DISC

    def step(self, action: int):  # type: ignore[override]
        aid = int(action) % self.N_ACTIONS
        if USING_NUMBA:
            self.state = _step_disc_core(
                self.state, aid,
                self.wheelbase, self.max_steer, self.max_speed, self.dt,
            )
        else:
            self._python_step(aid)
        self.direction = 1 if self.state[3] >= 0 else -1
        self._update_geom_cache()
        return self.state, self.direction

    def _python_step(self, aid):
        steer = STEER_GRID[aid % 3] * self.max_steer
        target_v = SPEED_GRID[aid // 3] * self.max_speed
        v = self.state[3]
        dv = target_v - v
        max_dv = self.max_speed * self.dt
        dv = max(-max_dv, min(max_dv, dv))
        v += dv
        x, y, yaw = self.state[:3]
        if abs(steer) < 1e-4:
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
        else:
            R = self.wheelbase / math.tan(steer)
            d_yaw = v * self.dt / R
            yaw_n = yaw + d_yaw
            x += R * (math.sin(yaw_n) - math.sin(yaw))
            y += R * (-math.cos(yaw_n) + math.cos(yaw))
            yaw = yaw_n
        yaw = _normalize_angle(yaw)
        self.state[:] = (x, y, yaw, v, steer)

# ---------------------------------------------------------------------------
# 3. Discrete arc‑length grid (15×4) -----------------------------------------
# ---------------------------------------------------------------------------
STEER_DEG = list(range(-28, 29, 4))  # ‑28 … +28, step 4 → 15
STEER_CHOICES = np.deg2rad(STEER_DEG).astype(np.float32)
ARC_CHOICES = np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32)
N_STEER, N_ARC = len(STEER_CHOICES), len(ARC_CHOICES)

if USING_NUMBA:

    STEER_CHOICES_NB = STEER_CHOICES
    ARC_CHOICES_NB = ARC_CHOICES

    @nb.njit(fastmath=True, cache=True)
    def _step_arc_core(state, steer_idx, arc_idx, wheelbase, dt):
        x, y, yaw, _, _ = state
        steer = STEER_CHOICES_NB[steer_idx]
        s = ARC_CHOICES_NB[arc_idx]
        if abs(steer) < 1e-6:
            x += s * math.cos(yaw)
            y += s * math.sin(yaw)
        else:
            R = wheelbase / math.tan(steer)
            d_yaw = s / R
            yaw_n = yaw + d_yaw
            x += R * (math.sin(yaw_n) - math.sin(yaw))
            y += R * (-math.cos(yaw_n) + math.cos(yaw))
            yaw = yaw_n
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        v = s / dt
        return np.array([x, y, yaw, v, steer], dtype=np.float32)


class VehicleArc(VehicleBase):
    """15×4 discrete (steer notch, arc length). Accepts int or pair."""

    N_STEER, N_ARC = N_STEER, N_ARC
    N_ACTIONS = N_STEER * N_ARC

    @staticmethod
    def decode(action: Union[int, Sequence[int]]):
        if isinstance(action, (tuple, list, np.ndarray)):
            s, a = int(action[0]), int(action[1])
        else:
            aid = int(action)
            s, a = divmod(aid, N_ARC)
        return max(0, min(N_STEER - 1, s)), max(0, min(N_ARC - 1, a))

    def step(self, action):  # type: ignore[override]
        s_idx, a_idx = self.decode(action)
        if USING_NUMBA:
            self.state = _step_arc_core(self.state, s_idx, a_idx, self.wheelbase, self.dt)
        else:
            self._python_step(s_idx, a_idx)
        self.direction = 1 if self.state[3] >= 0 else -1
        self._update_geom_cache()
        return self.state, self.direction

    def _python_step(self, s_idx, a_idx):
        steer = STEER_CHOICES[s_idx]
        s = ARC_CHOICES[a_idx]
        x, y, yaw = self.state[:3]
        if abs(steer) < 1e-6:
            x += s * math.cos(yaw)
            y += s * math.sin(yaw)
        else:
            R = self.wheelbase / math.tan(steer)
            d_yaw = s / R
            yaw_n = yaw + d_yaw
            x += R * (math.sin(yaw_n) - math.sin(yaw))
            y += R * (-math.cos(yaw_n) + math.cos(yaw))
            yaw = yaw_n
        yaw = _normalize_angle(yaw)
        v = s / self.dt
        self.state[:] = (x, y, yaw, v, steer)

# ---------------------------------------------------------------------------
# Quick sanity check ---------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    vehs = [
        VehicleContinuous(2.8, 1.8, 0.9, 0.9, math.radians(30), 5.0, 0.1),
        VehicleDiscAccel(2.8, 1.8, 0.9, 0.9, math.radians(30), 5.0, 0.1),
        VehicleArc(2.8, 1.8, 0.9, 0.9, math.radians(30), 5.0, 0.1),
    ]
    actions = [np.array([0.3, 0.4]), 2, (10, 3)]
    for v, a in zip(vehs, actions):
        v.reset_state(0.0, 0.0, 0.0)
        s, d = v.step(a)
        print(v.__class__.__name__, "→", s)
