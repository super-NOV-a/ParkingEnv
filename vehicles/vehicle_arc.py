import math
import numpy as np
from vehicles.vehicle_base import VehicleBase, _normalize_angle

# ------------------- discrete steering & arc tables -------------------
STEER_DEG = [-30, -24, -18, -12, -8, -5, -2, 0, 2, 5, 8, 12, 18, 24, 30]
STEER_CHOICES = np.deg2rad(STEER_DEG).astype(np.float32)
# fmt: off
ARC_CHOICES   = np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32)
# fmt: on

# ----------------------------------------------------------------------
try:
    import numba as nb
    USING_NUMBA = True
except ModuleNotFoundError:
    USING_NUMBA = False

if USING_NUMBA:
    @nb.njit(fastmath=True, cache=True)
    def _step_arc_core(state, steer, s, wheelbase):
        """Raw single‑step kinematics for the numba path."""
        x, y, yaw, _, _ = state
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
        v = 3 * s  # keep "velocity" field roughly consistent with old design
        return np.array([x, y, yaw, v, steer], dtype=np.float32)


class VehicleArc(VehicleBase):
    """Discrete arc‑length vehicle with steer‑change limiter (≤20° per action).

    The public API is unchanged – the class still takes `(steer_idx, arc_idx)`
    or flat integer as its action.  Internally we clamp successive steering
    commands so that `|Δsteer| ≤ 30 deg`, by moving to the nearest allowed
    `STEER_DEG` within that bound.  This guarantees piece‑wise continuity while
    keeping existing caller code intact.
    """

    N_STEER, N_ARC = len(STEER_CHOICES), len(ARC_CHOICES)
    N_ACTIONS = N_STEER * N_ARC
    MAX_STEER_DELTA_DEG = 20  # ° per macro‑step

    # --------------------------- constructor ---------------------------
    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self._prev_steer_idx: int | None = None  # track last discrete steer idx

    # --------------------------- helpers -------------------------------
    @staticmethod
    def decode(action):
        """Return `(steer_idx, arc_idx)` from heterogeneous action encoding."""
        if isinstance(action, (tuple, list, np.ndarray)):
            s, a = int(action[0]), int(action[1])
        else:
            aid = int(action)
            s, a = divmod(aid, VehicleArc.N_ARC)
        s = max(0, min(VehicleArc.N_STEER - 1, s))
        a = max(0, min(VehicleArc.N_ARC - 1, a))
        return s, a

    # --------------------------- stepping ------------------------------
    def _limit_steer_idx(self, s_idx: int) -> int:
        """Clamp `s_idx` so that |Δsteer| ≤ 30° w.r.t previous index."""
        if self._prev_steer_idx is None:
            return s_idx  # first action – nothing to compare
        prev_deg = STEER_DEG[self._prev_steer_idx]
        target_deg = STEER_DEG[s_idx]
        delta = target_deg - prev_deg
        if abs(delta) <= self.MAX_STEER_DELTA_DEG:
            return s_idx
        # clamp to ±30° around previous
        clamped_deg = prev_deg + self.MAX_STEER_DELTA_DEG * (1 if delta > 0 else -1)
        # choose nearest allowed discrete value
        best_idx = min(range(self.N_STEER), key=lambda i: abs(STEER_DEG[i] - clamped_deg))
        return best_idx

    # ------------------------------------------------------------------
    def step(self, action):
        s_idx, a_idx = self.decode(action)
        s_idx = self._limit_steer_idx(s_idx)
        steer = float(STEER_CHOICES[s_idx])
        s = float(ARC_CHOICES[a_idx])

        # --- kinematics ---
        if USING_NUMBA:
            self.state = _step_arc_core(self.state, steer, s, self.wheelbase)
        else:
            self._python_step(steer, s)

        self._update_geom_cache()

        # update direction & steer memory
        new_direction = 1 if self.state[3] >= 0 else -1
        if self._last_direction is not None and new_direction != self._last_direction:
            self.switch_count += 1
        self._last_direction = new_direction
        self.direction = new_direction
        self._prev_steer_idx = s_idx
        return self.state, self.direction

    # --------------------- pure‑python path ---------------------------
    def _python_step(self, steer: float, s: float):
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
        v = 3 * s
        self.state[:] = (x, y, yaw, v, steer)
