# vehicles/vehicle_disc_accel.py
import math
import numpy as np
from vehicles.vehicle_base import VehicleBase, _normalize_angle

# ───────────────────────────── Discrete action grid ────────────────────────────
STEER_DEG   = [-30, -24, -18, -12, -8, -5, -2, 0, 2, 5, 8, 12, 18, 24, 30]
STEER_CHOICES = np.deg2rad(STEER_DEG).astype(np.float32)   # 15
ACC_CHOICES   = np.array([-1.0, 0.0, 1.0], dtype=np.float32)  # 后退 / 怠速 / 前进 (m/s²)

try:
    import numba as nb
    USING_NUMBA = True
except ModuleNotFoundError:
    USING_NUMBA = False

# ───────────────────────────── Numba core ──────────────────────────────────────
if USING_NUMBA:
    @nb.njit(fastmath=True, cache=True)
    def _step_disc_core(state, steer_idx, acc_idx,
                        wheelbase, max_steer, max_speed, dt):
        """Kinematic-bicycle with discrete acceleration."""
        x, y, yaw, v, _ = state
        steer = STEER_CHOICES[steer_idx]
        acc   = ACC_CHOICES[acc_idx]

        # 积分速度并限幅
        v += acc * dt
        v = min(max(v, -max_speed), max_speed)

        if abs(steer) < 1e-6:
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

# ───────────────────────────── Python fallback ─────────────────────────────────
def _python_step(state, steer_idx, acc_idx, wheelbase, max_steer,
                 max_speed, dt):
    x, y, yaw, v, _ = state
    steer = STEER_CHOICES[steer_idx]
    acc   = ACC_CHOICES[acc_idx]

    v += acc * dt
    v = np.clip(v, -max_speed, max_speed)

    if abs(steer) < 1e-6:
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
    else:
        R = wheelbase / math.tan(steer)
        d_yaw = v * dt / R
        yaw_n = yaw + d_yaw
        x += R * (math.sin(yaw_n) - math.sin(yaw))
        y += R * (-math.cos(yaw_n) + math.cos(yaw))
        yaw = yaw_n
    yaw = _normalize_angle(yaw)
    return np.array([x, y, yaw, v, steer], dtype=np.float32)

# ───────────────────────────── Vehicle class ───────────────────────────────────
class VehicleDiscAccel(VehicleBase):
    N_STEER, N_ACC = len(STEER_CHOICES), len(ACC_CHOICES)
    N_ACTIONS = N_STEER * N_ACC

    # ------- 动作解码（支持二元 or 单 id） ------------------------------------
    @staticmethod
    def decode(action):
        if isinstance(action, (tuple, list, np.ndarray)):
            s, a = int(action[0]), int(action[1])
        else:
            aid = int(action)
            s, a = divmod(aid, VehicleDiscAccel.N_ACC)
        s = max(0, min(VehicleDiscAccel.N_STEER - 1, s))
        a = max(0, min(VehicleDiscAccel.N_ACC   - 1, a))
        return s, a

    # --------------------------------------------------------------------------
    def step(self, action):
        s_idx, a_idx = self.decode(action)
        if USING_NUMBA:
            self.state = _step_disc_core(
                self.state, s_idx, a_idx,
                self.wheelbase, self.max_steer, self.max_speed, self.dt
            )
        else:
            self.state = _python_step(
                self.state, s_idx, a_idx,
                self.wheelbase, self.max_steer, self.max_speed, self.dt
            )

        # 行驶方向、几何缓存
            new_dir = 1 if self.state[3] >= 0 else -1
        if self._last_direction is not None and new_dir != self._last_direction:
            self.switch_count += 1
        self._last_direction = new_dir
        self.direction = new_dir
        self._update_geom_cache()
        return self.state, self.direction
