import math
import numpy as np
from vehicles.vehicle_base import VehicleBase, _normalize_angle

# STEER_DEG = list(range(-28, 29, 4))
STEER_DEG = [-30, -24, -18, -12, -8, -5, -2, 0, 2, 5, 8, 12, 18, 24, 30]
STEER_CHOICES = np.deg2rad(STEER_DEG).astype(np.float32)
# ARC_CHOICES = np.array([-1.0, -0.25, -0.1, 0.1, 0.25, 1.0], dtype=np.float32)
ARC_CHOICES = np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32)

try:
    import numba as nb
    USING_NUMBA = True
except ModuleNotFoundError:
    USING_NUMBA = False

if USING_NUMBA:
    @nb.njit(fastmath=True, cache=True)
    def _step_arc_core(state, steer_idx, arc_idx, wheelbase):
        x, y, yaw, _, _ = state
        steer = STEER_CHOICES[steer_idx]
        s = ARC_CHOICES[arc_idx]
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
        v = 3*s  # velocity not physically meaningful here  不过其意义也能解释过去，乘以3以满足原本的最大速度3m/s
        return np.array([x, y, yaw, v, steer], dtype=np.float32)

class VehicleArc(VehicleBase):
    N_STEER, N_ARC = len(STEER_CHOICES), len(ARC_CHOICES)
    N_ACTIONS = N_STEER * N_ARC

    @staticmethod
    def decode(action):
        if isinstance(action, (tuple, list, np.ndarray)):
            s, a = int(action[0]), int(action[1])
        else:
            aid = int(action)
            s, a = divmod(aid, VehicleArc.N_ARC)
        return max(0, min(VehicleArc.N_STEER - 1, s)), max(0, min(VehicleArc.N_ARC - 1, a))

    def step(self, action):
        s_idx, a_idx = self.decode(action)
        if USING_NUMBA:
            self.state = _step_arc_core(self.state, s_idx, a_idx, self.wheelbase)
        else:
            self._python_step(s_idx, a_idx)
        self._update_geom_cache()

        # 在 step() 函数里执行后：
        new_direction = 1 if self.state[3] >= 0 else -1
        if self._last_direction != None and new_direction != self._last_direction:
            self.switch_count += 1
        self._last_direction = new_direction
        self.direction = new_direction
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
        v = 3*s  # not divided by dt anymore
        self.state[:] = (x, y, yaw, v, steer)
