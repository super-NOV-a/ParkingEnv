import math
import numpy as np
from vehicles.vehicle_base import VehicleBase, _normalize_angle

STEER_GRID = (-1.0, 0.0, 1.0)
SPEED_GRID = (1.0, 0.0, -1.0)

try:
    import numba as nb
    USING_NUMBA = True
except ModuleNotFoundError:
    USING_NUMBA = False

if USING_NUMBA:
    @nb.njit(fastmath=True, cache=True)
    def _step_disc_core(state, aid, wheelbase, max_steer, max_speed, dt):
        steer_cmd = STEER_GRID[aid % 3] * max_steer
        target_v = SPEED_GRID[aid // 3] * max_speed
        v = state[3]
        dv = target_v - v
        max_dv = max_speed * dt
        v += min(max(max_dv * -1, dv), max_dv)
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
    N_ACTIONS = len(STEER_GRID) * len(SPEED_GRID)

    def step(self, action):
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
        v += np.clip(dv, -max_dv, max_dv)
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