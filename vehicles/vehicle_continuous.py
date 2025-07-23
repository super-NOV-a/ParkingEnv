import math
import numpy as np
from vehicles.vehicle_base import VehicleBase, _normalize_angle

try:
    import numba as nb
    USING_NUMBA = True
except ModuleNotFoundError:
    USING_NUMBA = False

if USING_NUMBA:
    @nb.njit(fastmath=True, cache=True)
    def _step_cont_core(state, steer_cmd, acc_cmd, wheelbase, max_steer, max_speed, dt):
        x, y, yaw, v, steer = state
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
    """
    直接给出偏转角和加速度
    """
    N_STEER = 1        # 供 _encode_prev_action 判别“连续车辆”
    def step(self, action):
        steer_cmd, acc_cmd = float(action[0]), float(action[1])
        if USING_NUMBA:
            self.state = _step_cont_core(
                self.state, steer_cmd, acc_cmd,
                self.wheelbase, self.max_steer, self.max_speed, self.dt,
            )
        else:
            self._python_step(steer_cmd, acc_cmd)
        # ---- 方向 / 换挡 -------------------------------------------------
        new_dir = 1 if self.state[3] >= 0 else -1
        if self._last_direction is not None and new_dir != self._last_direction:
            self.switch_count += 1
        self._last_direction = new_dir
        self.direction = new_dir
        
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