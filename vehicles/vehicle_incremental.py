# -*- coding: utf-8 -*-
"""
VehicleIncremental — 连续动作 = (方向盘增量, 加速度)
-------------------------------------------------
方向盘增量：归一化到 [-1,1]，乘以 self.max_steer_rate * dt 后累加到当前 steer
加速度   ：归一化到 [-1,1]，乘以 self.max_acc            后累加到当前速度 v
"""
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
    def _step_inc_core(state, d_steer_norm, acc_norm,
              wb, max_steer, max_speed,
              max_acc, max_steer_rate, dt):
        """Numba-compatible 单步推进 (in-place 改 state)"""
        # unpack
        x, y, yaw, v, steer = state

        # ---------------- 1) 方向盘增量 & 速度增量 ---------------- #
        # 手工 clip —— numba 不支持 np.clip 对标量
        if d_steer_norm > 1.0:
            d_steer_norm = 1.0
        elif d_steer_norm < -1.0:
            d_steer_norm = -1.0
        steer += d_steer_norm * max_steer_rate * dt
        if steer > max_steer:
            steer = max_steer
        elif steer < -max_steer:
            steer = -max_steer

        if acc_norm > 1.0:
            acc_norm = 1.0
        elif acc_norm < -1.0:
            acc_norm = -1.0
        v += acc_norm * max_acc * dt
        v_min = -0.5 * max_speed
        if v > max_speed:
            v = max_speed
        elif v < v_min:
            v = v_min

        # ---------------- 2) 单轨迹运动学 ----------------------- #
        if abs(steer) < 1e-4:
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
        else:
            R = wb / math.tan(steer)
            d_yaw = v * dt / R
            yaw_n = yaw + d_yaw
            x += R * (math.sin(yaw_n) - math.sin(yaw))
            y += R * (-math.cos(yaw_n) + math.cos(yaw))
            yaw = yaw_n

        # ---------------- 3) 归一化航向角 ----------------------- #
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        return np.array([x, y, yaw, v, steer], dtype=np.float32)



class VehicleIncremental(VehicleBase):
    """增量控制车型：action = (d_steer_norm, acc_norm)"""

    def __init__(self,
                 max_acc: float = 4.0,          # m/s²
                 max_steer_rate: float = math.radians(90),  # rad/s
                 **base_kwargs):
        super().__init__(**base_kwargs)
        self.max_acc = float(max_acc)
        self.max_steer_rate = float(max_steer_rate)

    # ------------------------------------------------------------------ #
    #  公共属性 —— demo 里直接用到
    # ------------------------------------------------------------------ #
    @property
    def steer_rate(self):
        """最大方向盘转速 (rad/s)"""
        return self.max_steer_rate

    # ------------------------------------------------------------------ #
    #  单步推进
    # ------------------------------------------------------------------ #

    # ---------------- python 版本 ---------------- #
    def _python_step(self, d_steer_norm, acc_norm):
        x, y, yaw, v, steer = self.state
        steer += np.clip(d_steer_norm, -1, 1) * self.max_steer_rate * self.dt
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        v += np.clip(acc_norm, -1, 1) * self.max_acc * self.dt
        v = np.clip(v, -0.5 * self.max_speed, self.max_speed)

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

    # -------------------------------------------------------------- #
    def step(self, action):
        d_steer_norm, acc_norm = float(action[0]), float(action[1])
        if USING_NUMBA:
            self.state = _step_inc_core(
                self.state, d_steer_norm, acc_norm,
                self.wheelbase, self.max_steer, self.max_speed,
                self.max_acc, self.max_steer_rate, self.dt,
            )
        else:
            self._python_step(d_steer_norm, acc_norm)

        self.direction = 1 if self.state[3] >= 0 else -1
        self._update_geom_cache()
        return self.state, self.direction
