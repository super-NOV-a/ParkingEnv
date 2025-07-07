"""vehicle.py —— Numba‑accelerated vehicle dynamics

定义三个类：
    - Vehicle: 公共基类（尺寸参数、碰撞多边形、角度归一化…）
    - VehicleContinuous: 连续动作动力学，接口 step(action: np.ndarray)
    - VehicleDiscrete:  离散动作动力学，接口 step(action_id: int)

若系统安装了 ``numba``，则自动启用 JIT 内核；否则退回纯 Python 版本。
外部接口（reset_state / get_shapely_polygon / _normalize_angle）保持一致。
"""
from __future__ import annotations
import math
from typing import Tuple
import numpy as np

# ---------------------------------------------------------------
# 0. Numba 可选导入
# ---------------------------------------------------------------
try:
    import numba as nb
    USING_NUMBA = True
except ModuleNotFoundError:
    USING_NUMBA = False
    print("[Vehicle] numba 未安装，使用纯 Python (较慢)。")

# ---------------------------------------------------------------
# 1. 公共工具
# ---------------------------------------------------------------
from utils import _normalize_angle

# ---------------------------------------------------------------
# 2. Numba 内核（连续 / 离散）
# ---------------------------------------------------------------
if USING_NUMBA:

    @nb.njit(fastmath=True, cache=True)
    def _step_cont_core(state, steer_cmd, acc_cmd,
                        wheelbase, max_steer, max_speed, dt, steer_filter):
        # 转向滤波
        steer = state[4] * steer_filter + steer_cmd * max_steer * (1.0 - steer_filter)
        if steer > max_steer:
            steer = max_steer
        elif steer < -max_steer:
            steer = -max_steer
        # 速度更新
        v = state[3] + acc_cmd * max_speed * dt
        vmax_back = -max_speed * 0.5
        if v > max_speed:
            v = max_speed
        elif v < vmax_back:
            v = vmax_back
        # 积分
        x, y, yaw = state[0], state[1], state[2]
        if abs(steer) < 1e-4:
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
        else:
            R = wheelbase / math.tan(steer)
            omega = v / R
            yaw_n = yaw + omega * dt
            x += R * (math.sin(yaw_n) - math.sin(yaw))
            y += R * (-math.cos(yaw_n) + math.cos(yaw))
            yaw = yaw_n
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        return np.array([x, y, yaw, v, steer], dtype=np.float32)

    @nb.njit(fastmath=True, cache=True)
    def _step_disc_core(state, action_id,
                        wheelbase, max_steer, max_speed, dt):
        steer_map = (max_steer, 0.0, -max_steer)
        speed_map = (max_speed, -max_speed, 0.0)
        steer_idx = action_id % 3
        speed_idx = action_id // 3
        steer = steer_map[steer_idx]
        target_v = speed_map[speed_idx]
        # 速度单步限制
        v = state[3]
        dv = target_v - v
        max_dv = max_speed * dt
        if dv > max_dv:
            dv = max_dv
        elif dv < -max_dv:
            dv = -max_dv
        v += dv
        # 积分
        x, y, yaw = state[0], state[1], state[2]
        if abs(steer) < 1e-4:
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
        else:
            R = wheelbase / math.tan(steer)
            omega = v / R
            yaw_n = yaw + omega * dt
            x += R * (math.sin(yaw_n) - math.sin(yaw))
            y += R * (-math.cos(yaw_n) + math.cos(yaw))
            yaw = yaw_n
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        return np.array([x, y, yaw, v, steer], dtype=np.float32)

# ---------------------------------------------------------------
# 3. 基类 Vehicle（几何 & 公共属性）
# ---------------------------------------------------------------
class Vehicle:
    """车辆基类：只负责几何 / 状态存储，具体 step 由子类实现"""

    def __init__(self, wheelbase: float, width: float, front_hang: float, rear_hang: float,
                 max_steer: float, max_speed: float, dt: float, steer_filter: float = 0.7):
        self.wheelbase = wheelbase
        self.width = width
        self.front_hang = front_hang
        self.rear_hang = rear_hang
        self.car_length = wheelbase + front_hang + rear_hang

        self.max_steer = max_steer
        self.max_speed = max_speed
        self.dt = dt
        self.steer_filter = steer_filter

        # 状态: [x, y, yaw, v, steer]
        self.state = np.zeros(5, dtype=np.float32)
        self.direction = 1

    # -------------------------------------------------------
    # 公共方法供 Env 调用
    # -------------------------------------------------------
    def reset_state(self, x: float, y: float, yaw: float):
        """将车辆姿态重置到指定位置"""
        self.state[:] = [x, y, yaw, 0.0, 0.0]
        self.direction = 1

    def get_shapely_polygon(self):
        """返回 Shapely Polygon，用于碰撞检测 / 渲染"""
        from shapely.geometry import Polygon  # 延迟导入，避免 numba 干扰
        x, y, yaw = self.state[:3]
        half_w = self.width / 2
        l_f = self.front_hang + self.wheelbase
        l_r = -self.rear_hang
        # 四个角(局部)
        corners = [
            (l_r, -half_w), (l_r, half_w),
            (l_f, half_w), (l_f, -half_w)
        ]
        world_pts = []
        for cx, cy in corners:
            rx = cx * math.cos(yaw) - cy * math.sin(yaw)
            ry = cx * math.sin(yaw) + cy * math.cos(yaw)
            world_pts.append((x + rx, y + ry))
        return Polygon(world_pts)

    # 保留角度归一化——直接引用顶层函数
    _normalize_angle = staticmethod(_normalize_angle)

    # 子类需实现
    def step(self, action):  # type: ignore[override]
        raise NotImplementedError

# ---------------------------------------------------------------
# 4. 连续动作 VehicleContinuous
# ---------------------------------------------------------------
class VehicleContinuous(Vehicle):
    """连续动作版本：action = np.array([steer_cmd, acc_cmd])"""

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, int]:  # type: ignore[override]
        steer_cmd = float(action[0])
        acc_cmd = float(action[1])
        if USING_NUMBA:
            self.state = _step_cont_core(
                self.state, steer_cmd, acc_cmd,
                self.wheelbase, self.max_steer, self.max_speed,
                self.dt, self.steer_filter
            )
        else:
            self._python_step(steer_cmd, acc_cmd)
        self.direction = 1 if self.state[3] >= 0 else -1
        return self.state, self.direction

    # ---------- 纯 Python 后备实现 ----------
    def _python_step(self, steer_cmd: float, acc_cmd: float):
        steer = self.state[4] * self.steer_filter + steer_cmd * self.max_steer * (1 - self.steer_filter)
        steer = max(-self.max_steer, min(self.max_steer, steer))
        v = self.state[3] + acc_cmd * self.max_speed * self.dt
        v = max(-self.max_speed/2, min(self.max_speed, v))
        x, y, yaw = self.state[0:3]
        if abs(steer) < 1e-4:
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
        else:
            R = self.wheelbase / math.tan(steer)
            omega = v / R
            yaw_n = yaw + omega * self.dt
            x += R * (math.sin(yaw_n) - math.sin(yaw))
            y += R * (-math.cos(yaw_n) + math.cos(yaw))
            yaw = yaw_n
        yaw = _normalize_angle(yaw)
        self.state[:] = [x, y, yaw, v, steer]

# ---------------------------------------------------------------
# 5. 离散动作 VehicleDiscrete
# ---------------------------------------------------------------
class VehicleDiscrete(Vehicle):
    """离散动作版本：action_id ∈ {0..8}"""

    def step(self, action_id: int) -> Tuple[np.ndarray, int]:  # type: ignore[override]
        if USING_NUMBA:
            self.state = _step_disc_core(
                self.state, int(action_id),
                self.wheelbase, self.max_steer, self.max_speed,
                self.dt
            )
        else:
            self._python_step(action_id)
        self.direction = 1 if self.state[3] >= 0 else -1
        return self.state, self.direction

    # ---------- 纯 Python 后备实现 ----------
    def _python_step(self, action_id: int):
        steer_map = (-self.max_steer, 0.0, self.max_steer)
        speed_map = (self.max_speed, -self.max_speed, 0.0)
        steer_idx = action_id % 3
        speed_idx = action_id // 3
        steer = steer_map[steer_idx]
        target_v = speed_map[speed_idx]
        v = self.state[3]
        dv = target_v - v
        max_dv = self.max_speed * self.dt
        dv = max(-max_dv, min(max_dv, dv))
        v += dv
        x, y, yaw = self.state[0:3]
        if abs(steer) < 1e-4:
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
        else:
            R = self.wheelbase / math.tan(steer)
            omega = v / R
            yaw_new = yaw + omega * self.dt
            x += R * (math.sin(yaw_new) - math.sin(yaw))
            y += R * (-math.cos(yaw_new) + math.cos(yaw))
            yaw = yaw_new
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        self.state[:] = [x, y, yaw, v, steer]

    # ------------------------------------------------------
    def get_shapely_polygon(self):
        """与原始 vehicle.py 相同，用于碰撞检测/渲染"""
        from shapely.geometry import Polygon  # 延迟导入，避免 numba 影响
        x, y, yaw = self.state[:3]
        half_w = self.width / 2
        l_f = self.front_hang + self.wheelbase
        l_r = -self.rear_hang
        corners = [
            (l_r, -half_w), (l_r, half_w),
            (l_f, half_w), (l_f, -half_w)
        ]
        rot = []
        for cx, cy in corners:
            rx = cx * math.cos(yaw) - cy * math.sin(yaw)
            ry = cx * math.sin(yaw) + cy * math.cos(yaw)
            rot.append((x + rx, y + ry))
        return Polygon(rot)

    # 保留 reset_state 等原方法…
    def reset_state(self, x: float, y: float, yaw: float):
        self.state[:] = [x, y, yaw, 0.0, 0.0]
        self.direction = 1

# ------------------------------------------------------------------
# ★★★ 4. 快速基准 & 示例 ★★★
# ------------------------------------------------------------------
if __name__ == "__main__":
    veh_c = VehicleContinuous(3.0, 2.0, 1.0, 1.0, math.radians(30), 5.0, 0.05)
    veh_d = VehicleDiscrete(3.0, 2.0, 1.0, 1.0, math.radians(30), 5.0, 0.05)
    act = np.array([0.2, 0.5], dtype=np.float32)  # 连续示例
    aid = 0  # 离散示例

    import time
    t0 = time.perf_counter()
    for _ in range(100_000):
        veh_c.step(act)
    print("Continuous 100k steps:", time.perf_counter() - t0, "s")

    t0 = time.perf_counter()
    for _ in range(100_000):
        veh_d.step(aid)
    print("Discrete 100k steps:", time.perf_counter() - t0, "s")
