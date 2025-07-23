# -*- coding: utf-8 -*-
"""
vehicle_incremental.py — 增量式车辆模型
=====================================

动作定义
--------
action = (d_steer_norm, acc_norm)

* **d_steer_norm** ∈ [-1, 1]
  方向盘 **增量**（归一化）。真实增量 = `d_steer_norm × max_steer_rate × dt`，
  在每个时间步累加到当前方向盘转角 *steer*。

* **acc_norm** ∈ [-1, 1]
  纵向 **加速度增量**（归一化）。真实加速度 = `acc_norm × max_acc`，
  累加到速度 *v*。

车辆状态向量
-------------
`(x, y, yaw, v, steer)`

* **x, y**   — 后轴中心坐标（m）
* **yaw**    — 航向角（rad）
* **v**      — 纵向速度（m/s，前+ / 后‑）
* **steer**  — 前轮转角（rad）

新增特性
---------
`self.switch_count` 记录 **行驶方向（前进/倒车）切换次数**，在 `step()` 内部
自动维护，可用于奖励惩罚频繁换挡。
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from vehicles.vehicle_base import VehicleBase, _normalize_angle

# ---------------------------------------------------------------------------
# 可选 Numba 加速
# ---------------------------------------------------------------------------
try:
    import numba as nb

    USING_NUMBA = True
except ModuleNotFoundError:  # 无 Numba → 回退纯 Python
    USING_NUMBA = False


# ---------------------------------------------------------------------------
# 1. Numba 单步推进核心
# ---------------------------------------------------------------------------
if USING_NUMBA:

    @nb.njit(fastmath=True, cache=True)
    def _step_inc_core(
        state: np.ndarray,
        d_steer_norm: float,
        acc_norm: float,
        wb: float,
        max_steer: float,
        max_speed: float,
        max_acc: float,
        max_steer_rate: float,
        dt: float,
    ) -> np.ndarray:
        """Numba 版单步推进（就地修改 ``state``）。"""
        # --- 1) 方向盘 & 速度增量 ------------------------------------------------
        x, y, yaw, v, steer = state

        # 方向盘增量裁剪
        if d_steer_norm > 1.0:
            d_steer_norm = 1.0
        elif d_steer_norm < -1.0:
            d_steer_norm = -1.0
        steer += d_steer_norm * max_steer_rate * dt
        steer = min(max(steer, -max_steer), max_steer)

        # 速度增量裁剪
        if acc_norm > 1.0:
            acc_norm = 1.0
        elif acc_norm < -1.0:
            acc_norm = -1.0
        v += acc_norm * max_acc * dt
        v = min(max(v, -0.5 * max_speed), max_speed)  # 倒车限速 0.5×

        # --- 2) 单轨迹运动学 ---------------------------------------------------
        if abs(steer) < 1e-4:  # 近似直线
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
        else:  # 圆弧
            R = wb / math.tan(steer)
            d_yaw = v * dt / R
            yaw_n = yaw + d_yaw
            x += R * (math.sin(yaw_n) - math.sin(yaw))
            y += R * (-math.cos(yaw_n) + math.cos(yaw))
            yaw = yaw_n

        # --- 3) 航向角归一化 ---------------------------------------------------
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        state[:] = (x, y, yaw, v, steer)  # in‑place 写回
        return state


# ---------------------------------------------------------------------------
# 2. 增量式车辆类
# ---------------------------------------------------------------------------
class VehicleIncremental(VehicleBase):
    """增量控制车辆。

    与 :class:`VehicleBase` 的公共接口保持一致，动作空间为连续二元组
    ``(d_steer_norm, acc_norm)``。
    """

    # --------------------------- constructor ---------------------------
    def __init__(
        self,
        max_acc: float = 4.0,
        max_steer_rate: float = math.radians(90),  # 90 °/s
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)
        self.max_acc = float(max_acc)
        self.max_steer_rate = float(max_steer_rate)

    # ----------------------------- property -----------------------------
    @property
    def steer_rate(self) -> float:
        """最大方向盘转速 (rad/s)。"""
        return self.max_steer_rate

    # ----------------------- pure‑python stepping -----------------------
    def _python_step(self, d_steer_norm: float, acc_norm: float) -> None:
        """无 Numba 时的备用实现。"""
        x, y, yaw, v, steer = self.state

        # 1) 增量裁剪并累加
        steer += np.clip(d_steer_norm, -1.0, 1.0) * self.max_steer_rate * self.dt
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        v += np.clip(acc_norm, -1.0, 1.0) * self.max_acc * self.dt
        v = np.clip(v, -self.max_speed, self.max_speed)

        # 2) 单轨迹运动学
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

    # ------------------------------- main -------------------------------
    def step(self, action: Tuple[float, float]):
        """前向一步，返回 ``(state, direction)``。

        Parameters
        ----------
        action : Tuple[float, float]
            归一化动作 ``(d_steer_norm, acc_norm)``。
        """
        d_steer_norm, acc_norm = float(action[0]), float(action[1])

        # --- 动力学执行 ---------------------------------------------------
        if USING_NUMBA:
            self.state = _step_inc_core(
                self.state,
                d_steer_norm,
                acc_norm,
                self.wheelbase,
                self.max_steer,
                self.max_speed,
                self.max_acc,
                self.max_steer_rate,
                self.dt,
            )
        else:
            self._python_step(d_steer_norm, acc_norm)

        # --- 方向切换统计 -------------------------------------------------
        new_direction = 1 if self.state[3] >= 0 else -1
        if self._last_direction is not None and new_direction != self._last_direction:
            self.switch_count += 1
        self._last_direction = new_direction
        self.direction = new_direction

        # 更新几何缓存供碰撞/渲染使用
        self._update_geom_cache()
        return self.state, self.direction
