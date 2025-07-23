import math, numpy as np
from collections import deque
from typing import List, Tuple

Point = Tuple[float, float]

class ArcPathTracker:
    """纯跟踪+PID 速度的低速泊车跟踪器."""

    # --- 初始化 -----------------------------------------------------------
    def __init__(
        self,
        wheelbase: float,
        lookahead_gain: float = 0.8,
        lookahead_min: float = 0.8,
        kp_speed: float = 1.0,
    ):
        self.L = wheelbase
        self.k = lookahead_gain
        self.Lf_min = lookahead_min
        self.kp = kp_speed
        self._path: List[Point] = []
        self._cur_wp = 0  # 当前“正在追”的路径点索引

    # --- 路径输入 ---------------------------------------------------------
    def set_path(self, points: List[Point]):
        """path 应为 [(x0,y0), (x1,y1), …]，顺序表示行驶方向。"""
        self._path = points
        self._cur_wp = 0

    # --- 单步控制 ---------------------------------------------------------
    def control(
        self,
        x: float,
        y: float,
        yaw: float,
        v: float,
        v_ref: float,
    ) -> Tuple[float, float]:
        """
        输入: 当前坐标 / 朝向 / 速度 以及期望速度 v_ref (可正可负)。
        输出: (delta, v_cmd)，单位: rad, m/s
        """
        # —— 速度 PID ------------------------------------------------------
        a = self.kp * (v_ref - v)                      # 简单 P
        v_cmd = v + a                                  # 前馈 = 当前 + a·dt
        v_cmd = max(-1.5, min(1.5, v_cmd))             # 低速限制

        # —— 选前视点 ------------------------------------------------------
        if len(self._path) < 2:
            return 0.0, v_cmd

        # 找到最近点
        dists = [math.hypot(px - x, py - y) for px, py in self._path]
        nearest = int(np.argmin(dists))
        self._cur_wp = max(self._cur_wp, nearest)

        # 根据速度延长前视距离
        Lf = self.k * abs(v_cmd) + self.Lf_min

        # 沿路径累加距离直到超过 Lf
        path_len = len(self._path)
        idx = self._cur_wp
        dist_accum = 0.0
        while dist_accum < Lf and idx + 1 < path_len:
            p0, p1 = self._path[idx], self._path[idx + 1]
            dist_accum += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            idx += 1
        self._cur_wp = idx
        tx, ty = self._path[idx]

        # —— 纯跟踪几何 ----------------------------------------------------
        # 计算前视点在车辆坐标系下的角度
        dx, dy = tx - x, ty - y
        # 若倒车，应在车头改为车尾坐标；这里直接把 yaw ±π
        if v_cmd < 0:
            yaw = math.atan2(math.sin(yaw + math.pi), math.cos(yaw + math.pi))
        # 角差
        alpha = math.atan2(dy, dx) - yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # 纯跟踪公式
        Lf_actual = math.hypot(dx, dy)
        if Lf_actual < 1e-6:
            return 0.0, v_cmd
        delta = math.atan2(2 * self.L * math.sin(alpha) / Lf_actual, 1.0)

        # — 饱和与滤波 (可选) ---------------------------------------------
        max_steer = math.radians(30)
        delta = max(-max_steer, min(max_steer, delta))

        return delta, v_cmd


def sample_arc(x, y, yaw, delta, s, wheelbase, ds=0.05):
    """在常曲率 (delta, s) 上每 ds 采样一点，返回 [(x_i, y_i), …]."""
    pts = []
    steps = max(2, int(abs(s) / ds))
    sign = 1 if s >= 0 else -1
    kappa = math.tan(delta) / wheelbase
    for i in range(1, steps + 1):
        ds_i = sign * ds * i
        if abs(kappa) < 1e-6:
            xi = x + ds_i * math.cos(yaw)
            yi = y + ds_i * math.sin(yaw)
            yiw = yaw
        else:
            R = 1 / kappa
            d_yaw = ds_i * kappa
            xi = x + R * (math.sin(yaw + d_yaw) - math.sin(yaw))
            yi = y - R * (math.cos(yaw + d_yaw) - math.cos(yaw))
            yiw = yaw + d_yaw
        pts.append((xi, yi))
    return pts, yiw


# if __name__ == "__main__":
#     # 1) 高层 VehicleArc 规划出动作序列 plan = [(s_idx, a_idx), ...]
#     # 2) 解析为路径 points
#     points = []
#     x, y, yaw = env.vehicle.get_pose_center()
#     for s_idx, a_idx in plan:
#         delta = float(VehicleArc.STEER_CHOICES[s_idx])
#         s     = float(VehicleArc.ARC_CHOICES[a_idx])
#         seg_pts, yaw = sample_arc(x, y, yaw, delta, s, vehicle.wheelbase)
#         points.extend(seg_pts)
#         x, y = seg_pts[-1]

#     tracker = ArcPathTracker(vehicle.wheelbase)

#     tracker.set_path(points)

#     while not done:
#         x, y, yaw, v, _ = vehicle.state          # 取当前状态
#         # 参考速度：根据 arc 方向给 0.8 m/s 或 -0.8 m/s
#         v_ref = 0.8 * (1 if s >= 0 else -1)
#         delta, v_cmd = tracker.control(x, y, yaw, v, v_ref)
#         # ——— 发送到底层 ——————————
#         send_steer_cmd(delta)
#         send_speed_cmd(v_cmd)
