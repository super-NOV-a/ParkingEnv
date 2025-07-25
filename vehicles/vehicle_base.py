import math
from typing import Tuple
import numpy as np

def _normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))

class VehicleBase:
    def __init__(self, wheelbase, width, front_hang, rear_hang, max_steer, max_speed, dt):
        self.wheelbase = wheelbase
        self.width = width
        self.front_hang = front_hang
        self.rear_hang = rear_hang
        self.max_steer = max_steer
        self.max_speed = max_speed
        self.dt = dt

        self.state = np.zeros(5, dtype=np.float32)
        self.direction = 1
        self.switch_count = 0   # 用于vehicle_arc.py 计算运动方向切换次数
        self._last_direction = None  # 初始化为无方向

        self._geom_offset = 0.0
        self._geom_cache = np.zeros(3, dtype=np.float32)

    def reset_state(self, x: float, y: float, yaw: float):
        self.state[:] = (x, y, yaw, 0.0, 0.0)
        self._update_geom_cache()
        self.direction = 1
        self.switch_count = 0   # 用于vehicle_arc.py 计算运动方向切换次数
        self._last_direction = None  # 初始化为无方向

    def _update_geom_cache(self):
        # x_r, y_r, yaw = self.state[:3]
        # off = self._geom_offset
        # c, s = math.cos(yaw), math.sin(yaw)
        self._geom_cache[:] = self.state[:3]

    def get_pose_center(self) -> Tuple[float, float, float]:
        return tuple(self._geom_cache)

    def get_shapely_polygon(self):  # 这是耗时较多的部分
        from shapely.geometry import Polygon
        cx, cy, yaw = self.get_pose_center()
        # 以后轴为原点重新计算 4 个角
        half_w = self.width / 2
        l_f = self.wheelbase + self.front_hang        # 前保险杠
        l_r = -self.rear_hang                         # 后保险杠
        loc = [(l_r, -half_w), (l_r, half_w), (l_f, half_w), (l_f, -half_w)]
        c, s = math.cos(yaw), math.sin(yaw)
        return Polygon([(lx * c - ly * s + cx, lx * s + ly * c + cy) for lx, ly in loc])

    def step(self, action):
        raise NotImplementedError