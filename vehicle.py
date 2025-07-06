import numpy as np
import math
from shapely.geometry import Polygon


class Vehicle:
    def __init__(self, wheelbase, width, front_hang, rear_hang, max_steer, max_speed, dt, steer_filter=0.7):
        self.wheelbase = wheelbase
        self.width = width
        self.front_hang = front_hang
        self.rear_hang = rear_hang
        self.car_length = wheelbase + front_hang + rear_hang

        # 状态: [x, y, yaw, velocity, steer_angle]
        self.state = np.zeros(5)
        self.direction = 1  # 1: 前进，-1: 后退

        # 动力学模型参数
        self.max_steer = max_steer
        self.max_speed = max_speed
        self.dt = dt
        self.steer_filter = steer_filter

        # 上一次转向角，用于滤波
        self.prev_steer = 0.0

    def _normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def step(self, action):
        """
        action: np.array([steer_cmd, throttle_brake])，范围[-1,1]
        返回更新后的状态和方向
        """
        steer_cmd = np.clip(action[0], -1, 1)
        throttle_brake = np.clip(action[1], -1, 1)

        # 转向滤波
        current_steer = self.state[4]
        new_steer = current_steer * self.steer_filter + steer_cmd * self.max_steer * (1 - self.steer_filter)
        self.state[4] = new_steer

        # 速度更新（简单模型）
        # throttle_brake>0加速，<0减速
        v = self.state[3]
        acc = throttle_brake * self.max_speed  # 简单比例控制
        v += acc * self.dt
        v = np.clip(v, -self.max_speed, self.max_speed)

        # 根据速度方向调整方向符号
        if v >= 0:
            self.direction = 1
        else:
            self.direction = -1

        self.state[3] = v

        # 根据自行车模型更新位置
        x, y, yaw, _, steer = self.state
        if abs(steer) < 1e-4:
            # 无转向角，直线运动
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
        else:
            turning_radius = self.wheelbase / math.tan(steer)
            angular_velocity = v / turning_radius
            yaw += angular_velocity * self.dt
            x += turning_radius * (math.sin(yaw) - math.sin(self.state[2]))
            y += turning_radius * (-math.cos(yaw) + math.cos(self.state[2]))

        self.state[0] = x
        self.state[1] = y
        self.state[2] = self._normalize_angle(yaw)

        return self.state, self.direction

    def get_shapely_polygon(self):
        """
        返回车辆形状的Shapely Polygon，用于碰撞检测和渲染
        """
        x, y, yaw = self.state[:3]
        half_w = self.width / 2
        l_f = self.front_hang + self.wheelbase
        l_r = -self.rear_hang  # 后悬为负方向

        # 车辆四个角点(顺时针)
        corners = [
            (l_r, -half_w),
            (l_r, half_w),
            (l_f, half_w),
            (l_f, -half_w),
        ]

        rotated = []
        for cx, cy in corners:
            rx = cx * math.cos(yaw) - cy * math.sin(yaw)
            ry = cx * math.sin(yaw) + cy * math.cos(yaw)
            rotated.append((x + rx, y + ry))
        return Polygon(rotated)

    def reset_state(self, x, y, yaw):
        self.state[0] = x
        self.state[1] = y
        self.state[2] = yaw
        self.state[3] = 0.0  # 初速度为0
        self.state[4] = 0.0  # 初始转向角为0
        self.direction = 1
