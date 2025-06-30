import time
import warnings
import pygame
import numpy as np
import random
import math
from typing import Tuple, Dict, List, Optional
import gymnasium as gym
from gymnasium import spaces


class CarEnv(gym.Env):
    metadata = {'render_modes': ['human'], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        # 环境参数 - 值域范围[-1,1]
        self.total_reward = None
        self.reward = None
        self.agent_pos = None
        self.agent_heading = None
        self.agent_vel = None
        self.steering_angle = None
        self.obstacle_abs_positions = None
        self.points_vector = None
        self.target = None
        self.target_vector = None
        self.min_distance = None
        self.target_distance = None
        self.last_action = None
        self.current_points = None
        self.low = -1.0
        self.high = 1.0
        self.min_points = 5
        self.max_points = 30
        self.curriculum_level = 0
        self.max_level = 20
        self.min_level = 0

        # 车辆物理参数
        self.max_acceleration = 0.2  # 最大加速度
        self.max_steering_vel = 0.3  # 最大转向角速度 (rad/s)
        self.max_steering_angle = np.pi / 6  # 最大前轮转角 (30度)
        self.dt = 0.1  # 时间步长
        self.max_velocity = 0.2  # 最大速度     0.2在[-1,1]中对应30m的3m/s
        self.max_angular_velocity = 1.0  # 最大角速度 (rad/s)
        self.vehicle_length = 0.1  # 车辆长度
        self.vehicle_width = 0.05  # 车辆宽度
        self.friction = 0.05  # 摩擦系数
        self.wheel_base = 0.06  # 轴距

        # 环境限制
        self.max_steps = 300  # 最大步数
        self.current_step = 0
        self.target_threshold = 0.05  # 到达目标的距离阈值

        # 动作空间: [前向加速度, 转向角速度]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 观测空间设计 (所有值都在[-1,1]范围内)
        # 修改后观测空间: 1朝向 + 2车身速度(纵向,横向) + 1偏转角 + 2上一动作 + 障碍物信息 + 3目标信息(车身系x,y,距离)
        obs_shape = 1 + 2 + 1 + 2 + 3 * self.max_points + 3
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_shape,),
            dtype=np.float32
        )

        # 无效点可能的角点位置
        self.corner_points = [
            np.array([self.low, self.low]),
            np.array([self.low, self.high]),
            np.array([self.high, self.low]),
            np.array([self.high, self.high])
        ]

        # 渲染设置
        self.render_mode = render_mode
        self.scale = 400  # 比例因子
        if render_mode == "human":
            pygame.init()
            self.screen_size = (int((self.high - self.low) * self.scale),
                                int((self.high - self.low) * self.scale))
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("车辆强化学习环境")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)

        # 环境状态初始化
        self.reset()

    def _global_to_body_frame(self, vector: np.ndarray) -> np.ndarray:
        """将全局坐标系向量转换到车身坐标系"""
        heading = self.agent_heading
        rotation_matrix = np.array([
            [np.cos(heading), np.sin(heading)],
            [-np.sin(heading), np.cos(heading)]
        ])
        return rotation_matrix @ vector

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # 重置状态变量
        self.current_step = 0
        self.total_reward = 0
        self.reward = 0

        # 更新课程难度
        # self._update_level()
        self.current_points = int(self.min_points + (self.max_points - self.min_points) *
                                  (self.curriculum_level / self.max_level))

        # 重置车辆状态 (值域范围[-1,1])
        self.agent_pos = self.np_random.uniform(self.low + 0.1, self.high - 0.1, size=2)
        self.agent_heading = self.np_random.uniform(0, 2 * np.pi)  # 车辆朝向
        self.agent_vel = np.zeros(2)  # 车辆速度向量
        self.steering_angle = 0.0  # 前轮转角

        # 初始化障碍物
        self.obstacle_abs_positions = []
        self.points_vector = np.zeros((self.max_points, 3), dtype=np.float32)

        # 生成有效随机点
        for i in range(self.current_points):
            point = self.np_random.uniform(self.low + 0.05, self.high - 0.05, size=2)
            self.obstacle_abs_positions.append(point)

        # 填充无效点（角点）
        for i in range(self.current_points, self.max_points):
            corner = self.corner_points[i % len(self.corner_points)]
            self.obstacle_abs_positions.append(corner)

        # 重置目标位置
        self.target = self.np_random.uniform(self.low + 0.1, self.high - 0.1, size=2)
        # 确保目标不与车辆初始位置太近
        while np.linalg.norm(self.target - self.agent_pos) < 0.3:
            self.target = self.np_random.uniform(self.low + 0.1, self.high - 0.1, size=2)

        # 重置距离记录
        self.min_distance = float('inf')

        # 更新最近障碍物
        self._update_nearest_obstacle()
        self.last_action = np.zeros(2)

        return self._get_obs(), {}

    def _update_level(self):
        """根据智能体表现更新课程难度"""
        if self.total_reward >= 300:  # 升级条件
            self.curriculum_level = min(self.curriculum_level + 1, self.max_level)
        elif self.total_reward <= 0:  # 降级条件
            self.curriculum_level = max(self.curriculum_level - 1, self.min_level)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 解析动作 (值域[-1,1])
        acceleration = np.clip(action[0], -1.0, 1.0) * self.max_acceleration
        steering_vel = np.clip(action[1], -1.0, 1.0) * self.max_steering_vel

        # 更新前轮转角
        self.steering_angle += steering_vel * self.dt
        self.steering_angle = np.clip(self.steering_angle, -self.max_steering_angle, self.max_steering_angle)

        # 应用物理模型 (自行车模型)
        # 计算车辆旋转中心
        if abs(self.steering_angle) > 1e-5:
            turning_radius = self.wheel_base / np.tan(self.steering_angle)
            angular_velocity = np.linalg.norm(self.agent_vel) / turning_radius
        else:
            angular_velocity = 0.0

        # 限制角速度
        angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)

        # 更新车辆朝向
        self.agent_heading += angular_velocity * self.dt
        self.agent_heading = self.agent_heading % (2 * np.pi)

        # 计算加速度在车辆坐标系下的分量
        forward_acc = acceleration * np.array([np.cos(self.agent_heading), np.sin(self.agent_heading)])

        # 更新速度 (考虑摩擦)
        self.agent_vel = self.agent_vel * (1 - self.friction) + forward_acc * self.dt

        # 限制速度
        speed = np.linalg.norm(self.agent_vel)
        if speed > self.max_velocity:
            self.agent_vel = self.agent_vel * self.max_velocity / speed

        # 更新位置
        new_pos = self.agent_pos + self.agent_vel * self.dt

        # 边界检查 (值域范围[-1,1])
        self.agent_pos = np.clip(new_pos, self.low, self.high)

        # 边界碰撞处理
        boundary_collision = False
        for i in range(2):
            if self.agent_pos[i] <= self.low or self.agent_pos[i] >= self.high:
                self.agent_vel[i] *= -0.5  # 反弹
                boundary_collision = True

        # 更新步数计数器
        self.current_step += 1

        # 更新目标距离
        relative_target_vector_global = self.target - self.agent_pos
        self.target_distance = np.linalg.norm(relative_target_vector_global)

        # 更新最近障碍物
        self._update_nearest_obstacle()

        # 计算奖励
        self.reward = self._calculate_reward(boundary_collision)
        self.total_reward += self.reward

        # 检查终止条件
        terminated = False  # self.target_distance < self.target_threshold
        truncated = self.current_step >= self.max_steps

        # 收集额外信息
        info = {
            "curriculum_level": self.curriculum_level,
            "min_distance": self.min_distance,
            "target_distance": self.target_distance,
            "velocity": np.linalg.norm(self.agent_vel),
            "steering_angle": self.steering_angle
        }

        # 标记回合结束
        if truncated or terminated:
            info["episode"] = {
                "r": self.total_reward,
                "l": self.current_step,
                "curriculum_level": self.curriculum_level
            }
        self.last_action = action

        return self._get_obs(), self.reward, terminated, truncated, info

    def _update_nearest_obstacle(self):
        """更新最近障碍物的位置和距离"""
        if self.current_points == 0:
            self.min_distance = float('inf')
            return

        min_dist = float('inf')
        for i in range(self.current_points):
            dist = np.linalg.norm(self.agent_pos - self.obstacle_abs_positions[i])
            if dist < min_dist:
                min_dist = dist
        self.min_distance = min_dist

    def _calculate_reward(self, boundary_collision: bool) -> float:
        # 目标距离奖励 (主要驱动因素)
        # 环境常量
        MAX_DIST = 2.0  # 最大可能距离（对角线）
        SAFE_RADIUS = 0.2
        COLLISION_RADIUS = 0.05

        # 1. 目标距离奖励
        norm_target_dist = self.target_distance / MAX_DIST
        target_reward = 2 * np.exp(-3.0 * norm_target_dist) - 1  # 5 * np.exp(-20 * norm_target_dist) - 1

        # 组合奖励
        total_reward = target_reward

        return float(np.clip(total_reward, -2.0, 10.0))

    def _get_obs(self) -> np.ndarray:
        """构建观测向量 (所有值在[-1,1]范围内)"""
        # 车辆状态 (归一化)
        norm_heading = self.agent_heading / np.pi - 1  # [0, 2π] -> [-1,1]

        # 将速度转换到车身坐标系
        body_vel = self._global_to_body_frame(self.agent_vel)
        norm_longitudinal_vel = body_vel[0] / self.max_velocity
        norm_lateral_vel = body_vel[1] / self.max_velocity

        norm_steering = self.steering_angle / self.max_steering_angle  # 转向角归一化

        # 障碍物信息 (转换到车身坐标系)
        points_vector_body = np.zeros((self.max_points, 3), dtype=np.float32)
        for i in range(self.max_points):
            # 全局坐标系下的相对向量
            global_vec = self.obstacle_abs_positions[i] - self.agent_pos
            # 转换到车身坐标系
            body_vec = self._global_to_body_frame(global_vec)
            dist = np.linalg.norm(global_vec)

            # 归一化处理
            norm_x = body_vec[0] / 2.0  # 最大可能值2.0 (从-1到1)
            norm_y = body_vec[1] / 2.0
            dist_T = 2 / (dist + 0.5) - 1  # 映射到[-1,1]

            points_vector_body[i] = [norm_x, norm_y, dist_T]

        points_flat = points_vector_body.flatten()

        # 目标信息 (转换到车身坐标系)
        relative_target_global = self.target - self.agent_pos
        relative_target_body = self._global_to_body_frame(relative_target_global)
        self.target_distance = np.linalg.norm(relative_target_global)
        relative_target_body = relative_target_body / self.target_distance  # 前两维使用方向向量

        # 归一化处理
        norm_target_x = relative_target_body[0]
        norm_target_y = relative_target_body[1]
        target_dist_T = 3 / (self.target_distance + 1) - 2  # 映射到[-1,1]

        # 拼接所有观测
        return np.concatenate([
            [norm_heading],  # 1: 车辆朝向
            [norm_longitudinal_vel,  # 2: 纵向速度
             norm_lateral_vel],  # 横向速度
            [norm_steering],  # 1: 偏转角
            self.last_action,  # 2: 上一动作
            points_flat,  # max_points*3: 障碍物信息 (车身坐标系)
            [norm_target_x,  # 3: 目标在车身坐标系的x位置
             norm_target_y,  # 目标在车身坐标系的y位置
             target_dist_T]  # 目标距离归一化
        ], dtype=np.float32)

    def _get_vehicle_corners(self):
        """计算车辆矩形的四个角点"""
        cx, cy = self.agent_pos
        cos_h = np.cos(self.agent_heading)
        sin_h = np.sin(self.agent_heading)

        half_length = self.vehicle_length / 2
        half_width = self.vehicle_width / 2

        corners = []
        for dx, dy in [(-half_length, -half_width),
                       (-half_length, half_width),
                       (half_length, half_width),
                       (half_length, -half_width)]:
            x = cx + dx * cos_h - dy * sin_h
            y = cy + dx * sin_h + dy * cos_h
            corners.append((x, y))

        return corners

    def render(self):
        if self.render_mode != "human":
            return

        self.screen.fill((255, 255, 255))  # 白色背景

        # 坐标转换函数
        def to_screen(pos):
            x = (pos[0] - self.low) * self.scale
            y = (pos[1] - self.low) * self.scale
            return int(x), int(y)

        # 绘制障碍物
        for i in range(self.current_points):
            pygame.draw.circle(self.screen, (255, 0, 0), to_screen(self.obstacle_abs_positions[i]), 5)

        # 绘制目标
        pygame.draw.circle(self.screen, (0, 255, 0), to_screen(self.target), 8)

        # 绘制车辆
        corners = self._get_vehicle_corners()
        screen_corners = [to_screen(c) for c in corners]
        pygame.draw.polygon(self.screen, (0, 0, 255), screen_corners)

        # 绘制车头方向
        front_x = self.agent_pos[0] + (self.vehicle_length / 2) * np.cos(self.agent_heading)
        front_y = self.agent_pos[1] + (self.vehicle_length / 2) * np.sin(self.agent_heading)
        pygame.draw.circle(self.screen, (255, 255, 0), to_screen((front_x, front_y)), 4)

        # 绘制速度向量
        vel_end = self.agent_pos + self.agent_vel * 0.5
        pygame.draw.line(self.screen, (100, 100, 255), to_screen(self.agent_pos), to_screen(vel_end), 2)

        # 绘制前轮方向
        wheel_angle = self.agent_heading + self.steering_angle
        wheel_end_x = self.agent_pos[0] + self.wheel_base * np.cos(wheel_angle)
        wheel_end_y = self.agent_pos[1] + self.wheel_base * np.sin(wheel_angle)
        pygame.draw.line(self.screen, (255, 0, 255), to_screen(self.agent_pos), to_screen((wheel_end_x, wheel_end_y)),
                         2)

        # 绘制车身坐标系
        body_x_end = self.agent_pos + 0.1 * np.array([np.cos(self.agent_heading), np.sin(self.agent_heading)])
        body_y_end = self.agent_pos + 0.1 * np.array([-np.sin(self.agent_heading), np.cos(self.agent_heading)])
        pygame.draw.line(self.screen, (255, 0, 0), to_screen(self.agent_pos), to_screen(body_x_end), 2)  # 红色: 车头方向
        pygame.draw.line(self.screen, (0, 255, 0), to_screen(self.agent_pos), to_screen(body_y_end), 2)  # 绿色: 左侧方向

        # 显示信息
        texts = [
            f"Level: {self.curriculum_level}/{self.max_level}",
            f"Obstacles: {self.current_points}/{self.max_points}",
            f"Steps: {self.current_step}/{self.max_steps}",
            f"Reward: {self.reward:.2f}",
            f"Speed: {np.linalg.norm(self.agent_vel):.2f}/{self.max_velocity:.2f}",
            f"Steering: {np.degrees(self.steering_angle):.1f}°",
            f"Long Vel: {self._global_to_body_frame(self.agent_vel)[0]:.3f}",
            f"Lat Vel: {self._global_to_body_frame(self.agent_vel)[1]:.3f}"
        ]

        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 30))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human":
            pygame.quit()


# 测试环境
if __name__ == "__main__":
    env = CarEnv(render_mode="human")
    obs, _ = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 简单控制策略: 朝向目标前进
        target_dir = env.target - env.agent_pos
        target_angle = np.arctan2(target_dir[1], target_dir[0])
        angle_diff = (target_angle - env.agent_heading + np.pi) % (2 * np.pi) - np.pi

        # 动作: [加速度, 转向]
        acceleration = 1.0  # 全速前进
        steering = np.clip(angle_diff / (np.pi / 4), -1.0, 1.0)  # 归一化转向

        action = np.array([acceleration, steering])
        obs, reward, terminated, truncated, _ = env.step(action)
        print("状态:1朝向", obs[:1], "、2车身速度(纵向,横向)", obs[1:3], "、1偏转角", obs[3:4], "、2上一动作", obs[4:6])
        print("目标:车身系x,y", obs[-3:-1], "、距离归一化", obs[-1])
        env.render()

        if terminated or truncated:
            print(f"回合结束! 总奖励: {env.total_reward:.2f}")
            obs, _ = env.reset()

    env.close()
