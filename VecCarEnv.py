import time
import warnings
import pygame
import numpy as np
import random
import math
from typing import Tuple, Dict, List, Optional
import gymnasium as gym
from gymnasium import spaces
import colorsys


class ParkingEnv(gym.Env):  # 更名为ParkingEnv更贴切
    metadata = {'render_modes': ['human'], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, num_agents: int = 3):
        super().__init__()
        self.num_agents = num_agents

        # 真实场景参数 (30m x 30m)
        self.world_size = 30.0  # 米
        self.low = 0.0
        self.high = self.world_size

        # 环境参数
        self.total_rewards = [0] * num_agents
        self.rewards = [0] * num_agents
        self.agent_pos = [None] * num_agents
        self.agent_heading = [None] * num_agents
        self.agent_vel = [None] * num_agents
        self.steering_angles = [0.0] * num_agents
        self.obstacle_abs_positions = None
        self.points_vector = None
        self.parking_slot = None  # 停车位信息 (位置, 方向, 尺寸)
        self.target_vectors = [None] * num_agents
        self.min_distances = [float('inf')] * num_agents
        self.target_distances = [float('inf')] * num_agents
        self.last_actions = [np.zeros(2)] * num_agents
        self.current_points = None
        self.min_points = 5
        self.max_points = 30
        self.curriculum_level = 0
        self.max_level = 20
        self.min_level = 0

        # 真实车辆物理参数 (泊车场景)
        self.max_acceleration = 1.0  # m/s² (泊车时加速度较小)
        self.max_steering_vel = 0.5  # rad/s (约28.6度/秒)
        self.max_steering_angle = np.pi / 4  # 45度最大转角
        self.dt = 0.1  # 时间步长
        self.max_velocity = 1.5  # m/s (约5.4km/h，典型泊车速度)
        self.max_angular_velocity = 1.0  # rad/s
        self.vehicle_length = 4.8  # 米 (典型轿车长度)
        self.vehicle_width = 1.8  # 米 (典型轿车宽度)
        self.friction = 0.1  # 摩擦系数
        self.wheel_base = 2.8  # 轴距 (米)

        # 停车位参数
        self.parking_length = 5.5  # 停车位长度
        self.parking_width = 2.5  # 停车位宽度

        # 环境限制
        self.max_steps = 300
        self.current_step = 0
        self.target_threshold = 0.5  # 到达目标的距离阈值 (米)
        self.heading_threshold = 0.2  # 方向误差阈值 (弧度)

        # 动作空间: [前向加速度, 转向角速度] * num_agents
        self.action_space = spaces.Box(
            low=np.tile(np.array([-1.0, -1.0], dtype=np.float32), (num_agents, 1)),
            high=np.tile(np.array([1.0, 1.0], dtype=np.float32), (num_agents, 1)),
            dtype=np.float32
        )

        # 观测空间设计 (所有值都在[-1,1]范围内)
        # 增加停车方向差值观测
        obs_dim = 6 + 3 * self.max_points + 4  # 车辆状态4(朝向1、速度2、方向偏角1) + last动作2 + 障碍物信息 + 停车位信息
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_agents, obs_dim),
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
        self.scale = 20  # 比例因子 (30米场景 => 600像素，20像素/米)
        if render_mode == "human":
            pygame.init()
        screen_size = int(self.world_size * self.scale)
        self.screen_size = (screen_size, screen_size)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("多车辆泊车环境")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        # 为每个车辆生成独特的颜色 (HSV均匀分布)
        self.agent_colors = []
        for i in range(num_agents):
            hue = i / num_agents
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            color = (int(r * 255), int(g * 255), int(b * 255))
            self.agent_colors.append(color)

        # 停车位颜色
        self.parking_color = (100, 100, 100)  # 灰色

        # 环境状态初始化
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # 重置状态变量
        self.current_step = 0
        self.total_rewards = [0] * self.num_agents
        self.rewards = [0] * self.num_agents
        self.min_distances = [float('inf')] * self.num_agents

        # 更新课程难度
        self.current_points = int(self.min_points + (self.max_points - self.min_points) *
                                  (self.curriculum_level / self.max_level))

        # 初始化障碍物
        self.obstacle_abs_positions = []
        self.points_vector = np.zeros((self.max_points, 3), dtype=np.float32)

        # 生成有效随机点 (障碍物)
        for i in range(self.current_points):
            point = self.np_random.uniform(self.low + 2.0, self.high - 2.0, size=2)
            self.obstacle_abs_positions.append(point)
            relative_vector = point - np.array([self.world_size / 2, self.world_size / 2])
            dist = np.linalg.norm(relative_vector)
            dist_T = 2 / (dist + 0.5) - 1
            self.points_vector[i] = [relative_vector[0], relative_vector[1], dist_T]

        # 填充无效点（角点）
        for i in range(self.current_points, self.max_points):
            corner = self.corner_points[i % len(self.corner_points)]
            self.obstacle_abs_positions.append(corner)
            relative_vector = corner - np.array([self.world_size / 2, self.world_size / 2])
            self.points_vector[i] = [relative_vector[0], relative_vector[1], -1.0]

        # 创建共享停车位 (位于场景中心)
        parking_center = np.array([self.world_size / 2, self.world_size / 2])
        parking_angle = self.np_random.uniform(0, 2 * np.pi)
        self.parking_slot = {
            'center': parking_center,
            'angle': parking_angle,
            'corners': self._get_parking_corners(parking_center, parking_angle)
        }

        # 为每辆车初始化状态
        for i in range(self.num_agents):
            # 重置车辆状态 - 初始位置在场景边缘
            edge = self.np_random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                self.agent_pos[i] = np.array([
                    self.np_random.uniform(2.0, self.high - 2.0),
                    self.np_random.uniform(2.0, 5.0)
                ])
            elif edge == 'bottom':
                self.agent_pos[i] = np.array([
                    self.np_random.uniform(2.0, self.high - 2.0),
                    self.np_random.uniform(self.high - 5.0, self.high - 2.0)
                ])
            elif edge == 'left':
                self.agent_pos[i] = np.array([
                    self.np_random.uniform(2.0, 5.0),
                    self.np_random.uniform(2.0, self.high - 2.0)
                ])
            else:  # right
                self.agent_pos[i] = np.array([
                    self.np_random.uniform(self.high - 5.0, self.high - 2.0),
                    self.np_random.uniform(2.0, self.high - 2.0)
                ])

            # 初始朝向朝向停车位
            target_dir = self.parking_slot['center'] - self.agent_pos[i]
            self.agent_heading[i] = np.arctan2(target_dir[1], target_dir[0])

            self.agent_vel[i] = np.zeros(2)
            self.steering_angles[i] = 0.0
            self.last_actions[i] = np.zeros(2)

            # 更新目标向量
            self._update_target_vector(i)

        # 更新最近障碍物
        self._update_nearest_obstacle()

        return self._get_obs(), {}

    def _get_parking_corners(self, center, angle):
        """计算停车位的四个角点"""
        half_length = self.parking_length / 2
        half_width = self.parking_width / 2

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        corners = []
        for dx, dy in [(-half_length, -half_width),
                       (-half_length, half_width),
                       (half_length, half_width),
                       (half_length, -half_width)]:
            x = center[0] + dx * cos_a - dy * sin_a
            y = center[1] + dx * sin_a + dy * cos_a
            corners.append((x, y))

        return corners

    def _update_level(self):
        """根据智能体表现更新课程难度"""
        avg_reward = sum(self.total_rewards) / self.num_agents
        if avg_reward >= 300:
            self.curriculum_level = min(self.curriculum_level + 1, self.max_level)
        elif avg_reward <= 0:
            self.curriculum_level = max(self.curriculum_level - 1, self.min_level)

    def _update_target_vector(self, agent_idx):
        """更新车辆到停车位的向量"""
        # 停车位中心向量
        center_vector = self.parking_slot['center'] - self.agent_pos[agent_idx]
        center_dist = np.linalg.norm(center_vector)
        center_vector = center_vector / center_dist if center_dist > 0.01 else np.zeros(2)
        center_dist_T = 3 / (center_dist + 1) - 2

        # 方向差值 (车辆当前朝向与停车位朝向的差值)
        heading_diff = (self.agent_heading[agent_idx] - self.parking_slot['angle'] + np.pi) % (2 * np.pi) - np.pi
        norm_heading_diff = heading_diff / np.pi  # 归一化到[-1,1]

        # 存储目标向量
        self.target_vectors[agent_idx] = np.array([
            center_vector[0],
            center_vector[1],
            center_dist_T,
            norm_heading_diff  # 新增的方向差值
        ])
        self.target_distances[agent_idx] = center_dist

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, List[float], bool, bool, dict]:
        rewards = [0] * self.num_agents
        terminated = False

        # 对每辆车分别执行动作
        for i in range(self.num_agents):
            action = actions[i]
            # 解析动作 (值域[-1,1])
            acceleration = np.clip(action[0], -1.0, 1.0) * self.max_acceleration
            steering_vel = np.clip(action[1], -1.0, 1.0) * self.max_steering_vel

            # 更新前轮转角
            self.steering_angles[i] += steering_vel * self.dt
            self.steering_angles[i] = np.clip(
                self.steering_angles[i],
                -self.max_steering_angle,
                self.max_steering_angle
            )

            # 应用物理模型 (自行车模型)
            if abs(self.steering_angles[i]) > 1e-5:
                turning_radius = self.wheel_base / np.tan(self.steering_angles[i])
                angular_velocity = np.linalg.norm(self.agent_vel[i]) / turning_radius
            else:
                angular_velocity = 0.0

            # 限制角速度
            angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)

            # 更新车辆朝向
            self.agent_heading[i] += angular_velocity * self.dt
            self.agent_heading[i] = self.agent_heading[i] % (2 * np.pi)

            # 计算加速度在车辆坐标系下的分量
            forward_acc = acceleration * np.array([np.cos(self.agent_heading[i]), np.sin(self.agent_heading[i])])

            # 更新速度 (考虑摩擦)
            self.agent_vel[i] = self.agent_vel[i] * (1 - self.friction) + forward_acc * self.dt

            # 限制速度
            speed = np.linalg.norm(self.agent_vel[i])
            if speed > self.max_velocity:
                self.agent_vel[i] = self.agent_vel[i] * self.max_velocity / speed

            # 更新位置
            new_pos = self.agent_pos[i] + self.agent_vel[i] * self.dt

            # 边界检查
            self.agent_pos[i] = np.clip(new_pos, self.low, self.high)

            # 边界碰撞处理 (轻微反弹)
            boundary_collision = False
            for j in range(2):
                if self.agent_pos[i][j] <= self.low or self.agent_pos[i][j] >= self.high:
                    self.agent_vel[i][j] *= -0.3  # 轻微反弹
                    boundary_collision = True

            # 更新目标向量
            self._update_target_vector(i)

            # 计算奖励
            rewards[i] = self._calculate_reward(i, boundary_collision)
            self.total_rewards[i] += rewards[i]

            # 存储最后动作
            self.last_actions[i] = action

        # 更新步数计数器
        self.current_step += 1

        # 更新最近障碍物
        self._update_nearest_obstacle()

        # 检查终止条件 (所有车辆都成功停车)
        all_parked = True
        for i in range(self.num_agents):
            # 检查是否在停车位内且方向正确
            if (self.target_distances[i] > self.target_threshold or
                    abs(self.target_vectors[i][3] * np.pi) > self.heading_threshold):
                all_parked = False
                break

        terminated = all_parked
        truncated = self.current_step >= self.max_steps

        # 收集额外信息
        info = {
            "curriculum_level": self.curriculum_level,
            "min_distances": self.min_distances,
            "target_distances": self.target_distances,
            "velocities": [np.linalg.norm(vel) for vel in self.agent_vel],
            "steering_angles": self.steering_angles
        }

        # 标记回合结束
        if truncated or terminated:
            info["episode"] = {
                "r": self.total_rewards,
                "l": self.current_step,
                "curriculum_level": self.curriculum_level
            }
            # 更新课程难度
            self._update_level()

        return self._get_obs(), rewards, terminated, truncated, info

    def _update_nearest_obstacle(self):
        """更新每辆车到最近障碍物的距离"""
        for i in range(self.num_agents):
            min_dist = float('inf')
            for point in self.obstacle_abs_positions[:self.current_points]:
                dist = np.linalg.norm(self.agent_pos[i] - point)
                if dist < min_dist:
                    min_dist = dist
            self.min_distances[i] = min_dist

    def _calculate_reward(self, agent_idx: int, boundary_collision: bool) -> float:
        # 目标距离奖励
        MAX_DIST = self.world_size * np.sqrt(2)  # 最大对角线距离
        norm_target_dist = self.target_distances[agent_idx] / MAX_DIST
        target_reward = 2 * np.exp(-3.0 * norm_target_dist) - 1

        # 方向对齐奖励 (新增)
        heading_diff = abs(self.target_vectors[agent_idx][3])  # 归一化的方向差绝对值
        alignment_reward = 0.5 * (1 - heading_diff)  # 方向越接近奖励越高

        # 障碍物惩罚
        safe_radius = 2.0  # 米
        collision_radius = 0.5  # 米
        min_dist = self.min_distances[agent_idx]

        if min_dist < safe_radius:
            norm_dist = max(0, (min_dist - collision_radius)) / (safe_radius - collision_radius)
            obstacle_penalty = -1.0 * (1 - norm_dist) ** 2
        else:
            obstacle_penalty = 0.0

        # 成功停车奖励
        if (self.target_distances[agent_idx] < self.target_threshold and
                abs(self.target_vectors[agent_idx][3] * np.pi) < self.heading_threshold):
            success_bonus = 10.0
        else:
            success_bonus = 0.0

        # 时间惩罚
        time_penalty = -0.01

        # 边界碰撞惩罚
        boundary_penalty = -0.5 if boundary_collision else 0.0

        # 速度惩罚 (鼓励低速泊车)
        speed = np.linalg.norm(self.agent_vel[agent_idx])
        speed_penalty = -0.1 * (speed / self.max_velocity) if speed > 0.5 else 0.0

        # 组合奖励
        total_reward = (
                target_reward +
                alignment_reward +  # 新增的方向奖励
                obstacle_penalty +
                success_bonus +
                time_penalty +
                boundary_penalty +
                speed_penalty
        )

        return float(np.clip(total_reward, -2.0, 10.0))

    def _get_obs(self) -> np.ndarray:
        """构建所有车辆的观测向量"""
        observations = []

        for i in range(self.num_agents):
            # 车辆状态 (归一化)
            # norm_pos = self.agent_pos[i] / self.world_size * 2 - 1  # [0,30] -> [-1,1] 位置
            norm_heading = self.agent_heading[i] / np.pi - 1  # 朝向
            norm_vel = self.agent_vel[i] / self.max_velocity  # 速度
            norm_steering = self.steering_angles[i] / self.max_steering_angle  # 方向舵

            # 更新障碍物相对向量 (基于当前车辆位置)
            points_vector = np.zeros((self.max_points, 3), dtype=np.float32)
            for j in range(self.max_points):
                relative_vector = self.obstacle_abs_positions[j] - self.agent_pos[i]
                points_vector[j, 0] = relative_vector[0] / self.world_size * 2  # 归一化
                points_vector[j, 1] = relative_vector[1] / self.world_size * 2  # 归一化

                if j < self.current_points:
                    dist = np.linalg.norm(relative_vector)
                    dist_T = 2 / (dist + 0.5) - 1
                    points_vector[j, 2] = dist_T
                else:
                    points_vector[j, 2] = -1.0

            # 停车位向量 (已包含方向差值)
            target_vec = self.target_vectors[i].copy()
            target_vec[0] /= self.world_size / 2  # 位置归一化
            target_vec[1] /= self.world_size / 2  # 位置归一化

            # 构建单辆车的观测
            agent_obs = np.concatenate([
                # norm_pos,           #
                [norm_heading],  # 1
                norm_vel,  # 2
                [norm_steering],  # 1
                self.last_actions[i],  # 2
                points_vector.flatten(),  # 30*3
                target_vec  # 大小是 4（方向向量2，归一化距离，目标角度差）
            ], dtype=np.float32)

            observations.append(agent_obs)

        return np.array(observations, dtype=np.float32)

    def _get_vehicle_corners(self, agent_idx: int):
        """计算指定车辆矩形的四个角点"""
        cx, cy = self.agent_pos[agent_idx]
        heading = self.agent_heading[agent_idx]
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

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

        # 坐标转换函数 (米到像素)
        def to_screen(pos):
            x = pos[0] * self.scale
            y = pos[1] * self.scale
            return int(x), int(y)

        # 绘制障碍物 (红色)
        for i in range(self.current_points):
            pygame.draw.circle(self.screen, (255, 0, 0), to_screen(self.obstacle_abs_positions[i]),
                               int(0.5 * self.scale))

        # 绘制停车位 (灰色矩形)
        parking_corners = [to_screen(corner) for corner in self.parking_slot['corners']]
        pygame.draw.polygon(self.screen, self.parking_color, parking_corners)

        # 绘制停车位方向指示 (白色箭头)
        center = to_screen(self.parking_slot['center'])
        arrow_end = (
            center[0] + 1.5 * self.scale * np.cos(self.parking_slot['angle']),
            center[1] + 1.5 * self.scale * np.sin(self.parking_slot['angle'])
        )
        pygame.draw.line(self.screen, (255, 255, 255), center, arrow_end, 3)
        pygame.draw.circle(self.screen, (255, 255, 255), center, 5)

        # 绘制所有车辆
        for i in range(self.num_agents):
            color = self.agent_colors[i]

            # 绘制车辆
            corners = self._get_vehicle_corners(i)
            screen_corners = [to_screen(c) for c in corners]
            pygame.draw.polygon(self.screen, color, screen_corners)

            # 添加黑色轮廓使车辆更清晰
            pygame.draw.polygon(self.screen, (0, 0, 0), screen_corners, 2)

            # 绘制车头方向
            front_x = self.agent_pos[i][0] + (self.vehicle_length / 2) * np.cos(self.agent_heading[i])
            front_y = self.agent_pos[i][1] + (self.vehicle_length / 2) * np.sin(self.agent_heading[i])
            pygame.draw.circle(self.screen, (255, 255, 0), to_screen((front_x, front_y)), 4)

            # 绘制速度向量
            vel_end = self.agent_pos[i] + self.agent_vel[i] * 0.5
            pygame.draw.line(self.screen, (100, 100, 255), to_screen(self.agent_pos[i]), to_screen(vel_end), 2)

            # 绘制前轮方向
            wheel_angle = self.agent_heading[i] + self.steering_angles[i]
            wheel_end_x = self.agent_pos[i][0] + self.wheel_base * np.cos(wheel_angle)
            wheel_end_y = self.agent_pos[i][1] + self.wheel_base * np.sin(wheel_angle)
            pygame.draw.line(self.screen, (255, 0, 255),
                             to_screen(self.agent_pos[i]),
                             to_screen((wheel_end_x, wheel_end_y)), 2)

            # 在车辆上方显示编号
            text_surface = self.font.render(str(i + 1), True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=to_screen(self.agent_pos[i]))
            self.screen.blit(text_surface, text_rect)

            # 显示车辆到停车位的距离和方向差
            dist_text = f"{self.target_distances[i]:.1f}m"
            dir_text = f"{abs(self.target_vectors[i][3] * 180):.0f}°"  # 角度差
            dist_surface = self.font.render(dist_text, True, color)
            dir_surface = self.font.render(dir_text, True, color)

            # 在车辆上方显示信息
            pos = to_screen(self.agent_pos[i])
            self.screen.blit(dist_surface, (pos[0] + 15, pos[1] - 30))
            self.screen.blit(dir_surface, (pos[0] + 15, pos[1] - 50))

        # 显示全局信息
        texts = [
            f"Level: {self.curriculum_level}/{self.max_level}",
            f"Vehicles: {self.num_agents}",
            f"Obstacles: {self.current_points}",
            f"Steps: {self.current_step}/{self.max_steps}",
            f"Avg Reward: {sum(self.rewards) / self.num_agents if self.num_agents > 0 else 0:.2f}",
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
    env = ParkingEnv(render_mode="human", num_agents=100)
    obs, _ = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 为每辆车生成随机动作
        actions = []
        for i in range(env.num_agents):
            # 简单控制策略: 朝向停车位前进
            target_dir = env.parking_slot['center'] - env.agent_pos[i]
            target_angle = np.arctan2(target_dir[1], target_dir[0])
            angle_diff = (target_angle - env.agent_heading[i] + np.pi) % (2 * np.pi) - np.pi

            # 根据距离调整速度
            distance = np.linalg.norm(target_dir)
            if distance > 10:
                acceleration = 1.0  # 全速前进
            elif distance > 3:
                acceleration = 0.5  # 中速
            else:
                acceleration = 0.2  # 低速接近

            steering = np.clip(angle_diff / (np.pi / 4), -1.0, 1.0)  # 归一化转向

            actions.append(np.array([acceleration, steering]))

        obs, rewards, terminated, truncated, _ = env.step(np.array(actions))
        print("状态:1朝向", obs[0][:1], "、2速度", obs[0][1:3], "、1偏转角", obs[0][3:4], "、2上一动作", obs[0][4:6])
        print("目标:2方向", obs[0][-4:-2], "、1距离", obs[0][-2:-1], "、1角度差", obs[0][-1])
        env.render()

        if terminated:
            print(f"所有车辆成功停车! 车辆奖励: {env.total_rewards}")
            obs, _ = env.reset()
        elif truncated:
            print(f"步数限制达到! 车辆奖励: {env.total_rewards}")
            obs, _ = env.reset()

    env.close()
