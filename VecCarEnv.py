import time
import warnings
import pygame
import numpy as np
import random
import math
from typing import Tuple, Dict, List, Optional, Any
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv


class VectorCarEnv(VectorEnv):
    metadata = {'render_modes': ['human'], "render_fps": 30}

    def __init__(self, num_agents: int = 4, render_mode: Optional[str] = None):
        super().__init__(num_agents, spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        ), spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8 + 3 * 30 + 3,),  # 单个agent的观测维度
            dtype=np.float32
        ))

        self.num_agents = num_agents
        self.render_mode = render_mode

        # 环境参数
        self.low = -1.0
        self.high = 1.0
        self.min_points = 5
        self.max_points = 30
        self.curriculum_level = 0
        self.max_level = 20
        self.min_level = 0

        # 车辆物理参数
        self.max_acceleration = 0.5
        self.max_steering_vel = 1.0
        self.max_steering_angle = np.pi / 6
        self.dt = 0.1
        self.max_velocity = 0.5
        self.max_angular_velocity = 1.0
        self.vehicle_length = 0.1
        self.vehicle_width = 0.05
        self.friction = 0.05
        self.wheel_base = 0.06

        # 环境限制
        self.max_steps = 300
        self.current_step = 0
        self.target_threshold = 0.05

        # 无效点可能的角点位置
        self.corner_points = np.array([
            [self.low, self.low],
            [self.low, self.high],
            [self.high, self.low],
            [self.high, self.high]
        ])

        # 状态初始化
        self.reset()

        # 渲染设置
        self.scale = 400
        if render_mode == "human":
            pygame.init()
            self.screen_size = (int((self.high - self.low) * self.scale),
                                int((self.high - self.low) * self.scale))
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption(f"矢量车辆环境 ({num_agents} 个小车)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)
            # 为每个小车分配不同的颜色
            self.agent_colors = [
                                    (0, 0, 255),  # 蓝色
                                    (255, 0, 0),  # 红色
                                    (0, 255, 0),  # 绿色
                                    (255, 255, 0),  # 黄色
                                    (255, 0, 255),  # 紫色
                                    (0, 255, 255),  # 青色
                                ][:num_agents]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # 重置状态变量
        self.current_step = 0
        self.total_rewards = np.zeros(self.num_agents)
        self.rewards = np.zeros(self.num_agents)

        # 更新课程难度
        self.current_points = int(self.min_points + (self.max_points - self.min_points) *
                                  (self.curriculum_level / self.max_level))

        # 初始化所有小车状态 (形状: [num_agents, 2])
        self.agent_pos = np.random.uniform(self.low + 0.1, self.high - 0.1, size=(self.num_agents, 2))
        self.agent_heading = np.random.uniform(0, 2 * np.pi, size=self.num_agents)
        self.agent_vel = np.zeros((self.num_agents, 2))
        self.steering_angles = np.zeros(self.num_agents)

        # 初始化每个小车的障碍物 (形状: [num_agents, max_points, 2])
        self.obstacle_abs_positions = np.zeros((self.num_agents, self.max_points, 2))
        self.points_vector = np.zeros((self.num_agents, self.max_points, 3))

        # 为每个小车生成障碍物
        for agent_idx in range(self.num_agents):
            # 生成有效随机点
            for i in range(self.current_points):
                point = np.random.uniform(self.low + 0.05, self.high - 0.05, size=2)
                self.obstacle_abs_positions[agent_idx, i] = point
                relative_vector = point - self.agent_pos[agent_idx]
                dist = np.linalg.norm(relative_vector)
                dist_T = 2 / (dist + 0.5) - 1
                self.points_vector[agent_idx, i] = [relative_vector[0], relative_vector[1], dist_T]

            # 填充无效点（角点）
            for i in range(self.current_points, self.max_points):
                corner = self.corner_points[i % len(self.corner_points)]
                self.obstacle_abs_positions[agent_idx, i] = corner
                relative_vector = corner - self.agent_pos[agent_idx]
                self.points_vector[agent_idx, i] = [relative_vector[0], relative_vector[1], -1.0]

        # 初始化每个小车的目标位置 (形状: [num_agents, 2])
        self.targets = np.zeros((self.num_agents, 2))
        self.target_vectors = np.zeros((self.num_agents, 3))
        self.target_distances = np.zeros(self.num_agents)
        self.min_distances = np.full(self.num_agents, float('inf'))

        for agent_idx in range(self.num_agents):
            target = np.random.uniform(self.low + 0.1, self.high - 0.1, size=2)
            # 确保目标不与车辆初始位置太近
            while np.linalg.norm(target - self.agent_pos[agent_idx]) < 0.3:
                target = np.random.uniform(self.low + 0.1, self.high - 0.1, size=2)

            self.targets[agent_idx] = target
            relative_target_vector = target - self.agent_pos[agent_idx]
            target_dist = np.linalg.norm(relative_target_vector)
            target_dist_T = 2 / (target_dist + 0.5) - 1
            self.target_vectors[agent_idx] = [relative_target_vector[0], relative_target_vector[1], target_dist_T]
            self.target_distances[agent_idx] = target_dist

        # 更新最近障碍物
        self._update_nearest_obstacles()

        # 初始化上一个动作
        self.last_actions = np.zeros((self.num_agents, 2))

        return self._get_obs(), {}

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        # 解析动作 (形状: [num_agents, 2])
        accelerations = np.clip(actions[:, 0], -1.0, 1.0) * self.max_acceleration
        steering_vels = np.clip(actions[:, 1], -1.0, 1.0) * self.max_steering_vel

        # 更新前轮转角
        self.steering_angles += steering_vels * self.dt
        self.steering_angles = np.clip(
            self.steering_angles,
            -self.max_steering_angle,
            self.max_steering_angle
        )

        # 边界碰撞标记
        boundary_collisions = np.zeros(self.num_agents, dtype=bool)

        # 为每个小车应用物理模型
        for agent_idx in range(self.num_agents):
            # 计算车辆旋转中心
            if abs(self.steering_angles[agent_idx]) > 1e-5:
                turning_radius = self.wheel_base / np.tan(self.steering_angles[agent_idx])
                angular_velocity = np.linalg.norm(self.agent_vel[agent_idx]) / turning_radius
            else:
                angular_velocity = 0.0

            # 限制角速度
            angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)

            # 更新车辆朝向
            self.agent_heading[agent_idx] = (self.agent_heading[agent_idx] + angular_velocity * self.dt) % (2 * np.pi)

            # 计算加速度在车辆坐标系下的分量
            heading = self.agent_heading[agent_idx]
            forward_acc = accelerations[agent_idx] * np.array([np.cos(heading), np.sin(heading)])

            # 更新速度 (考虑摩擦)
            self.agent_vel[agent_idx] = self.agent_vel[agent_idx] * (1 - self.friction) + forward_acc * self.dt

            # 限制速度
            speed = np.linalg.norm(self.agent_vel[agent_idx])
            if speed > self.max_velocity:
                self.agent_vel[agent_idx] = self.agent_vel[agent_idx] * self.max_velocity / speed

            # 更新位置
            new_pos = self.agent_pos[agent_idx] + self.agent_vel[agent_idx] * self.dt

            # 边界检查
            clipped_pos = np.clip(new_pos, self.low, self.high)
            self.agent_pos[agent_idx] = clipped_pos

            # 边界碰撞处理
            if np.any(new_pos != clipped_pos):
                self.agent_vel[agent_idx] *= -0.5  # 反弹
                boundary_collisions[agent_idx] = True

        # 更新步数计数器
        self.current_step += 1

        # 更新所有点的相对向量和距离倒数
        for agent_idx in range(self.num_agents):
            for i in range(self.max_points):
                relative_vector = self.obstacle_abs_positions[agent_idx, i] - self.agent_pos[agent_idx]
                self.points_vector[agent_idx, i, 0] = relative_vector[0]
                self.points_vector[agent_idx, i, 1] = relative_vector[1]

                # 只更新有效点的距离倒数
                if i < self.current_points:
                    dist = np.linalg.norm(relative_vector)
                    dist_T = 2 / (dist + 0.5) - 1
                    self.points_vector[agent_idx, i, 2] = dist_T

            # 更新目标向量
            relative_target_vector = self.targets[agent_idx] - self.agent_pos[agent_idx]
            self.target_distances[agent_idx] = np.linalg.norm(relative_target_vector)
            target_dist_T = 2 / (self.target_distances[agent_idx] + 0.5) - 1
            self.target_vectors[agent_idx] = [relative_target_vector[0], relative_target_vector[1], target_dist_T]

        # 更新最近障碍物
        self._update_nearest_obstacles()

        # 计算奖励
        self.rewards = self._calculate_rewards(boundary_collisions)
        self.total_rewards += self.rewards

        # 检查终止条件
        terminated = np.zeros(self.num_agents, dtype=bool)  # 可以根据需要修改
        truncated = np.full(self.num_agents, self.current_step >= self.max_steps)

        # 收集额外信息
        info = {
            "curriculum_level": self.curriculum_level,
            "min_distances": self.min_distances,
            "target_distances": self.target_distances,
            "velocities": np.linalg.norm(self.agent_vel, axis=1),
            "steering_angles": self.steering_angles
        }

        # 标记回合结束
        if np.any(truncated) or np.any(terminated):
            info["episode"] = {
                "r": self.total_rewards,
                "l": np.full(self.num_agents, self.current_step),
                "curriculum_level": self.curriculum_level
            }

        self.last_actions = actions

        return self._get_obs(), self.rewards, terminated, truncated, info

    def _update_nearest_obstacles(self):
        """更新每个小车的最近障碍物距离"""
        for agent_idx in range(self.num_agents):
            if self.current_points == 0:
                self.min_distances[agent_idx] = float('inf')
                continue

            min_dist = float('inf')
            for i in range(self.current_points):
                dist = np.linalg.norm(self.agent_pos[agent_idx] - self.obstacle_abs_positions[agent_idx, i])
                if dist < min_dist:
                    min_dist = dist
            self.min_distances[agent_idx] = min_dist

    def _calculate_rewards(self, boundary_collisions: np.ndarray) -> np.ndarray:
        # 目标距离奖励 (主要驱动因素)
        MAX_DIST = 2.0  # 最大可能距离（对角线）
        rewards = np.zeros(self.num_agents)

        for agent_idx in range(self.num_agents):
            norm_target_dist = self.target_distances[agent_idx] / MAX_DIST
            target_reward = 2 * np.exp(-3.0 * norm_target_dist) - 1
            rewards[agent_idx] = target_reward

            # 这里可以添加其他奖励项，如避障惩罚等

        return rewards

    def _get_obs(self) -> np.ndarray:
        """构建观测向量 (所有值在[-1,1]范围内)"""
        # 初始化观测数组 [num_agents, obs_dim]
        obs = np.zeros((self.num_agents, self.observation_space.shape[0]), dtype=np.float32)

        for agent_idx in range(self.num_agents):
            # 车辆状态 (归一化)
            norm_pos = self.agent_pos[agent_idx]  # 位置已在[-1,1]内
            norm_heading = self.agent_heading[agent_idx] / np.pi - 1  # [0, 2π] -> [-1,1]
            norm_vel = self.agent_vel[agent_idx] / self.max_velocity  # 速度归一化
            norm_steering = self.steering_angles[agent_idx] / self.max_steering_angle  # 转向角归一化

            # 障碍物信息 (已归一化)
            points_flat = self.points_vector[agent_idx].flatten()

            # 目标信息 (已归一化)
            target_vec = self.target_vectors[agent_idx]

            # 拼接所有观测
            obs[agent_idx] = np.concatenate([
                norm_pos,  # 2
                [norm_heading],  # 1
                norm_vel,  # 2
                [norm_steering],  # 1
                self.last_actions[agent_idx],  # 2
                points_flat,  # 30*3
                target_vec  # 3
            ])

        return obs

    def _get_vehicle_corners(self, agent_idx: int):
        """计算车辆矩形的四个角点"""
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

        # 坐标转换函数
        def to_screen(pos):
            x = (pos[0] - self.low) * self.scale
            y = (pos[1] - self.low) * self.scale
            return int(x), int(y)

        # 绘制所有小车
        for agent_idx in range(self.num_agents):
            # 绘制障碍物 (每个小车有自己的障碍物)
            for i in range(self.current_points):
                pygame.draw.circle(
                    self.screen,
                    (200, 200, 200),  # 灰色障碍物
                    to_screen(self.obstacle_abs_positions[agent_idx, i]),
                    5
                )

            # 绘制目标
            pygame.draw.circle(
                self.screen,
                (0, 255, 0),  # 绿色目标
                to_screen(self.targets[agent_idx]),
                8
            )

            # 绘制车辆
            corners = self._get_vehicle_corners(agent_idx)
            screen_corners = [to_screen(c) for c in corners]
            pygame.draw.polygon(
                self.screen,
                self.agent_colors[agent_idx],
                screen_corners
            )

            # 绘制车头方向
            front_x = self.agent_pos[agent_idx, 0] + (self.vehicle_length / 2) * np.cos(self.agent_heading[agent_idx])
            front_y = self.agent_pos[agent_idx, 1] + (self.vehicle_length / 2) * np.sin(self.agent_heading[agent_idx])
            pygame.draw.circle(
                self.screen,
                (255, 255, 0),
                to_screen((front_x, front_y)),
                4
            )

            # 绘制速度向量
            vel_end = self.agent_pos[agent_idx] + self.agent_vel[agent_idx] * 0.5
            pygame.draw.line(
                self.screen,
                (100, 100, 255),
                to_screen(self.agent_pos[agent_idx]),
                to_screen(vel_end),
                2
            )

            # 绘制前轮方向
            wheel_angle = self.agent_heading[agent_idx] + self.steering_angles[agent_idx]
            wheel_end_x = self.agent_pos[agent_idx, 0] + self.wheel_base * np.cos(wheel_angle)
            wheel_end_y = self.agent_pos[agent_idx, 1] + self.wheel_base * np.sin(wheel_angle)
            pygame.draw.line(
                self.screen,
                (255, 0, 255),
                to_screen(self.agent_pos[agent_idx]),
                to_screen((wheel_end_x, wheel_end_y)),
                2
            )

        # 显示信息
        texts = [
            f"Agents: {self.num_agents}",
            f"Level: {self.curriculum_level}/{self.max_level}",
            f"Steps: {self.current_step}/{self.max_steps}",
            f"Total Reward: {np.sum(self.total_rewards):.2f}",
            f"Avg Distance: {np.mean(self.target_distances):.2f}"
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
    num_agents = 4  # 可以修改并行小车数量
    env = VectorCarEnv(num_agents=num_agents, render_mode="human")
    obs, _ = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 为每个小车生成随机动作
        actions = np.zeros((num_agents, 2))
        for agent_idx in range(num_agents):
            # 简单控制策略: 朝向目标前进
            target_dir = env.targets[agent_idx] - env.agent_pos[agent_idx]
            target_angle = np.arctan2(target_dir[1], target_dir[0])
            angle_diff = (target_angle - env.agent_heading[agent_idx] + np.pi) % (2 * np.pi) - np.pi

            # 动作: [加速度, 转向]
            acceleration = 1.0  # 全速前进
            steering = np.clip(angle_diff / (np.pi / 4), -1.0, 1.0)  # 归一化转向

            actions[agent_idx] = [acceleration, steering]

        obs, rewards, terminated, truncated, _ = env.step(actions)
        env.render()

        if np.any(terminated) or np.any(truncated):
            print(f"回合结束! 总奖励: {env.total_rewards}")
            obs, _ = env.reset()

    env.close()
