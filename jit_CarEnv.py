import time
import warnings
import pygame
import numpy as np
import random
import math
from typing import Tuple, Dict, List, Optional
import gymnasium as gym
from gymnasium import spaces
from numba import jit, float32, float64, njit

from numba import njit
import numpy as np


@njit(cache=True)
def clip_scalar(x, lower, upper):
    return min(max(x, lower), upper)


@njit(cache=True)
def calculate_reward(agent_pos, agent_heading, target, target_distance, boundary_collision):
    MAX_DIST = 2.0
    norm_target_dist = target_distance / MAX_DIST
    target_reward = 2 * np.exp(-3.0 * norm_target_dist) - 1

    # 朝向奖励
    target_dir_x = target[0] - agent_pos[0]
    target_dir_y = target[1] - agent_pos[1]
    target_angle = np.arctan2(target_dir_y, target_dir_x)
    angle_diff = np.abs((agent_heading - target_angle + np.pi) % (2 * np.pi) - np.pi)
    alignment_reward = 0.1 * (1 - angle_diff / np.pi)

    collision_penalty = -1.0 if boundary_collision else 0.0

    total_reward = target_reward + alignment_reward + collision_penalty
    total_reward = clip_scalar(total_reward, -2.0, 10.0)
    return total_reward


@njit(cache=True)
def global_to_body_frame(heading, vector):
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    x = vector[0] * cos_h + vector[1] * sin_h
    y = -vector[0] * sin_h + vector[1] * cos_h
    return np.array([x, y])


@njit(cache=True)
def normalize_heading(heading):
    return heading / np.pi - 1


@njit(cache=True)
def distance_to_normalized(dist):
    return 2 / (dist + 0.5) - 1


@njit(cache=True)
def target_distance_normalized(dist):
    return 3 / (dist + 1) - 2


@njit(cache=True)
def calculate_min_distance(agent_pos, obstacle_positions, current_points):
    min_dist = np.inf
    for i in range(current_points):
        dx = agent_pos[0] - obstacle_positions[i, 0]
        dy = agent_pos[1] - obstacle_positions[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < min_dist:
            min_dist = dist
    return min_dist


@njit(cache=True)
def update_physics(
        agent_pos, agent_heading, agent_vel,
        steering_angle, action, dt,
        max_acceleration, max_steering_vel,
        max_steering_angle, wheel_base,
        max_angular_velocity, friction,
        max_velocity, low, high):
    acceleration = clip_scalar(action[0], -1.0, 1.0) * max_acceleration
    steering_vel = clip_scalar(action[1], -1.0, 1.0) * max_steering_vel

    new_steering_angle = steering_angle + steering_vel * dt
    new_steering_angle = clip_scalar(new_steering_angle, -max_steering_angle, max_steering_angle)

    speed = np.sqrt(agent_vel[0] ** 2 + agent_vel[1] ** 2)
    if np.abs(new_steering_angle) > 1e-5:
        turning_radius = wheel_base / np.tan(new_steering_angle)
        angular_velocity = speed / turning_radius
    else:
        angular_velocity = 0.0

    angular_velocity = clip_scalar(angular_velocity, -max_angular_velocity, max_angular_velocity)

    new_heading = (agent_heading + angular_velocity * dt) % (2 * np.pi)

    cos_h = np.cos(new_heading)
    sin_h = np.sin(new_heading)
    forward_acc = acceleration * np.array([cos_h, sin_h])

    new_vel = agent_vel * (1 - friction) + forward_acc * dt

    new_speed = np.sqrt(new_vel[0] ** 2 + new_vel[1] ** 2)
    if new_speed > max_velocity:
        new_vel = new_vel * (max_velocity / new_speed)

    new_pos = agent_pos + new_vel * dt

    boundary_collision = False
    for i in range(2):
        if new_pos[i] < low:
            new_pos[i] = low
            new_vel[i] *= -0.5
            boundary_collision = True
        elif new_pos[i] > high:
            new_pos[i] = high
            new_vel[i] *= -0.5
            boundary_collision = True

    return new_pos, new_heading, new_vel, new_steering_angle, boundary_collision


class CarEnv(gym.Env):
    metadata = {'render_modes': ['human'], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        # 环境参数 - 值域范围[-1,1]
        self.total_reward = 0.0
        self.reward = 0.0
        self.agent_pos = np.zeros(2, dtype=np.float64)
        self.agent_heading = 0.0
        self.agent_vel = np.zeros(2, dtype=np.float64)
        self.steering_angle = 0.0
        self.obstacle_abs_positions = np.zeros((30, 2), dtype=np.float64)  # 预分配内存
        self.target = np.zeros(2, dtype=np.float64)
        self.min_distance = 0.0
        self.target_distance = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.current_points = 5
        self.low = -1.0
        self.high = 1.0
        self.min_points = 5
        self.max_points = 30
        self.curriculum_level = 0
        self.max_level = 20
        self.min_level = 0

        # 车辆物理参数（使用float64提高精度）
        self.max_acceleration = 0.2
        self.max_steering_vel = 0.3
        self.max_steering_angle = np.pi / 6
        self.dt = 0.1
        self.max_velocity = 0.2
        self.max_angular_velocity = 1.0
        self.vehicle_length = 0.1
        self.vehicle_width = 0.05
        self.friction = 0.05
        self.wheel_base = 0.06

        # 环境限制
        self.max_steps = 300
        self.current_step = 0
        self.target_threshold = 0.05

        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 观测空间设计 (所有值都在[-1,1]范围内)
        obs_shape = 1 + 2 + 1 + 2 + 3 * self.max_points + 3
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_shape,),
            dtype=np.float32
        )

        # 预计算角点位置
        self.corner_points = np.array([
            [self.low, self.low],
            [self.low, self.high],
            [self.high, self.low],
            [self.high, self.high]
        ], dtype=np.float64)

        # 渲染设置
        self.render_mode = render_mode
        self.scale = 400
        if render_mode == "human":
            pygame.init()
            self.screen_size = (int((self.high - self.low) * self.scale),
                                int((self.high - self.low) * self.scale))
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("车辆强化学习环境")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)
        else:
            self.screen = None
            self.clock = None
            self.font = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # 重置状态变量
        self.current_step = 0
        self.total_reward = 0.0
        self.reward = 0.0

        # 更新课程难度
        self.current_points = int(self.min_points + (self.max_points - self.min_points) *
                                  (self.curriculum_level / self.max_level))

        # 重置车辆状态
        self.agent_pos[0] = self.np_random.uniform(self.low + 0.1, self.high - 0.1)
        self.agent_pos[1] = self.np_random.uniform(self.low + 0.1, self.high - 0.1)
        self.agent_heading = self.np_random.uniform(0, 2 * np.pi)
        self.agent_vel[0] = 0.0
        self.agent_vel[1] = 0.0
        self.steering_angle = 0.0

        # 生成有效随机障碍物
        for i in range(self.current_points):
            self.obstacle_abs_positions[i, 0] = self.np_random.uniform(self.low + 0.05, self.high - 0.05)
            self.obstacle_abs_positions[i, 1] = self.np_random.uniform(self.low + 0.05, self.high - 0.05)

        # 填充无效点（角点）
        for i in range(self.current_points, self.max_points):
            idx = i % 4
            self.obstacle_abs_positions[i] = self.corner_points[idx]

        # 重置目标位置
        self.target[0] = self.np_random.uniform(self.low + 0.1, self.high - 0.1)
        self.target[1] = self.np_random.uniform(self.low + 0.1, self.high - 0.1)

        # 确保目标不与车辆初始位置太近
        while np.linalg.norm(self.target - self.agent_pos) < 0.3:
            self.target[0] = self.np_random.uniform(self.low + 0.1, self.high - 0.1)
            self.target[1] = self.np_random.uniform(self.low + 0.1, self.high - 0.1)

        # 重置距离记录
        self.min_distance = calculate_min_distance(self.agent_pos, self.obstacle_abs_positions, self.current_points)
        self.target_distance = np.linalg.norm(self.target - self.agent_pos)
        self.last_action[0] = 0.0
        self.last_action[1] = 0.0

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 更新物理状态
        (self.agent_pos, self.agent_heading, self.agent_vel, self.steering_angle, boundary_collision
         ) = update_physics(
            self.agent_pos,
            self.agent_heading,
            self.agent_vel,
            self.steering_angle,
            action,
            self.dt,
            self.max_acceleration,
            self.max_steering_vel,
            self.max_steering_angle,
            self.wheel_base,
            self.max_angular_velocity,
            self.friction,
            self.max_velocity,
            self.low,
            self.high
        )

        # 更新步数计数器
        self.current_step += 1

        # 更新距离
        dx = self.target[0] - self.agent_pos[0]
        dy = self.target[1] - self.agent_pos[1]
        self.target_distance = math.sqrt(dx * dx + dy * dy)
        self.min_distance = calculate_min_distance(self.agent_pos, self.obstacle_abs_positions, self.current_points)

        # 计算奖励
        self.reward = calculate_reward(
            self.agent_pos, self.agent_heading,
            self.target, self.target_distance,
            boundary_collision
        )

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

        self.last_action[0] = action[0]
        self.last_action[1] = action[1]

        return self._get_obs(), self.reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        # 车辆状态
        norm_heading = normalize_heading(self.agent_heading)

        # 速度转换到车身坐标系
        body_vel = global_to_body_frame(self.agent_heading, self.agent_vel)
        norm_longitudinal_vel = body_vel[0] / self.max_velocity
        norm_lateral_vel = body_vel[1] / self.max_velocity

        norm_steering = self.steering_angle / self.max_steering_angle

        # 障碍物信息
        points_vector_body = np.zeros((self.max_points, 3), dtype=np.float32)
        for i in range(self.max_points):
            global_vec = self.obstacle_abs_positions[i] - self.agent_pos
            body_vec = global_to_body_frame(self.agent_heading, global_vec)
            dist = math.sqrt(global_vec[0] * global_vec[0] + global_vec[1] * global_vec[1])

            # 归一化处理
            norm_x = body_vec[0] / 2.0
            norm_y = body_vec[1] / 2.0
            dist_T = distance_to_normalized(dist)
            points_vector_body[i] = [norm_x, norm_y, dist_T]

        # 目标信息
        relative_target_global = self.target - self.agent_pos
        relative_target_body = global_to_body_frame(self.agent_heading, relative_target_global)

        # 归一化处理
        norm_target_x = relative_target_body[0] / 2.0
        norm_target_y = relative_target_body[1] / 2.0
        target_dist_T = target_distance_normalized(self.target_distance)

        # 拼接所有观测
        return np.concatenate([
            [norm_heading],
            [norm_longitudinal_vel, norm_lateral_vel],
            [norm_steering],
            self.last_action,
            points_vector_body.flatten(),
            [norm_target_x, norm_target_y, target_dist_T]
        ], dtype=np.float32)

    def _get_vehicle_corners(self):
        """计算车辆矩形的四个角点"""
        cx, cy = self.agent_pos
        cos_h = math.cos(self.agent_heading)
        sin_h = math.sin(self.agent_heading)

        half_length = self.vehicle_length / 2
        half_width = self.vehicle_width / 2

        corners = np.zeros((4, 2))
        # 左下
        corners[0, 0] = cx - half_length * cos_h - half_width * sin_h
        corners[0, 1] = cy - half_length * sin_h + half_width * cos_h
        # 左上
        corners[1, 0] = cx - half_length * cos_h + half_width * sin_h
        corners[1, 1] = cy - half_length * sin_h - half_width * cos_h
        # 右上
        corners[2, 0] = cx + half_length * cos_h + half_width * sin_h
        corners[2, 1] = cy + half_length * sin_h - half_width * cos_h
        # 右下
        corners[3, 0] = cx + half_length * cos_h - half_width * sin_h
        corners[3, 1] = cy + half_length * sin_h + half_width * cos_h

        return corners

    def render(self):
        if self.render_mode != "human" or self.screen is None:
            return

        self.screen.fill((255, 255, 255))

        # 坐标转换函数
        def to_screen(pos):
            x = int((pos[0] - self.low) * self.scale)
            y = int((pos[1] - self.low) * self.scale)
            return x, y

        # 绘制障碍物
        for i in range(self.current_points):
            pygame.draw.circle(self.screen, (255, 0, 0),
                               to_screen(self.obstacle_abs_positions[i]), 5)

        # 绘制目标
        pygame.draw.circle(self.screen, (0, 255, 0), to_screen(self.target), 8)

        # 绘制车辆
        corners = self._get_vehicle_corners()
        screen_corners = [to_screen(c) for c in corners]
        pygame.draw.polygon(self.screen, (0, 0, 255), screen_corners)

        # 绘制车头方向
        front_x = self.agent_pos[0] + (self.vehicle_length / 2) * math.cos(self.agent_heading)
        front_y = self.agent_pos[1] + (self.vehicle_length / 2) * math.sin(self.agent_heading)
        pygame.draw.circle(self.screen, (255, 255, 0), to_screen((front_x, front_y)), 4)

        # 绘制速度向量
        vel_end = self.agent_pos + self.agent_vel * 0.5
        pygame.draw.line(self.screen, (100, 100, 255),
                         to_screen(self.agent_pos), to_screen(vel_end), 2)

        # 绘制前轮方向
        wheel_angle = self.agent_heading + self.steering_angle
        wheel_end_x = self.agent_pos[0] + self.wheel_base * math.cos(wheel_angle)
        wheel_end_y = self.agent_pos[1] + self.wheel_base * math.sin(wheel_angle)
        pygame.draw.line(self.screen, (255, 0, 255),
                         to_screen(self.agent_pos), to_screen((wheel_end_x, wheel_end_y)), 2)

        # 显示信息
        texts = [
            f"Level: {self.curriculum_level}/{self.max_level}",
            f"Obstacles: {self.current_points}/{self.max_points}",
            f"Steps: {self.current_step}/{self.max_steps}",
            f"Reward: {self.reward:.2f}",
            f"Speed: {np.linalg.norm(self.agent_vel):.2f}/{self.max_velocity:.2f}",
            f"Steering: {math.degrees(self.steering_angle):.1f}°"
        ]

        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 30))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human" and pygame.get_init():
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

        # 简单控制策略
        target_dir = env.target - env.agent_pos
        target_angle = math.atan2(target_dir[1], target_dir[0])
        angle_diff = (target_angle - env.agent_heading + math.pi) % (2 * math.pi) - math.pi

        acceleration = 1.0 if env.target_distance > 0.1 else 0.0
        steering = np.clip(angle_diff / (math.pi / 4), -1.0, 1.0)

        action = np.array([acceleration, steering])
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        time.sleep(0.05)

        if terminated or truncated:
            print(f"回合结束! 总奖励: {env.total_reward:.2f}")
            obs, _ = env.reset()

    env.close()
