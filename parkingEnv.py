import time
import warnings
import pygame
import numpy as np
import random
import math
from typing import Tuple, Dict, List, Optional
import gymnasium as gym
from gymnasium import spaces

# 极坐标参数
NUM_SECTORS = 36  # 36个扇形区域（每10度一个）
MAX_DIST = math.sqrt(2)  # 正方形地图最大对角线长度


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
        self.polar_obstacles = None  # 极坐标障碍物表示
        self.target_pos = None
        self.target_heading = None
        self.target_vector = None
        self.min_distance = None
        self.target_distance = None
        self.target_heading_diff = None
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
        self.max_velocity = 0.2  # 最大速度
        self.max_angular_velocity = 1.0  # 最大角速度 (rad/s)
        self.vehicle_length = 0.1  # 车辆长度
        self.vehicle_width = 0.05  # 车辆宽度
        self.friction = 0.05  # 摩擦系数
        self.wheel_base = 0.06  # 轴距

        # 环境限制
        self.max_steps = 300  # 最大步数
        self.current_step = 0
        self.target_pos_threshold = 0.05  # 到达目标位置的阈值
        self.target_heading_threshold = 0.1  # 到达目标朝向的阈值 (约5.7度)

        # 动作空间: [前向加速度, 转向角速度]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 观测空间设计 (所有值都在[-1,1]范围内)
        # 修改后观测空间: 1朝向 + 1纵向速度 + 1偏转角 + 2上一动作 + NUM_SECTORS障碍物信息 + 3目标信息(车身系x,y,距离) + 1目标朝向差
        obs_shape = 1 + 1 + 1 + 2 + NUM_SECTORS + 3 + 1
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
        self.polar_obstacles = np.ones(NUM_SECTORS, dtype=np.float32) * MAX_DIST  # 初始化为最大距离

        # 生成有效随机点
        for i in range(self.current_points):
            point = self.np_random.uniform(self.low + 0.05, self.high - 0.05, size=2)
            self.obstacle_abs_positions.append(point)

        # 填充无效点（角点）
        for i in range(self.current_points, self.max_points):
            corner = self.corner_points[i % len(self.corner_points)]
            self.obstacle_abs_positions.append(corner)

        # 重置目标位置和朝向
        self.target_pos = self.np_random.uniform(self.low + 0.1, self.high - 0.1, size=2)
        self.target_heading = self.np_random.uniform(0, 2 * np.pi)

        # 确保目标不与车辆初始位置太近
        while np.linalg.norm(self.target_pos - self.agent_pos) < 0.3:
            self.target_pos = self.np_random.uniform(self.low + 0.1, self.high - 0.1, size=2)

        # 重置距离记录
        self.min_distance = float('inf')
        self.target_distance = np.linalg.norm(self.target_pos - self.agent_pos)

        # 计算目标朝向差
        self.target_heading_diff = self._normalize_angle(self.target_heading - self.agent_heading)

        # 更新最近障碍物
        self._update_nearest_obstacle()
        self.last_action = np.zeros(2)

        return self._get_obs(), {}

    def _normalize_angle(self, angle: float) -> float:
        """将角度归一化到[-π, π]范围内"""
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        return angle

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
        # 计算角速度
        angular_velocity = 0.0
        if abs(self.steering_angle) > 1e-5:
            # 使用自行车模型计算角速度
            angular_velocity = (np.linalg.norm(self.agent_vel) * np.tan(self.steering_angle)) / self.wheel_base

        # 限制角速度
        angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)

        # 更新车辆朝向
        self.agent_heading += angular_velocity * self.dt
        self.agent_heading = self.agent_heading % (2 * np.pi)

        # 计算加速度在车身方向上的分量 (投影约束)
        heading_vector = np.array([np.cos(self.agent_heading), np.sin(self.agent_heading)])
        # 将加速度投影到车身方向上
        forward_acc = acceleration * heading_vector

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
        relative_target_vector_global = self.target_pos - self.agent_pos
        self.target_distance = np.linalg.norm(relative_target_vector_global)

        # 更新目标朝向差
        self.target_heading_diff = self._normalize_angle(self.target_heading - self.agent_heading)

        # 更新最近障碍物
        self._update_nearest_obstacle()

        # 计算奖励
        self.reward = self._calculate_reward(boundary_collision)
        self.total_reward += self.reward

        # 检查终止条件
        terminated = False  # (self.target_distance < self.target_pos_threshold and
                            # abs(self.target_heading_diff) < self.target_heading_threshold and
                            # np.linalg.norm(self.agent_vel) < 0.05)  # 速度足够小

        truncated = self.current_step >= self.max_steps

        # 收集额外信息
        info = {
            "curriculum_level": self.curriculum_level,
            "min_distance": self.min_distance,
            "target_distance": self.target_distance,
            "target_heading_diff": self.target_heading_diff,
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
        # 环境常量
        MAX_DIST = 2.0  # 最大可能距离（对角线）
        MAX_HEADING_DIFF = np.pi  # 最大朝向差

        # 1. 目标位置奖励
        norm_target_dist = self.target_distance / MAX_DIST
        pos_reward = 2 * np.exp(-3.0 * norm_target_dist) - 1

        # 2. 目标朝向奖励  靠近目标才考虑朝向
        if norm_target_dist <= 0.2:
            norm_heading_diff = abs(self.target_heading_diff) / MAX_HEADING_DIFF
            heading_reward = 1.5 * np.exp(-5.0 * norm_heading_diff) - 0.5
        else:
            heading_reward = 0

        # # 3. 障碍物惩罚
        # safe_radius = 0.15
        # if self.min_distance < safe_radius:
        #     obstacle_penalty = -2.0 * (1 - self.min_distance / safe_radius)
        # else:
        #     obstacle_penalty = 0.0

        # # 4. 边界碰撞惩罚
        # boundary_penalty = -1.0 if boundary_collision else 0.0

        # 5. 速度惩罚 (鼓励停车)
        speed_penalty = -0.1 * (np.linalg.norm(self.agent_vel) / self.max_velocity)

        # 6. 成功奖励
        success_bonus = 0.0
        if (self.target_distance < self.target_pos_threshold and
                abs(self.target_heading_diff) < self.target_heading_threshold and
                np.linalg.norm(self.agent_vel) < 0.05):
            success_bonus = 10.0

        # 组合奖励  # obstacle_penalty + boundary_penalty +
        total_reward = pos_reward + heading_reward +  speed_penalty + success_bonus

        return float(np.clip(total_reward, -2.0, 10.0))

    def _get_obs(self) -> np.ndarray:
        """构建观测向量 (所有值在[-1,1]范围内)"""
        # 车辆状态 (归一化)
        norm_heading = self.agent_heading / np.pi - 1  # [0, 2π] -> [-1,1]

        # 将速度转换到车身坐标系
        body_vel = self._global_to_body_frame(self.agent_vel)
        norm_longitudinal_vel = body_vel[0] / self.max_velocity
        # 由于投影约束，横向速度应为0，但我们仍然计算它用于观测
        norm_lateral_vel = body_vel[1] / self.max_velocity

        norm_steering = self.steering_angle / self.max_steering_angle  # 转向角归一化

        # === 极坐标障碍物表示 (扇形划分) ===
        # 初始化每个扇形区域的最小距离为最大可能值
        sector_min_dist = np.ones(NUM_SECTORS) * MAX_DIST

        # 处理每个障碍物点
        for pt in self.obstacle_abs_positions[:self.current_points]:
            # 计算相对向量（全局坐标系）
            rel_vec = pt - self.agent_pos

            # 转换到车身坐标系
            rel_body = self._global_to_body_frame(rel_vec)

            # 计算距离
            dist = np.linalg.norm(rel_body)

            # 计算角度（车身坐标系）
            angle = math.atan2(rel_body[1], rel_body[0])  # [-pi, pi]

            # 将角度转换到[0, 2pi)范围
            angle = (angle + 2 * math.pi) % (2 * math.pi)

            # 确定扇形区域ID
            bin_id = int(angle / (2 * math.pi) * NUM_SECTORS)

            # 更新该扇形区域的最小距离
            if dist < sector_min_dist[bin_id]:
                sector_min_dist[bin_id] = dist

        # 归一化处理：使用倒数压缩法映射到[-1,1]
        # 公式: 2/(d+0.2)-1，其中d是距离
        polar_encoded = 2 / (sector_min_dist + 0.2) - 1

        # === 目标位置信息 ===
        relative_target_global = self.target_pos - self.agent_pos
        relative_target_body = self._global_to_body_frame(relative_target_global)
        self.target_distance = np.linalg.norm(relative_target_global)

        # 归一化处理
        norm_target_x = relative_target_body[0] / 2.0
        norm_target_y = relative_target_body[1] / 2.0
        target_dist_T = 2 / (self.target_distance + 0.5) - 1  # 映射到[-1,1]

        # 目标朝向差归一化
        norm_heading_diff = self.target_heading_diff / np.pi  # [-1,1] 范围

        # 拼接所有观测
        return np.concatenate([
            [norm_heading],  # 1: 车辆朝向
            [norm_longitudinal_vel],  # 1: 纵向速度 (由于投影约束，横向速度应为0)
            [norm_steering],  # 1: 偏转角
            self.last_action,  # 2: 上一动作
            polar_encoded,  # NUM_SECTORS: 极坐标障碍物信息
            [norm_target_x,  # 目标在车身坐标系的x位置
             norm_target_y,  # 目标在车身坐标系的y位置
             target_dist_T],  # 目标距离归一化
            [norm_heading_diff]  # 1: 目标朝向差归一化
        ], dtype=np.float32)

    def _get_target_corners(self):
        """计算目标矩形的四个角点"""
        cx, cy = self.target_pos
        cos_h = np.cos(self.target_heading)
        sin_h = np.sin(self.target_heading)

        # 目标矩形尺寸 (比车辆稍大)
        half_length = self.vehicle_length * 1.5 / 2
        half_width = self.vehicle_width * 1.5 / 2

        corners = []
        for dx, dy in [(-half_length, -half_width),
                       (-half_length, half_width),
                       (half_length, half_width),
                       (half_length, -half_width)]:
            x = cx + dx * cos_h - dy * sin_h
            y = cy + dx * sin_h + dy * cos_h
            corners.append((x, y))

        return corners

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

        # 绘制目标矩形
        target_corners = self._get_target_corners()
        screen_target_corners = [to_screen(c) for c in target_corners]
        pygame.draw.polygon(self.screen, (0, 255, 0), screen_target_corners, 2)  # 绿色边框

        # 绘制目标朝向
        target_front_x = self.target_pos[0] + (self.vehicle_length / 2) * np.cos(self.target_heading)
        target_front_y = self.target_pos[1] + (self.vehicle_length / 2) * np.sin(self.target_heading)
        pygame.draw.line(self.screen, (0, 200, 0), to_screen(self.target_pos),
                         to_screen((target_front_x, target_front_y)), 2)

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

        # 绘制目标朝向差
        pygame.draw.line(self.screen, (150, 0, 150), to_screen(self.target_pos), to_screen(self.agent_pos), 1)

        # 绘制扇形区域可视化（调试用）
        for i in range(NUM_SECTORS):
            angle = i * (2 * math.pi / NUM_SECTORS)
            end_x = self.agent_pos[0] + 0.2 * math.cos(angle + self.agent_heading)
            end_y = self.agent_pos[1] + 0.2 * math.sin(angle + self.agent_heading)
            pygame.draw.line(self.screen, (200, 200, 200), to_screen(self.agent_pos),
                             to_screen((end_x, end_y)), 1)

        # 显示信息
        texts = [
            f"Level: {self.curriculum_level}/{self.max_level}",
            f"Obstacles: {self.current_points}/{self.max_points}",
            f"Steps: {self.current_step}/{self.max_steps}",
            f"Reward: {self.reward:.2f}",
            f"Speed: {np.linalg.norm(self.agent_vel):.2f}/{self.max_velocity:.2f}",
            f"Steering: {np.degrees(self.steering_angle):.1f}°",
            f"Long Vel: {self._global_to_body_frame(self.agent_vel)[0]:.3f}",
            f"Target Dist: {self.target_distance:.3f}",
            f"Heading Diff: {np.degrees(self.target_heading_diff):.1f}°"
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

        # 简单控制策略: 朝向目标前进并调整车头方向
        target_dir = env.target_pos - env.agent_pos
        target_angle = np.arctan2(target_dir[1], target_dir[0])
        angle_diff = (target_angle - env.agent_heading + np.pi) % (2 * np.pi) - np.pi

        # 目标朝向调整
        heading_diff = env.target_heading_diff
        heading_adjustment = np.clip(heading_diff / (np.pi / 4), -1.0, 1.0)

        # 动作: [加速度, 转向]
        # 当距离较远时，优先位置控制；当距离较近时，优先朝向控制
        if env.target_distance > 0.1:
            steering = np.clip(angle_diff / (np.pi / 4), -1.0, 1.0)
            acceleration = 1.0 if env.target_distance > 0.05 else 0.5
        else:
            steering = heading_adjustment
            acceleration = 0.3 if abs(heading_diff) > 0.1 else 0.1

        action = np.array([acceleration, steering])
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        if terminated or truncated:
            print(f"回合结束! 总奖励: {env.total_reward:.2f}")
            obs, _ = env.reset()

    env.close()
