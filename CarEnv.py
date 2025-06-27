import time
import warnings
import pygame
import numpy as np
import random
import math
from typing import Tuple, Dict, List, Optional
import gymnasium as gym
from gymnasium import spaces


class PointEnv(gym.Env):
    metadata = {'render_modes': ['human'], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        # 环境参数 - 值域范围改为[-1,1]
        self.low = -1.0
        self.high = 1.0
        self.min_points = 5
        self.max_points = 50
        self.point_increment = 2
        self.curriculum_level = 0
        self.max_level = 20
        self.min_level = 0
        self.max_steering_acc = 0.5  # 最大转向角加速度 (rad/s^2)
        self.max_throttle = 0.2  # 最大油门加速度
        self.dt = 0.1  # 时间步长
        self.max_velocity = 0.3  # 最大速度限制
        self.max_angular_velocity = 2.0  # 最大角速度限制 (rad/s)
        self.max_steps = 1000  # 最大步数限制
        self.current_step = 0  # 当前步数计数器
        self.total_reward = 0
        self.target_threshold = 0.05  # 到达目标的距离阈值
        self.vehicle_length = 0.1  # 车辆长度 (环境单位)
        self.vehicle_width = 0.05  # 车辆宽度 (环境单位)
        self.friction = 0.05  # 摩擦系数

        # 定义动作空间为两个连续动作：转向角加速度和油门大小
        self.action_space = spaces.Box(
            low=np.array([-self.max_steering_acc, -self.max_throttle], dtype=np.float32),
            high=np.array([self.max_steering_acc, self.max_throttle], dtype=np.float32),
            dtype=np.float32
        )

        # 固定长度观测空间 (车辆位置 + 方向 + 速度 + 角速度 + 全部点(位置差与距离倒数) + 目标点(位置差与距离倒数))
        obs_shape = 6 + 3 * self.max_points + 3
        # 修改观测空间范围以适应相对向量
        self.observation_space = spaces.Box(
            low=np.array([
                             self.low, self.low, -2 * np.pi, -self.max_velocity, -self.max_angular_velocity,
                             -self.max_angular_velocity
                         ] + [self.low * 2, self.low * 2, -1] * self.max_points +  # 相对向量范围[-2,2]
                         [self.low * 2, self.low * 2, -1]),
            high=np.array([
                              self.high, self.high, 2 * np.pi, self.max_velocity, self.max_angular_velocity,
                              self.max_angular_velocity
                          ] + [self.high * 2, self.high * 2, 1] * self.max_points +  # 相对向量范围[-2,2]
                          [self.high * 2, self.high * 2, 1]),
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
        self.scale = 400  # 增加比例因子以在屏幕上显示更大的区域
        if render_mode == "human":
            pygame.init()
            # 屏幕大小基于值域范围计算
            self.screen_size = (int((self.high - self.low) * self.scale),
                                int((self.high - self.low) * self.scale))
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("车辆强化学习课程环境")
            self.clock = pygame.time.Clock()

        self.points_vector = []
        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)  # 车辆中心位置
        self.agent_heading = 0.0  # 车辆朝向 (弧度)
        self.agent_vel = 0.0  # 车辆线速度
        self.agent_angular_vel = 0.0  # 车辆角速度
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        self.target_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.min_distance = float('inf')  # 记录最近障碍物距离
        self.target_distance = float('inf')  # 记录目标距离
        self.current_points = 0  # 当前有效点数
        self.obstacle_abs_positions = []  # 存储障碍物的绝对位置
        self.nearest_obstacle_pos = None  # 最近障碍物的位置
        self.nearest_obstacle_dist_T = -1  # 最近障碍物的距离倒数归一化值
        self.reward = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # 重置步数计数器
        self.current_step = 0
        self.current_points = int(self.min_points + (self.max_points - self.min_points) *
                                  (self.curriculum_level / self.max_level))

        # 重置车辆状态 (值域范围[-1,1])
        self.agent_pos = np.clip(np.random.random(2), -1, 1)
        self.agent_heading = self.np_random.uniform(0, 2 * np.pi)  # 随机初始方向
        self.agent_vel = 0.0  # 初始速度为0
        self.agent_angular_vel = 0.0  # 初始角速度为0

        # 初始化空列表收集点特征
        point_list = []
        self.obstacle_abs_positions = []  # 重置障碍物绝对位置

        # 生成有效随机点
        for _ in range(self.current_points):
            point = np.array([
                self.np_random.uniform(self.low, self.high),
                self.np_random.uniform(self.low, self.high)
            ], dtype=np.float32)
            self.obstacle_abs_positions.append(point)  # 存储绝对位置
            relative_point_vector = point - self.agent_pos
            point_distance_T = 3 / (np.linalg.norm(relative_point_vector) + 1) - 2  # 映射到[-1,1]
            point_list.append([relative_point_vector[0], relative_point_vector[1], point_distance_T])

        # 填充无效点（角点）
        num_invalid = self.max_points - self.current_points
        for _ in range(num_invalid):
            idx = np.random.randint(len(self.corner_points))
            corner = self.corner_points[idx]
            self.obstacle_abs_positions.append(corner)  # 存储绝对位置
            point_list.append([corner[0], corner[1], -1])

        # 转换为NumPy数组并调整维度
        self.points_vector = np.array(point_list, dtype=np.float32)  # 默认形状 [max_points, 3]

        # 重置目标位置 (值域范围[-1,1])
        self.target = np.array([
            self.np_random.uniform(self.low, self.high),
            self.np_random.uniform(self.low, self.high)
        ], dtype=np.float32) * (self.curriculum_level + 2) / (self.max_level + 2)  # 更柔和的目标位置
        # 计算相对向量
        relative_target_vector = self.target - self.agent_pos
        # 计算距离映射值
        target_distance_T = 3 / (np.linalg.norm(relative_target_vector) + 1) - 2  # 映射到[-1,1]的距离
        self.target_vector = np.array([relative_target_vector[0], relative_target_vector[1], target_distance_T])

        # 重置距离记录
        self.min_distance = float('inf')
        self.target_distance = np.linalg.norm(self.agent_pos - self.target)
        self.total_reward = 0

        # 重置最近障碍物信息
        self.nearest_obstacle_pos = None
        self.nearest_obstacle_dist_T = -1

        # 计算初始最近障碍物
        self._update_nearest_obstacle()

        return self._get_obs(), {}

    def _update_level(self):
        if self.total_reward >= 300:  # 升级条件
            # 更新课程难度
            self.curriculum_level = min(self.curriculum_level + 1, self.max_level)
            print('What a Smart Agent!-----------Level Up------------------')
        elif self.total_reward <= 0:  # 降级条件
            # 更新课程难度
            self.curriculum_level = max(self.curriculum_level - 1, self.min_level)
            print('What a Stupid Agent!----------Level Down----------------')

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 动作解析: [转向角加速度, 油门大小]
        steering_acc = np.clip(action[0], -self.max_steering_acc, self.max_steering_acc)
        throttle = np.clip(action[1], -self.max_throttle, self.max_throttle)

        # 应用物理模型更新角速度和线速度
        self.agent_angular_vel += steering_acc * self.dt
        self.agent_angular_vel = np.clip(self.agent_angular_vel, -self.max_angular_velocity, self.max_angular_velocity)

        # 应用摩擦
        self.agent_vel *= (1 - self.friction)
        self.agent_angular_vel *= (1 - self.friction * 0.5)

        # 应用油门
        self.agent_vel += throttle * self.dt
        self.agent_vel = np.clip(self.agent_vel, -self.max_velocity, self.max_velocity)

        # 更新朝向
        self.agent_heading += self.agent_angular_vel * self.dt
        # 归一化角度到 [0, 2π]
        self.agent_heading = self.agent_heading % (2 * np.pi)

        # 计算位移向量
        displacement = np.array([
            np.cos(self.agent_heading) * self.agent_vel * self.dt,
            np.sin(self.agent_heading) * self.agent_vel * self.dt
        ])

        # 更新位置
        new_pos = self.agent_pos + displacement

        # 边界检查 (值域范围[-1,1])
        self.agent_pos = np.clip(new_pos, [self.low, self.low], [self.high, self.high])

        # 边界碰撞处理 - 反弹并损失能量
        for i in range(2):
            if self.agent_pos[i] <= self.low or self.agent_pos[i] >= self.high:
                self.agent_vel *= -0.5  # 反弹并损失部分能量
                self.agent_angular_vel *= 0.7  # 角速度也损失部分能量

        # 更新步数计数器
        self.current_step += 1

        # 更新所有点的相对向量和距离倒数
        for i in range(self.max_points):
            # 计算新的相对向量
            relative_vector = self.obstacle_abs_positions[i] - self.agent_pos
            # 对于有效点，更新整个向量
            if i < self.current_points:
                dist = np.linalg.norm(relative_vector)
                dist_T = 3 / (dist + 1) - 2  # 映射到[-1,1]
                self.points_vector[i] = [relative_vector[0], relative_vector[1], dist_T]
            else:  # 无效点，只更新相对向量
                self.points_vector[i, 0] = relative_vector[0]
                self.points_vector[i, 1] = relative_vector[1]

        # 更新目标向量
        relative_target_vector = self.target - self.agent_pos
        target_dist = np.linalg.norm(relative_target_vector)
        target_dist_T = 3 / (target_dist + 1) - 2
        self.target_vector = np.array([relative_target_vector[0], relative_target_vector[1], target_dist_T])
        self.target_distance = target_dist

        # 更新最近障碍物
        self._update_nearest_obstacle()

        # 计算奖励
        self.reward = self._calculate_reward()
        self.total_reward += self.reward

        # 检查是否结束 (步数限制或到达目标)
        terminated = target_dist < self.target_threshold
        truncated = self.current_step >= self.max_steps

        # 收集额外信息
        info = {
            "curriculum_level": self.curriculum_level,
            "min_distance": self.min_distance,
            "target_distance": self.target_distance,
            "episode_length": self.current_step,
            "nearest_obstacle_dist_T": self.nearest_obstacle_dist_T,
            "target_dist_T": target_dist_T,
            "velocity": self.agent_vel,
            "angular_velocity": self.agent_angular_vel,
            "heading": self.agent_heading
        }

        # 标记回合结束
        if truncated or terminated:
            info["episode"] = {
                "r": self.total_reward,  # 累计奖励
                "l": self.current_step,  # 回合长度
                "curriculum_level": self.curriculum_level
            }

        return self._get_obs(), self.reward, terminated, truncated, info

    def _update_nearest_obstacle(self):
        """更新最近障碍物的位置和距离倒数归一化值"""
        if self.current_points == 0:
            self.nearest_obstacle_pos = None
            self.nearest_obstacle_dist_T = -1
            self.min_distance = float('inf')
            return

        # 初始化变量
        min_dist = float('inf')
        nearest_idx = -1

        # 遍历有效障碍物
        for i in range(self.current_points):
            # 计算距离
            dist = np.linalg.norm(self.agent_pos - self.obstacle_abs_positions[i])
            # 更新最小距离
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # 更新最近障碍物信息
        self.min_distance = min_dist
        self.nearest_obstacle_pos = self.obstacle_abs_positions[nearest_idx]
        self.nearest_obstacle_dist_T = self.points_vector[nearest_idx, 2]

    def _calculate_reward(self) -> float:
        # 标准化参数（基于环境尺度[-1,1]）
        ENV_SCALE = 2.0  # 环境总宽度
        MAX_DIST = 2.0  # 最大可能距离（对角线）

        # 1. 目标距离奖励（主激励信号）
        # 使用归一化距离（0-1范围），采用反比例奖励曲线
        norm_target_dist = self.target_distance / MAX_DIST
        target_reward = 1.0 / (1.0 + 5.0 * norm_target_dist) - 0.5  # 范围[-0.5, 0.5]

        # 2. 障碍物惩罚（安全约束）
        # 考虑到车辆尺寸，使用更大的安全半径
        safe_radius = 0.25  # 安全距离（占环境尺度的12.5%）
        collision_radius = 0.1  # 碰撞阈值（考虑车辆尺寸）

        if self.min_distance < safe_radius:
            # 标准化障碍距离（0-1范围内）
            norm_obs_dist = max(0, (self.min_distance - collision_radius)) / safe_radius
            # 三次方惩罚曲线（近距离时梯度急剧增大）
            obstacle_penalty = -1.5 * (1 - norm_obs_dist) ** 3
        else:
            obstacle_penalty = 0.0

        # 3. 成功奖励（稀疏奖励）
        if self.target_distance < self.target_threshold:
            success_bonus = ENV_SCALE * 10.0  # 基于环境尺度的成功奖励
        else:
            success_bonus = 0.0

        # 4. 时间惩罚（防止原地不动）
        time_penalty = -0.005 * ENV_SCALE  # 按比例缩小时间惩罚

        # 5. 方向奖励（鼓励车辆朝向目标）
        target_dir = self.target - self.agent_pos
        if np.linalg.norm(target_dir) > 0.01:
            target_angle = np.arctan2(target_dir[1], target_dir[0])
            # 计算车辆朝向与目标方向的夹角差
            angle_diff = abs((self.agent_heading - target_angle + np.pi) % (2 * np.pi) - np.pi)
            # 夹角差在0到π之间，归一化到0-1
            norm_angle_diff = angle_diff / np.pi
            # 夹角越小奖励越大
            alignment_reward = 0.2 * (1 - norm_angle_diff)
        else:
            alignment_reward = 0.0

        # 6. 速度惩罚（防止高速碰撞）
        velocity_penalty = -0.02 * (abs(self.agent_vel) ** 2)

        # 7. 角速度惩罚（防止过度转向）
        angular_vel_penalty = -0.01 * (abs(self.agent_angular_vel) ** 2)

        # 组合奖励
        total_reward = (
                2.0 * target_reward +  # 放大目标奖励
                1.8 * obstacle_penalty +  # 平衡安全惩罚
                success_bonus +
                time_penalty +
                alignment_reward +
                velocity_penalty +
                angular_vel_penalty
        )

        return float(np.clip(total_reward, -5.0, 5.0))  # 防止极端值

    def _get_obs(self) -> np.ndarray:
        """返回固定长度的一维观测向量"""
        # 将向量（50x3）展平成一维数组（150维）
        points_flat = self.points_vector.flatten().astype(np.float32)
        # 拼接所有部分形成最终的一维观测向量
        _obs = np.concatenate([
            self.agent_pos,
            [self.agent_heading],
            [self.agent_vel],
            [self.agent_angular_vel],
            [0.0],  # 预留位置，用于未来扩展
            points_flat,
            self.target_vector
        ])
        return _obs

    def _get_vehicle_corners(self):
        """计算车辆矩形的四个角点"""
        # 车辆中心位置
        cx, cy = self.agent_pos

        # 车辆方向向量
        cos_heading = np.cos(self.agent_heading)
        sin_heading = np.sin(self.agent_heading)

        # 车辆尺寸
        half_length = self.vehicle_length / 2
        half_width = self.vehicle_width / 2

        # 计算四个角点
        corners = []
        for dx, dy in [(-half_length, -half_width),
                       (-half_length, half_width),
                       (half_length, half_width),
                       (half_length, -half_width)]:
            # 旋转并平移
            x = cx + dx * cos_heading - dy * sin_heading
            y = cy + dx * sin_heading + dy * cos_heading
            corners.append((x, y))

        return corners

    def _get_vehicle_front(self):
        """计算车头位置"""
        cx, cy = self.agent_pos
        front_x = cx + (self.vehicle_length / 2) * np.cos(self.agent_heading)
        front_y = cy + (self.vehicle_length / 2) * np.sin(self.agent_heading)
        return np.array([front_x, front_y])

    def render(self):
        """渲染环境 - 将归一化坐标转换为屏幕坐标"""
        if self.render_mode != "human":
            return

        self.screen.fill((255, 255, 255))  # 白色背景

        # 将归一化坐标转换为屏幕坐标
        def to_screen_coords(pos):
            x = (pos[0] - self.low) * self.scale
            y = (pos[1] - self.low) * self.scale
            return int(x), int(y)

        # 绘制随机点 (只绘制有效点)
        for i in range(self.current_points):
            point = self.obstacle_abs_positions[i]
            pygame.draw.circle(
                self.screen,
                (255, 0, 0),  # 红色
                to_screen_coords(point),
                5
            )

        # 绘制车辆
        vehicle_corners = self._get_vehicle_corners()
        screen_corners = [to_screen_coords(corner) for corner in vehicle_corners]
        pygame.draw.polygon(
            self.screen,
            (0, 0, 255),  # 蓝色
            screen_corners
        )

        # 绘制车头标记
        front_pos = self._get_vehicle_front()
        front_screen = to_screen_coords(front_pos)
        pygame.draw.circle(
            self.screen,
            (255, 255, 0),  # 黄色
            front_screen,
            5
        )

        # 绘制车辆朝向指示线
        direction_end = self.agent_pos + np.array([
            np.cos(self.agent_heading) * self.vehicle_length * 0.8,
            np.sin(self.agent_heading) * self.vehicle_length * 0.8
        ])
        pygame.draw.line(
            self.screen,
            (0, 255, 0),  # 绿色
            to_screen_coords(self.agent_pos),
            to_screen_coords(direction_end),
            2
        )

        # 绘制目标
        target_screen_pos = to_screen_coords(self.target)
        pygame.draw.circle(
            self.screen,
            (0, 255, 0),  # 绿色
            target_screen_pos,
            8
        )

        # 绘制指向目标的箭头
        direction_vector = self.target - self.agent_pos
        line_end = self.agent_pos + direction_vector
        line_end_screen = to_screen_coords(line_end)
        pygame.draw.line(
            self.screen,
            (0, 0, 0),  # 黑色
            to_screen_coords(self.agent_pos),
            line_end_screen,
            1
        )
        # 绘制箭头头部
        arrow_size = 8
        if np.linalg.norm(direction_vector) > 0.001:
            direction = direction_vector / np.linalg.norm(direction_vector)
        else:
            direction = np.array([0, 0])
        perp = np.array([-direction[1], direction[0]]) * arrow_size
        arrow_point1 = (
            line_end_screen[0] - direction[0] * arrow_size + perp[0],
            line_end_screen[1] - direction[1] * arrow_size + perp[1]
        )
        arrow_point2 = (
            line_end_screen[0] - direction[0] * arrow_size - perp[0],
            line_end_screen[1] - direction[1] * arrow_size - perp[1]
        )
        pygame.draw.polygon(
            self.screen,
            (0, 0, 0),  # 黑色
            [line_end_screen, arrow_point1, arrow_point2]
        )

        # 绘制指向最近障碍物的箭头（如果存在）
        if self.nearest_obstacle_pos is not None:
            direction_vector_obs = self.nearest_obstacle_pos - self.agent_pos
            line_end_obs = self.agent_pos + direction_vector_obs
            line_end_obs_screen = to_screen_coords(line_end_obs)
            pygame.draw.line(
                self.screen,
                (255, 0, 0),  # 红色
                to_screen_coords(self.agent_pos),
                line_end_obs_screen,
                1
            )
            # 绘制箭头头部
            if np.linalg.norm(direction_vector_obs) > 0.001:
                direction_obs = direction_vector_obs / np.linalg.norm(direction_vector_obs)
            else:
                direction_obs = np.array([0, 0])
            perp_obs = np.array([-direction_obs[1], direction_obs[0]]) * arrow_size
            arrow_point1_obs = (
                line_end_obs_screen[0] - direction_obs[0] * arrow_size + perp_obs[0],
                line_end_obs_screen[1] - direction_obs[1] * arrow_size + perp_obs[1]
            )
            arrow_point2_obs = (
                line_end_obs_screen[0] - direction_obs[0] * arrow_size - perp_obs[0],
                line_end_obs_screen[1] - direction_obs[1] * arrow_size - perp_obs[1]
            )
            pygame.draw.polygon(
                self.screen,
                (255, 0, 0),  # 红色
                [line_end_obs_screen, arrow_point1_obs, arrow_point2_obs]
            )

        # 显示车辆状态信息
        font = pygame.font.SysFont(None, 24)
        level_text = font.render(f"Level: {self.curriculum_level}/{self.max_level}", True, (0, 0, 0))
        points_text = font.render(
            f"Obstacles: {self.min_points + int((self.max_points - self.min_points) * (self.curriculum_level / self.max_level))}",
            True, (0, 0, 0))
        step_text = font.render(f"Steps: {self.current_step}/{self.max_steps}", True, (0, 0, 0))
        reward_text = font.render(f"Reward:{self.reward:.2f}", True, (0, 0, 0))
        vel_text = font.render(f"Velocity: {self.agent_vel:.3f}", True, (0, 0, 0))
        ang_vel_text = font.render(f"Ang Vel: {self.agent_angular_vel:.3f}", True, (0, 0, 0))
        heading_text = font.render(f"Heading: {np.degrees(self.agent_heading):.1f}°", True, (0, 0, 0))

        self.screen.blit(level_text, (10, 10))
        self.screen.blit(points_text, (10, 40))
        self.screen.blit(step_text, (10, 70))
        self.screen.blit(reward_text, (10, 100))
        self.screen.blit(vel_text, (10, 130))
        self.screen.blit(ang_vel_text, (10, 160))
        self.screen.blit(heading_text, (10, 190))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])  # 控制帧率

    def close(self):
        """关闭环境"""
        if self.render_mode == "human":
            pygame.quit()


# 测试环境
if __name__ == "__main__":
    env = PointEnv(render_mode="human")
    obs, _ = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 随机动作测试: [转向角加速度, 油门大小]
        steering_acc = np.random.uniform(-env.max_steering_acc, env.max_steering_acc)
        throttle = np.random.uniform(-env.max_throttle, env.max_throttle)
        action = np.array([steering_acc, throttle])

        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        # 定期重置环境
        if terminated or truncated:
            print(f"Episode finished! Total reward: {env.total_reward}")
            obs, _ = env.reset()

    env.close()
