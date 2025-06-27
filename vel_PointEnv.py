import time
import warnings
import pygame
import numpy as np
import random
from typing import Tuple, Dict, List, Optional
import gymnasium as gym
from gymnasium import spaces
from numba import jit

np.random.seed(42)


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
        self.max_acceleration = 0.5  # 最大加速度
        self.dt = 0.1  # 时间步长
        self.max_velocity = 0.5  # 最大速度限制
        self.max_steps = 300  # 最大步数限制
        self.current_step = 0  # 当前步数计数器
        self.total_reward = 0
        self.target_threshold = 0.05  # 到达目标的距离阈值  0.05 相当于20个像素

        # 定义动作空间为连续的加速度向量
        self.action_space = spaces.Box(
            low=np.array([self.low, self.low], dtype=np.float32),
            high=np.array([self.high, self.high], dtype=np.float32),
            dtype=np.float32
        )

        # 固定长度观测空间 (智能体位置 + 速度 + 全部点(位置差与距离倒数) + 目标点(位置差与距离倒数))
        obs_shape = 2 + 3 * self.max_points + 3
        # 修改观测空间范围以适应相对向量
        self.observation_space = spaces.Box(
            low=np.array([-self.max_velocity, -self.max_velocity] + [self.low, self.low] +
                         [self.low * 2, self.low * 2, -1] * self.max_points +  # 相对向量范围[-2,2]
                         [self.low * 2, self.low * 2, -1]),
            high=np.array([self.max_velocity, self.max_velocity] + [self.high, self.high] +
                          [self.high * 2, self.high * 2, 1] * self.max_points +  # 相对向量范围[-2,2]
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
            pygame.display.set_caption("强化学习课程环境")
            self.clock = pygame.time.Clock()

        self.points_vector = []
        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)  # 中心位置
        self.agent_vel = np.array([0.0, 0.0], dtype=np.float32)  # 速度向量
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        self.target_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.min_distance = float('inf')  # 记录最近障碍物距离
        self.target_distance = float('inf')  # 记录目标距离
        self.current_points = 0  # 当前有效点数
        self.obstacle_abs_positions = []  # 存储障碍物的绝对位置
        self.nearest_obstacle_pos = None  # 最近障碍物的位置
        self.nearest_obstacle_dist_T = -1  # 最近障碍物的距离倒数归一化值
        self.reward = 0
        self.last_action = np.zeros(2)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # 重置步数计数器
        self.current_step = 0
        self.last_action = np.zeros(2)
        # self._update_level()
        self.current_points = int(self.min_points + (self.max_points - self.min_points) *
                                  (self.curriculum_level / self.max_level))

        # 首先应该先重置智能体位置 (值域范围[-1,1])
        self.agent_pos = self.np_random.uniform(self.low, self.high)
        self.agent_vel = np.array([0.0, 0.0], dtype=np.float32)  # 重置速度

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
        ], dtype=np.float32) * (self.curriculum_level + 2)/(self.max_level + 2)  # 更柔和的目标位置
        # 计算相对向量
        # todo
        relative_target_vector = self.target - self.agent_pos
        self.target_distance = np.linalg.norm(relative_target_vector)
        relative_target_vector = relative_target_vector/self.target_distance    # 前两维使用方向向量
        target_distance_T = 3 / (self.target_distance + 1) - 2  # 映射到[-1,1]的距离
        self.target_vector = np.array([relative_target_vector[0], relative_target_vector[1], target_distance_T])

        # 重置距离记录
        self.min_distance = float('inf')
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
        # 动作是加速度向量 [ax, ay]
        acceleration = np.clip(action, self.low, self.high) * self.max_acceleration
        # 新增摩擦系数
        FRICTION = 0.9
        # 更新速度（带惯性）
        self.agent_vel = self.agent_vel * FRICTION + acceleration * self.dt

        # 限制最大速度（防止振荡）
        speed = np.linalg.norm(self.agent_vel)
        if speed > self.max_velocity:
            self.agent_vel = self.agent_vel * self.max_velocity / speed

        new_pos = self.agent_pos + self.agent_vel * self.dt

        # 边界检查 (值域范围[-1,1])
        self.agent_pos = np.clip(new_pos, [self.low, self.low], [self.high, self.high])

        # 边界碰撞处理 - 反弹
        punish = 0
        for i in range(2):
            if self.agent_pos[i] <= self.low or self.agent_pos[i] >= self.high:
                self.agent_vel[i] = -self.agent_vel[i] * 0.5  # 反弹并损失部分能量
                punish = 1  # 不希望发生碰撞

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
        self.target_distance = np.linalg.norm(relative_target_vector)
        relative_target_vector = relative_target_vector/self.target_distance    # 前两维使用方向向量
        target_distance_T = 3 / (self.target_distance + 1) - 2  # 映射到[-1,1]的距离
        self.target_vector = np.array([relative_target_vector[0], relative_target_vector[1], target_distance_T])

        # 更新最近障碍物
        self._update_nearest_obstacle()

        # 计算奖励
        self.reward = self._calculate_reward() # - punish
        self.total_reward += self.reward

        # 检查是否结束 (步数限制)
        terminated = False  # self.target_distance < self.target_threshold and np.linalg.norm(self.agent_vel) <= 0.1
        truncated = self.current_step >= self.max_steps

        # 收集额外信息
        info = {
            "curriculum_level": self.curriculum_level,
            "min_distance": self.min_distance,
            "target_distance": self.target_distance,
            "episode_length": self.current_step,
            "nearest_obstacle_dist_T": self.nearest_obstacle_dist_T,
            "target_dist_T": target_distance_T,
            "velocity": np.linalg.norm(self.agent_vel),
            "acceleration": np.linalg.norm(acceleration)
        }

        # 标记回合结束
        if truncated or terminated:
            info["episode"] = {
                "r": self.total_reward,  # 累计奖励
                "l": self.current_step,  # 回合长度
                "curriculum_level": self.curriculum_level
            }

        _obs = self._get_obs()
        self.last_action = action

        return _obs, self.reward, terminated, truncated, info

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
        """调用 JIT 优化的奖励函数"""
        return calculate_reward_core(
            target_distance=self.target_distance,
            min_distance=self.min_distance,
            agent_vel=self.agent_vel,
            target_threshold=self.target_threshold
        )

    def _get_obs(self) -> np.ndarray:
        """返回固定长度的一维观测向量"""
        # 将向量（50x3）展平成一维数组（150维）
        points_flat = self.points_vector.flatten().astype(np.float32)
        # 拼接所有部分形成最终的一维观测向量
        _obs = np.concatenate([
            # self.agent_pos,
            self.agent_vel,
            self.last_action,
            points_flat,
            self.target_vector
        ])
        return _obs

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

        # 绘制智能体
        agent_screen_pos = to_screen_coords(self.agent_pos)
        pygame.draw.circle(
            self.screen,
            (0, 0, 255),  # 蓝色
            agent_screen_pos,
            8
        )

        # 绘制速度向量
        vel_end = self.agent_pos + self.agent_vel
        vel_end_screen = to_screen_coords(vel_end)
        pygame.draw.line(
            self.screen,
            (100, 100, 255),  # 浅蓝色
            agent_screen_pos,
            vel_end_screen,
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
            agent_screen_pos,
            line_end_screen,
            2
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
                agent_screen_pos,
                line_end_obs_screen,
                2
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

        # 显示课程难度
        font = pygame.font.SysFont(None, 24)
        level_text = font.render(f"Level: {self.curriculum_level}/{self.max_level}", True, (0, 0, 0))
        points_text = font.render(
            f"Obstacles: {self.min_points + int((self.max_points - self.min_points) * (self.curriculum_level / self.max_level))}",
            True, (0, 0, 0))
        step_text = font.render(f"Steps: {self.current_step}/{self.max_steps}", True, (0, 0, 0))
        reward_text = font.render(f"Reward:{self.reward:.2f}", True, (0, 0, 0))
        vel_text = font.render(f"Velocity: {np.linalg.norm(self.agent_vel):.3f}", True, (0, 0, 0))

        self.screen.blit(level_text, (10, 10))
        self.screen.blit(points_text, (10, 40))
        self.screen.blit(step_text, (10, 70))
        self.screen.blit(reward_text, (10, 100))
        self.screen.blit(vel_text, (10, 130))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])  # 控制帧率

    def close(self):
        """关闭环境"""
        if self.render_mode == "human":
            pygame.quit()


# 在类外部定义 JIT 兼容的辅助函数
@jit(nopython=True)
def vec_norm(v):
    """计算二维向量的范数（JIT兼容版本）"""
    return np.sqrt(v[0] ** 2 + v[1] ** 2)


@jit(nopython=True)
def calculate_reward_core(
        target_distance: float,
        min_distance: float,
        agent_vel: np.ndarray,
        target_threshold: float
) -> float:
    """
    JIT 优化的奖励计算核心函数
    参数:
        target_distance: 到目标的距离
        min_distance: 到最近障碍物的距离
        agent_vel: 智能体速度向量 [vx, vy]
        target_threshold: 目标达成阈值
    """
    # 环境常量
    MAX_DIST = 2.0  # 最大可能距离（对角线）
    SAFE_RADIUS = 0.2
    COLLISION_RADIUS = 0.05

    # 1. 目标距离奖励
    norm_target_dist = target_distance / MAX_DIST
    target_reward = 2 * np.exp(-3.0 * norm_target_dist) - 1     # 5 * np.exp(-20 * norm_target_dist) - 1

    # 2. 障碍物惩罚
    if min_distance < SAFE_RADIUS:
        norm_obs_dist = max(0, (min_distance - COLLISION_RADIUS)) / SAFE_RADIUS
        obstacle_penalty = -1.2 * (1 - norm_obs_dist) ** 2
    else:
        # 安全区域内的额外奖励
        extra_dist = min_distance - SAFE_RADIUS
        obstacle_penalty = 0.1 * min(1.0, extra_dist / SAFE_RADIUS)

    # 3. 成功奖励
    success_bonus = 0.0
    vel_norm = vec_norm(agent_vel)

    # 接近目标时的奖励
    if target_distance < 3 * target_threshold:
        success_bonus = (3 * target_threshold - target_distance) / (3 * target_threshold)

        # 完全成功的奖励
        if target_distance < target_threshold and vel_norm < 0.1:
            success_bonus += 3.0

    # 4. 时间惩罚
    time_penalty = -0.01

    # 5. 速度惩罚
    velocity_reward = -0.2 * vel_norm

    # 组合奖励
    total_reward = (
            target_reward +
            # obstacle_penalty +
            # success_bonus +
            # time_penalty +
            # velocity_reward +
            0
    )

    return total_reward


# 测试环境
if __name__ == "__main__":
    env = PointEnv(render_mode="human")
    obs, _ = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 随机加速度测试
        # action = np.random.uniform(-1, 1, size=2)
        action = 0.5 * env.target_vector[:2] - 0.5 * env.nearest_obstacle_pos
        obs, reward, terminated, truncated, _ = env.step(action)
        # print("自身速度", obs[:2], "--目标点", obs[2:5], "--动作", action)
        env.render()
        # print(f"Reward: {reward:.2f}, Min Dist: {env.min_distance:.4f}, Target Dist: {env.target_distance:.4f}")
        # print(f"episode Retrun:{env.total_reward}")

        # 定期重置环境
        if terminated or truncated:
            print(f"Episode finished! Total reward: {env.total_reward}")
            obs, _ = env.reset()

        # 控制循环速度
        # pygame.time.delay(50)

    env.close()
