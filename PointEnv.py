import time
import warnings
import pygame
import numpy as np
import random
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
        self.agent_speed = 0.05  # 降低速度以适应更小的范围
        self.max_steps = 1000  # 最大步数限制
        self.current_step = 0  # 当前步数计数器
        self.total_reward = 0
        self.target_threshold = 0.05  # 到达目标的距离阈值

        # 定义动作空间和观测空间
        self.action_space = spaces.Discrete(4)  # 4个方向

        # 固定长度观测空间 (智能体位置 + 全部点(位置差与距离倒数) + 目标点(位置差与距离倒数))
        obs_shape = 3 + 3 * self.max_points + 3
        # 修改观测空间范围以适应相对向量
        self.observation_space = spaces.Box(
            low=np.array([self.low, self.low] +
                         [self.low * 2, self.low * 2, -1] * self.max_points +  # 相对向量范围[-2,2]
                         [self.low * 2, self.low * 2, -1]),
            high=np.array([self.high, self.high] +
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
        # self._update_level()
        self.current_points = int(self.min_points + (self.max_points - self.min_points) *
                                  (self.curriculum_level / self.max_level))

        # 首先应该先重置智能体位置 (值域范围[-1,1])
        self.agent_pos = np.clip(np.random.random(2), -1, 1)

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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 动作映射
        direction_map = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        new_pos = self.agent_pos + direction_map[action] * self.agent_speed

        # 边界检查 (值域范围[-1,1])
        self.agent_pos = np.clip(new_pos, [self.low, self.low], [self.high, self.high])

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
        terminated = False # self.target_distance < self.target_threshold
        truncated = self.current_step >= self.max_steps

        # 收集额外信息
        info = {
            "curriculum_level": self.curriculum_level,
            "min_distance": self.min_distance,
            "target_distance": self.target_distance,
            "episode_length": self.current_step,
            "nearest_obstacle_dist_T": self.nearest_obstacle_dist_T,
            "target_dist_T": target_dist_T
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
        safe_radius = 0.2  # 安全距离（占环境尺度的10%）
        collision_radius = 0.05  # 碰撞阈值

        if self.min_distance < safe_radius:
            # 标准化障碍距离（0-1范围内）
            norm_obs_dist = max(0, (self.min_distance - collision_radius)) / safe_radius
            # 三次方惩罚曲线（近距离时梯度急剧增大）
            obstacle_penalty = -1.0 * (1 - norm_obs_dist) ** 3
        else:
            obstacle_penalty = 0.0

        # 3. 成功奖励（稀疏奖励）
        if self.target_distance < self.target_threshold:
            success_bonus = ENV_SCALE * 5.0  # 基于环境尺度的成功奖励
        else:
            success_bonus = 0.0

        # 4. 时间惩罚（防止原地不动）
        time_penalty = -0.005 * ENV_SCALE  # 按比例缩小时间惩罚

        # 5. 运动平滑奖励（可选）
        # 计算移动方向与目标方向的夹角余弦
        if hasattr(self, 'last_pos'):
            movement = self.agent_pos - self.last_pos
            if np.linalg.norm(movement) > 0.1 * self.agent_speed:  # 有效移动
                target_dir = self.target - self.agent_pos
                if np.linalg.norm(target_dir) > 0:
                    cos_sim = np.dot(movement, target_dir) / (
                            np.linalg.norm(movement) * np.linalg.norm(target_dir))
                    alignment_reward = 0.1 * cos_sim
                else:
                    alignment_reward = 0.0
            else:
                alignment_reward = -0.02  # 微小惩罚防止抖动
        else:
            alignment_reward = 0.0

        # 组合奖励（各分量权重已隐含在公式中）
        total_reward = (
                2.0 * target_reward +  # 放大目标奖励
                1.5 * obstacle_penalty +  # 平衡安全惩罚
                success_bonus +
                time_penalty +
                alignment_reward
        )

        # 更新上一位置（用于计算运动方向）
        self.last_pos = self.agent_pos.copy()

        return float(np.clip(total_reward, -5.0, 5.0))  # 防止极端值

    def _get_obs(self) -> np.ndarray:
        """返回固定长度的一维观测向量"""
        # 将向量（50x3）展平成一维数组（150维）
        points_flat = self.points_vector.flatten().astype(np.float32)
        # 拼接所有部分形成最终的一维观测向量
        _obs = np.concatenate([self.agent_pos, points_flat, self.target_vector])
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
        reward = font.render(f"Reward:{self.reward}", True, (0, 0, 0))
        # min_dist_text = font.render(f"Min Dist: {self.min_distance:.4f}", True, (0, 0, 0))
        # target_dist_text = font.render(f"Target Dist: {self.target_distance:.4f}", True, (0, 0, 0))
        #
        # # 显示距离倒数归一化值
        # obs_dist_T_text = font.render(f"Nearest Obstacle Dist T: {self.nearest_obstacle_dist_T:.4f}", True, (0, 0, 0))
        # target_dist_T_text = font.render(f"Target Dist T: {self.target_vector[2]:.4f}", True, (0, 0, 0))

        self.screen.blit(level_text, (10, 10))
        self.screen.blit(points_text, (10, 40))
        self.screen.blit(step_text, (10, 70))
        self.screen.blit(reward, (10, 100))
        # self.screen.blit(min_dist_text, (10, 130))
        # self.screen.blit(target_dist_text, (10, 160))
        # self.screen.blit(obs_dist_T_text, (10, 190))
        # self.screen.blit(target_dist_T_text, (10, 220))

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

        # 随机动作测试
        action = random.randint(0, 3)
        obs, reward, terminated, truncated, _ = env.step(action)
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
