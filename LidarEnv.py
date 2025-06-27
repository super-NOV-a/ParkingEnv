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

        # 雷达扫描参数
        self.radar_beams = 360  # 36个方向（每1度一个）
        self.max_detection_distance = 2.0  # 最大探测距离（环境对角线约为2.828）
        self.obstacle_radius = 0.05  # 障碍物半径
        self.agent_radius = 0.01  # 智能体半径

        # 定义动作空间和观测空间
        self.action_space = spaces.Discrete(4)  # 4个方向

        # 观测空间: 雷达扫描距离(36维) + 目标相对方向(2维) + 目标距离(1维)
        obs_shape = self.radar_beams + 3
        self.observation_space = spaces.Box(
            low=np.array([0.0] * self.radar_beams + [-1.0, -1.0, 0.0]),
            high=np.array([1.0] * self.radar_beams + [1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # 渲染设置
        self.render_mode = render_mode
        self.scale = 400  # 增加比例因子以在屏幕上显示更大的区域
        if render_mode == "human":
            pygame.init()
            # 屏幕大小基于值域范围计算
            self.screen_size = (int((self.high - self.low) * self.scale),
                                int((self.high - self.low) * self.scale))
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("强化学习课程环境(雷达观测)")
            self.clock = pygame.time.Clock()

        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)  # 中心位置
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        self.target_distance = float('inf')  # 记录目标距离
        self.current_points = 0  # 当前有效点数
        self.obstacles = []  # 存储障碍物的绝对位置
        self.radar_scan = np.ones(self.radar_beams, dtype=np.float32)  # 雷达扫描结果
        self.collision = False  # 碰撞标志
        self.last_pos = self.agent_pos.copy()  # 用于计算运动方向

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # 重置步数计数器
        self.current_step = 0
        self._update_level()
        self.current_points = int(self.min_points + (self.max_points - self.min_points) *
                                  (self.curriculum_level / self.max_level))

        # 首先应该先重置智能体位置 (值域范围[-1,1])
        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)

        # 生成障碍物
        self.obstacles = []
        for _ in range(self.current_points):
            # 确保障碍物不会生成在智能体初始位置附近
            valid = False
            while not valid:
                point = np.array([
                    self.np_random.uniform(self.low, self.high),
                    self.np_random.uniform(self.low, self.high)
                ], dtype=np.float32)
                # 检查是否与智能体初始位置太近
                if np.linalg.norm(point - self.agent_pos) > self.obstacle_radius + self.agent_radius + 0.1:
                    self.obstacles.append(point)
                    valid = True

        # 重置目标位置 (值域范围[-1,1])
        valid = False
        while not valid:
            self.target = np.array([
                self.np_random.uniform(self.low, self.high),
                self.np_random.uniform(self.low, self.high)
            ], dtype=np.float32) * (self.curriculum_level + 2) / (self.max_level + 2)  # 更柔和的目标位置

            # 确保目标不会生成在障碍物上
            valid = True
            for obstacle in self.obstacles:
                if np.linalg.norm(self.target - obstacle) < self.obstacle_radius * 2:
                    valid = False
                    break

        # 更新目标距离
        self.target_distance = np.linalg.norm(self.agent_pos - self.target)
        self.total_reward = 0
        self.collision = False
        self.last_pos = self.agent_pos.copy()

        # 执行一次雷达扫描
        self._update_radar_scan()

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

    def _update_radar_scan(self):
        """更新雷达扫描结果"""
        self.radar_scan.fill(1.0)  # 初始化为最大距离

        for i in range(self.radar_beams):
            angle = 2 * np.pi * i / self.radar_beams
            direction = np.array([np.cos(angle), np.sin(angle)])

            min_distance = self.max_detection_distance

            # 检查每条射线与所有障碍物的交点
            for obstacle in self.obstacles:
                # 计算障碍物中心到射线的距离
                obstacle_to_agent = obstacle - self.agent_pos
                cross = np.abs(np.cross(direction, obstacle_to_agent))

                # 如果距离小于障碍物半径，则射线与障碍物相交
                if cross <= self.obstacle_radius:
                    # 计算射线与障碍物的交点
                    dot_product = np.dot(obstacle_to_agent, direction)
                    if dot_product > 0:  # 障碍物在射线方向上
                        # 计算交点距离
                        distance = dot_product - np.sqrt(self.obstacle_radius ** 2 - cross ** 2)
                        if distance > 0 and distance < min_distance:
                            min_distance = distance

            # 归一化距离 (0=最近, 1=最大探测距离)
            normalized_distance = min(min_distance / self.max_detection_distance, 1.0)
            self.radar_scan[i] = normalized_distance

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 动作映射
        direction_map = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        new_pos = self.agent_pos + direction_map[action] * self.agent_speed

        # 边界检查 (值域范围[-1,1])
        self.agent_pos = np.clip(new_pos, [self.low, self.low], [self.high, self.high])

        # 更新步数计数器
        self.current_step += 1

        # 更新目标距离
        self.target_distance = np.linalg.norm(self.agent_pos - self.target)

        # 碰撞检测
        self.collision = False
        for obstacle in self.obstacles:
            if np.linalg.norm(self.agent_pos - obstacle) < self.obstacle_radius + self.agent_radius:
                self.collision = True
                break

        # 更新雷达扫描
        self._update_radar_scan()

        # 计算奖励
        self.reward = self._calculate_reward()
        self.total_reward += self.reward

        # 检查是否结束 (到达目标或碰撞或步数限制)
        terminated = self.target_distance < self.target_threshold or self.collision
        truncated = self.current_step >= self.max_steps

        # 收集额外信息
        info = {
            "curriculum_level": self.curriculum_level,
            "target_distance": self.target_distance,
            "episode_length": self.current_step,
            "collision": self.collision
        }

        # 标记回合结束
        if truncated or terminated:
            info["episode"] = {
                "r": self.total_reward,  # 累计奖励
                "l": self.current_step,  # 回合长度
                "curriculum_level": self.curriculum_level
            }

        return self._get_obs(), self.reward, terminated, truncated, info

    def _calculate_reward(self) -> float:
        # 1. 目标距离奖励
        target_reward = 1.0 / (1.0 + 5.0 * self.target_distance) - 0.5

        # 2. 碰撞惩罚
        collision_penalty = -10.0 if self.collision else 0.0

        # 3. 成功奖励
        success_bonus = 10.0 if self.target_distance < self.target_threshold else 0.0

        # 4. 时间惩罚
        time_penalty = -0.005

        # 5. 运动平滑奖励
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

        # 组合奖励
        total_reward = (
                2.0 * target_reward +
                1.5 * collision_penalty +
                success_bonus +
                time_penalty +
                alignment_reward
        )

        # 更新上一位置
        self.last_pos = self.agent_pos.copy()

        return float(total_reward)

    def _get_obs(self) -> np.ndarray:
        """返回雷达观测向量"""
        # 目标方向向量 (归一化)
        target_vector = self.target - self.agent_pos
        target_norm = np.linalg.norm(target_vector)
        if target_norm > 0:
            target_direction = target_vector / target_norm
        else:
            target_direction = np.zeros(2)

        # 目标距离归一化 (0-1)
        max_possible_distance = np.sqrt(2) * 2  # 环境对角线长度
        norm_target_dist = min(target_norm / max_possible_distance, 1.0)

        # 组合观测: 雷达扫描 + 目标方向 + 目标距离
        obs = np.concatenate([
            self.radar_scan,
            target_direction,
            [norm_target_dist]
        ])
        return obs.astype(np.float32)

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

        # 绘制障碍物 (带半径)
        for obstacle in self.obstacles:
            pygame.draw.circle(
                self.screen,
                (255, 0, 0),  # 红色
                to_screen_coords(obstacle),
                int(self.obstacle_radius * self.scale)
            )

        # 绘制智能体
        agent_screen_pos = to_screen_coords(self.agent_pos)
        pygame.draw.circle(
            self.screen,
            (0, 0, 255),  # 蓝色
            agent_screen_pos,
            int(self.agent_radius * self.scale)
        )

        # 绘制目标
        target_screen_pos = to_screen_coords(self.target)
        pygame.draw.circle(
            self.screen,
            (0, 255, 0),  # 绿色
            target_screen_pos,
            int(self.target_threshold * self.scale)
        )

        # 绘制雷达扫描线
        for i in range(self.radar_beams):
            angle = 2 * np.pi * i / self.radar_beams
            distance = self.radar_scan[i] * self.max_detection_distance
            end_x = self.agent_pos[0] + distance * np.cos(angle)
            end_y = self.agent_pos[1] + distance * np.sin(angle)
            end_pos = to_screen_coords((end_x, end_y))

            # 根据距离设置颜色 (近=红, 远=绿)
            color_value = int(255 * (1 - self.radar_scan[i]))
            color = (255, color_value, 0)

            pygame.draw.line(
                self.screen,
                color,
                agent_screen_pos,
                end_pos,
                1
            )

        # 显示课程难度
        font = pygame.font.SysFont(None, 24)
        level_text = font.render(f"Level: {self.curriculum_level}/{self.max_level}", True, (0, 0, 0))
        points_text = font.render(f"Obstacles: {self.current_points}", True, (0, 0, 0))
        step_text = font.render(f"Steps: {self.current_step}/{self.max_steps}", True, (0, 0, 0))
        reward = font.render(f"Reward:{self.reward:.2f}", True, (0, 0, 0))
        target_dist_text = font.render(f"Target Dist: {self.target_distance:.4f}", True, (0, 0, 0))
        collision_text = font.render(f"Collision: {self.collision}", True, (255, 0, 0) if self.collision else (0, 0, 0))

        self.screen.blit(level_text, (10, 10))
        self.screen.blit(points_text, (10, 40))
        self.screen.blit(step_text, (10, 70))
        self.screen.blit(reward, (10, 100))
        self.screen.blit(target_dist_text, (10, 130))
        self.screen.blit(collision_text, (10, 160))

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

        # 定期重置环境
        if terminated or truncated:
            print(f"Episode finished! Total reward: {env.total_reward}")
            obs, _ = env.reset()

    env.close()