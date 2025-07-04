import json
import numpy as np
import pygame
import math
import os
import shapely
from shapely.geometry import Point, Polygon, LineString, MultiPoint, MultiLineString
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry
from gymnasium import Env, spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from multiprocessing import Pool, cpu_count
import time
import random
from lidar import Lidar2D

np.random.seed(123)
random.seed(123)

class ParkingEnv(Env):
    def __init__(self, config):
        # 配置参数
        self.data_dir = config.get('data_dir', 'scenarios')
        self.dt = config.get('timestep', 0.1)
        self.max_range = 30.0
        self.max_steps = config.get('max_steps', 500)
        self.render_mode = config.get('render_mode', 'human')
        self.scenario_mode = config.get('scenario_mode', 'random')  # 默认改为随机模式
        self.collision_threshold = config.get('collision_threshold', 0.5)
        
        # 仅在文件模式下初始化场景文件列表
        self.scenario_files = None
        if self.scenario_mode == 'file':
            self.scenario_files = self._get_scenario_files()
        
        # 随机场景参数
        self.world_size = config.get('world_size', 30.0)
        self.min_obstacles = config.get('min_obstacles', 3)
        self.max_obstacles = config.get('max_obstacles', 8)
        self.min_parking_size = config.get('min_parking_size', 3.0)
        self.max_parking_size = config.get('max_parking_size', 5.0)
        self.min_obstacle_size = config.get('min_obstacle_size', 1.0)
        self.max_obstacle_size = config.get('max_obstacle_size', 10.0)
        
        # 车辆参数
        self.wheelbase = 2.5
        self.max_steer = np.radians(30)
        self.max_speed = 2.0
        self.car_length = 5
        self.car_width = 2
        
        # 雷达配置
        lidar_config = {
            'range_min': 0.5,
            'max_range': config.get('max_range', 10.0),
            'angle_range': 360,
            'num_beams': 72,
            'noise': False,
            'std': 0.05,
            'angle_std': 0.5
        }
        self.lidar = Lidar2D(lidar_config)
        
        # 观测空间
        self.observation_space = spaces.Box(
            low=0,
            high=self.max_range,
            shape=(lidar_config['num_beams'] + 5,),
            dtype=np.float32
        )
        
        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([-1, -1]), 
            high=np.array([1, 1]),
            dtype=np.float32
        )
        
        # 场景数据
        self.current_scenario = "Random Scenario" if self.scenario_mode == 'random' else None
        self.ego_info = None
        self.target_info = None
        self.obstacles = []
        self.obstacle_geoms = []
        
        # 车辆状态
        self.vehicle_state = None
        self.step_count = 0
        self.prev_dist = float('inf')
        
        # 渲染相关
        self.screen = None
        self.clock = pygame.time.Clock()
        self.screen_size = (800, 800)
        self.scale = 10
    
    def _get_scenario_files(self):
        """仅在文件模式下获取场景JSON文件"""
        files = []
        for f in os.listdir(self.data_dir):
            if f.endswith('.json'):
                files.append(os.path.join(self.data_dir, f))
        return files
    
    def _load_scenario(self, file_path):
        """从JSON文件加载场景"""
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        nfm_origin = data['Frames']['0'].get("m_nfmOrigin", [0, 0])
        m_pathOrigin = data['Frames']['0']['PlanningRequest'].get("m_origin", [0, 0])
        
        # 提取自车信息
        ego_data = data['Frames']['0']['PlanningRequest']['m_startPosture']['m_pose']
        ego_info = [
            ego_data[0] + m_pathOrigin[0] - nfm_origin[0],
            ego_data[1] + m_pathOrigin[1] - nfm_origin[1],
            self._normalize_angle(ego_data[2])
        ]
        
        # 提取目标信息
        target_data = data['Frames']['0']['PlanningRequest']['m_targetArea']['m_targetPosture']['m_pose']
        target_info = [
            target_data[0] + m_pathOrigin[0] - nfm_origin[0],
            target_data[1] + m_pathOrigin[1] - nfm_origin[1],
            self._normalize_angle(target_data[2])
        ]
        
        # 提取障碍物
        obstacles = []
        for obj in data['Frames']['0']['NfmAggregatedPolygonObjects']:
            points = []
            if 'nfmPolygonObjectNodes' in obj.keys():
                for point in obj['nfmPolygonObjectNodes']:
                    x = point['m_x'] + m_pathOrigin[0] - nfm_origin[0]
                    y = point['m_y'] + m_pathOrigin[1] - nfm_origin[1]
                    points.append((x, y))
                obstacles.append(points)
        
        return ego_info, target_info, obstacles
    
    def _generate_random_scenario(self):
        """生成随机障碍物和停车位场景"""
        # 生成停车位
        parking_size = random.uniform(self.min_parking_size, self.max_parking_size)
        parking_orientation = random.uniform(0, 2 * math.pi)
        
        padding = parking_size * 1.5
        target_x = random.uniform(padding, self.world_size - padding)
        target_y = random.uniform(padding, self.world_size - padding)
        self.target_info = [target_x, target_y, parking_orientation]
        
        # 生成自车初始位置
        min_start_dist = parking_size * 2
        max_start_dist = parking_size * 4
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(min_start_dist, max_start_dist)
        ego_x = target_x + dist * math.cos(angle)
        ego_y = target_y + dist * math.sin(angle)
        ego_x = np.clip(ego_x, padding, self.world_size - padding)
        ego_y = np.clip(ego_y, padding, self.world_size - padding)
        ego_yaw = random.uniform(0, 2 * math.pi)
        self.ego_info = [ego_x, ego_y, ego_yaw]
        
        # 生成随机障碍物
        self.obstacles = []
        num_obstacles = random.randint(self.min_obstacles, self.max_obstacles)
        
        for _ in range(num_obstacles):
            for attempt in range(10):
                obstacle_type = random.choice(['rectangle', 'polygon'])
                
                if obstacle_type == 'rectangle':
                    width = random.uniform(self.min_obstacle_size, self.max_obstacle_size)
                    height = random.uniform(self.min_obstacle_size, self.max_obstacle_size)
                    x = random.uniform(0, self.world_size)
                    y = random.uniform(0, self.world_size)
                    angle = random.uniform(0, 2 * math.pi)
                    
                    half_w = width / 2
                    half_h = height / 2
                    corners = [
                        (-half_w, -half_h),
                        (-half_w, half_h),
                        (half_w, half_h),
                        (half_w, -half_h)
                    ]
                    
                    obstacle = []
                    for cx, cy in corners:
                        rx = cx * math.cos(angle) - cy * math.sin(angle)
                        ry = cx * math.sin(angle) + cy * math.cos(angle)
                        obstacle.append((x + rx, y + ry))
                else:
                    num_sides = random.randint(3, 6)
                    radius = random.uniform(self.min_obstacle_size, self.max_obstacle_size)
                    x = random.uniform(radius, self.world_size - radius)
                    y = random.uniform(radius, self.world_size - radius)
                    angle_offset = random.uniform(0, 2 * math.pi)
                    
                    obstacle = []
                    for i in range(num_sides):
                        angle = angle_offset + i * (2 * math.pi / num_sides)
                        px = x + radius * math.cos(angle)
                        py = y + radius * math.sin(angle)
                        obstacle.append((px, py))
                
                poly = Polygon(obstacle)
                ego_point = Point(self.ego_info[0], self.ego_info[1])
                target_point = Point(self.target_info[0], self.target_info[1])
                
                min_dist_to_ego = poly.distance(ego_point)
                min_dist_to_target = poly.distance(target_point)
                
                if min_dist_to_ego > self.car_length and min_dist_to_target > parking_size:
                    self.obstacles.append(obstacle)
                    break
        
        return self.ego_info, self.target_info, self.obstacles
    
    def _normalize_angle(self, angle):
        """角度归一化"""
        return math.atan2(math.sin(angle), math.cos(angle))
    
    def step(self, action):
        steer_cmd = np.clip(action[0], -1, 1)
        throttle_cmd = np.clip(action[1], -1, 1)
        
        # 应用转向滤波
        current_steer = self.vehicle_state[4]
        new_steer = current_steer * 0.7 + steer_cmd * self.max_steer * 0.3
        
        # 更新车辆状态
        x, y, yaw, v, _ = self.vehicle_state
        new_v = np.clip(v + throttle_cmd * 1.0 * self.dt, -self.max_speed, self.max_speed)
        new_x = x + new_v * math.cos(yaw) * self.dt
        new_y = y + new_v * math.sin(yaw) * self.dt
        
        # 更新朝向
        if abs(new_steer) > 1e-5:
            turn_radius = self.wheelbase / math.tan(new_steer)
            angular_velocity = new_v / turn_radius
            new_yaw = yaw + angular_velocity * self.dt
        else:
            new_yaw = yaw
        
        new_yaw = self._normalize_angle(new_yaw)
        self.vehicle_state = np.array([new_x, new_y, new_yaw, new_v, new_steer])
        
        # 获取观测
        obs = self._get_observation()
        
        # 检查终止条件
        terminated, truncated = self._check_termination()
        
        # 计算奖励
        reward = self._calculate_reward(terminated, truncated)
        self.step_count += 1
        
        # 渲染
        if self.render_mode == 'human' and self.step_count % 5 == 0:
            self.render()

        return obs, reward, terminated, truncated, {}
    
    def reset(self, scenario_idx=None):
        # 根据场景模式加载或生成场景
        if self.scenario_mode == 'random':
            self.ego_info, self.target_info, self.obstacles = self._generate_random_scenario()
            self.current_scenario = "Random Scenario"
        else:
            if scenario_idx is None:
                scenario_idx = np.random.randint(0, len(self.scenario_files))
            self.current_scenario = self.scenario_files[scenario_idx]
            self.ego_info, self.target_info, self.obstacles = self._load_scenario(self.current_scenario)
        
        # 创建障碍物几何对象
        self.obstacle_geoms = []
        for obs in self.obstacles:
            if len(obs) >= 3:
                poly = Polygon(obs)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                self.obstacle_geoms.append(poly)
            elif len(obs) == 2:
                self.obstacle_geoms.append(LineString(obs))
            elif len(obs) == 1:
                self.obstacle_geoms.append(Point(obs[0]))

        # 更新雷达障碍物
        self.lidar.update_obstacles(self.obstacle_geoms)
        
        # 初始化车辆状态
        self.vehicle_state = np.array([
            self.ego_info[0],
            self.ego_info[1],
            self.ego_info[2],
            0.0,
            0.0
        ])
        
        self.step_count = 0
        self.prev_dist = float('inf')
        
        return self._get_observation(), {}

    def render(self, mode=None):
        """渲染当前环境状态 - 优化版本"""
        if mode is None:
            mode = self.render_mode
        
        # 初始化Pygame（仅第一次）
        if self.screen is None and mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Parking Environment")
        
        # 处理事件（防止卡死）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    return
        
        # 清空屏幕
        self.screen.fill((255, 255, 255))
        
        # 计算视图中心 (以自车为中心)
        center_x, center_y = self.screen_size[0] // 2, self.screen_size[1] // 2
        
        # 绘制障碍物
        for obs in self.obstacles:
            pygame_points = []
            for x, y in obs:
                # 转换到屏幕坐标 (以自车为中心)
                dx = x - self.vehicle_state[0]
                dy = y - self.vehicle_state[1]
                # 旋转以对齐车辆朝向
                rx = dx * math.cos(-self.vehicle_state[2]) - dy * math.sin(-self.vehicle_state[2])
                ry = dx * math.sin(-self.vehicle_state[2]) + dy * math.cos(-self.vehicle_state[2])
                # 缩放并移动到中心
                screen_x = center_x + int(rx * self.scale)
                screen_y = center_y - int(ry * self.scale)  # 注意Y轴方向
                pygame_points.append((screen_x, screen_y))
            
            if len(pygame_points) >= 3:
                pygame.draw.polygon(self.screen, (100, 100, 100), pygame_points)
            elif len(pygame_points) == 2:
                pygame.draw.line(self.screen, (100, 100, 100), pygame_points[0], pygame_points[1], 2)
        
        # 绘制目标位置和朝向
        dx = self.target_info[0] - self.vehicle_state[0]
        dy = self.target_info[1] - self.vehicle_state[1]
        rx = dx * math.cos(-self.vehicle_state[2]) - dy * math.sin(-self.vehicle_state[2])
        ry = dx * math.sin(-self.vehicle_state[2]) + dy * math.cos(-self.vehicle_state[2])
        target_x = center_x + int(rx * self.scale)
        target_y = center_y - int(ry * self.scale)
        
        # 绘制停车位（矩形）
        parking_length = self.car_length * 1.2
        parking_width = self.car_width * 1.5
        
        # 计算停车位四个角点（在全局坐标系）
        half_length = parking_length / 2
        half_width = parking_width / 2
        
        # 计算停车位角点（相对于目标中心）
        corners = [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, half_width),
            (half_length, -half_width)
        ]
        
        # 旋转并平移角点
        target_yaw = self.target_info[2]
        rotated_corners = []
        for cx, cy in corners:
            # 旋转
            rx = cx * math.cos(target_yaw) - cy * math.sin(target_yaw)
            ry = cx * math.sin(target_yaw) + cy * math.cos(target_yaw)
            # 平移
            px = self.target_info[0] + rx
            py = self.target_info[1] + ry
            # 转换到车辆坐标系
            dx = px - self.vehicle_state[0]
            dy = py - self.vehicle_state[1]
            # 旋转以对齐车辆朝向
            vx = dx * math.cos(-self.vehicle_state[2]) - dy * math.sin(-self.vehicle_state[2])
            vy = dx * math.sin(-self.vehicle_state[2]) + dy * math.cos(-self.vehicle_state[2])
            # 转换到屏幕坐标
            screen_x = center_x + int(vx * self.scale)
            screen_y = center_y - int(vy * self.scale)
            rotated_corners.append((screen_x, screen_y))
        
        # 绘制停车位
        pygame.draw.polygon(self.screen, (0, 200, 0), rotated_corners, 2)
        
        # 绘制目标方向指示器
        dir_x = target_x + int(math.cos(target_yaw - self.vehicle_state[2]) * parking_length * 0.5 * self.scale)
        dir_y = target_y - int(math.sin(target_yaw - self.vehicle_state[2]) * parking_length * 0.5 * self.scale)
        pygame.draw.line(self.screen, (0, 255, 0), (target_x, target_y), (dir_x, dir_y), 3)
        

        # 绘制雷达数据 - 使用新雷达模型
        for i, dist in enumerate(self.lidar.ranges):
            angle = self.lidar.angles[i] + self.vehicle_state[2]
            
            # 转换到屏幕坐标
            dx = dist * math.cos(angle)
            dy = dist * math.sin(angle)
            
            # 旋转以对齐车辆朝向
            rx = dx * math.cos(-self.vehicle_state[2]) - dy * math.sin(-self.vehicle_state[2])
            ry = dx * math.sin(-self.vehicle_state[2]) + dy * math.cos(-self.vehicle_state[2])
            
            # 缩放并移动到中心
            end_x = center_x + int(rx * self.scale)
            end_y = center_y - int(ry * self.scale)  # 注意Y轴方向
            
            # 根据距离设置颜色
            color_intensity = min(255, int(255 * (dist / self.max_range)))
            ray_color = (255 - color_intensity, color_intensity, 0)
            
            pygame.draw.line(self.screen, ray_color, (center_x, center_y), (end_x, end_y), 2)
            
            # 绘制雷达点（仅在探测到障碍物时）
            if dist < self.max_range:
                pygame.draw.circle(self.screen, (0, 150, 0), (end_x, end_y), 3)
        
        # 绘制车辆 (在中心)
        half_length_px = int(self.car_length / 2 * self.scale)
        half_width_px = int(self.car_width / 2 * self.scale)
        
        # 创建车辆矩形
        car_rect = pygame.Rect(
            center_x - half_length_px,
            center_y - half_width_px,
            2 * half_length_px,
            2 * half_width_px
        )
        
        # 绘制车辆（带方向指示）
        pygame.draw.rect(self.screen, (200, 0, 0), car_rect, 2)
        # 绘制车辆前方指示器
        pygame.draw.line(self.screen, (0, 0, 255), 
                        (center_x, center_y),
                        (center_x + half_length_px, center_y), 3)
        
        # 显示状态信息
        font = pygame.font.SysFont(None, 24)
        speed_text = font.render(f"Speed: {self.vehicle_state[3]:.2f} m/s", True, (0, 0, 0))
        steer_text = font.render(f"Steer: {math.degrees(self.vehicle_state[4]):.1f}°", True, (0, 0, 0))
        step_text = font.render(f"Step: {self.step_count}/{self.max_steps}", True, (0, 0, 0))
        dist_text = font.render(f"Dist: {self.prev_dist:.2f} m", True, (0, 0, 0))
        scenario_text = font.render(f"Scenario: {os.path.basename(self.current_scenario)}", True, (0, 0, 0))
        
        self.screen.blit(speed_text, (10, 10))
        self.screen.blit(steer_text, (10, 40))
        self.screen.blit(step_text, (10, 70))
        self.screen.blit(dist_text, (10, 100))
        self.screen.blit(scenario_text, (10, 130))
        
        # 更新显示
        pygame.display.flip()
        self.clock.tick(30)
        
        if mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def _get_observation(self):
        # 使用雷达扫描
        x, y, yaw = self.vehicle_state[:3]
        radar_data = self.lidar.scan(x, y, yaw)
        
        # 组合观测: 雷达数据 + [速度, 转向角]
        state_info = np.array([
            self.vehicle_state[3] / self.max_speed,  # 归一化速度
            self.vehicle_state[4] / self.max_steer,  # 归一化转向角
        ])
        
        return np.concatenate([radar_data, state_info])

    def _get_vehicle_polygon(self):
        """获取车辆的多边形表示（带缓存）"""
        if self.vehicle_poly_cache is not None:
            return self.vehicle_poly_cache
        
        x, y, yaw = self.vehicle_state[:3]
        half_length = self.car_length / 2
        half_width = self.car_width / 2
        
        # 车辆四个角点 (车辆坐标系)
        corners = [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, half_width),
            (half_length, -half_width)
        ]
        
        # 旋转并转换到全局坐标系
        rotated_corners = []
        for cx, cy in corners:
            rx = cx * math.cos(yaw) - cy * math.sin(yaw)
            ry = cx * math.sin(yaw) + cy * math.cos(yaw)
            rotated_corners.append((x + rx, y + ry))
        
        self.vehicle_poly_cache = Polygon(rotated_corners)
        return self.vehicle_poly_cache
    
    def _check_termination(self):
        """检查终止条件，返回(terminated, truncated)"""
        # 检查碰撞（使用雷达数据）
        # if self.last_radar is not None and min(self.last_radar) < self.collision_threshold:
        #     return True, False  # 碰撞导致终止

        # 检查是否到达目标
        dist_to_target = math.hypot(
            self.vehicle_state[0] - self.target_info[0],
            self.vehicle_state[1] - self.target_info[1]
        )
        yaw_diff = abs(self._normalize_angle(self.vehicle_state[2] - self.target_info[2]))
        
        if dist_to_target < 0.5 and yaw_diff < math.radians(10):
            return True, False  # 成功到达目标

        # 检查步数限制
        if self.step_count >= self.max_steps:
            return False, True  # 截断

        return False, False
    
    def _check_collision(self, vehicle_poly):

        return False
    
    def _calculate_reward(self, terminated, truncated):
        """计算奖励"""
                # 基本奖励
        reward = -0.1  # 时间惩罚
        
        # 进度奖励 (向目标靠近)
        current_dist = math.hypot(
            self.vehicle_state[0] - self.target_info[0],
            self.vehicle_state[1] - self.target_info[1]
        )
        
        # 计算距离变化
        dist_diff = self.prev_dist - current_dist
        reward += 5.0 * dist_diff  # 向目标靠近的奖励
        self.prev_dist = current_dist
        
        # 目标达成奖励
        if terminated and current_dist < 0.5:
            reward += 100.0  # 成功到达目标的奖励

        # 速度奖励 (鼓励移动)
        reward += 0.1 * abs(self.vehicle_state[3])
        
        return reward
    
    def close(self):
        """关闭环境，释放资源"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        # 清除缓存
        self.radar_cache = {}
        self.vehicle_poly_cache = None
        
    def plot_scenario(self):
        """使用Matplotlib绘制当前场景"""
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        # 绘制障碍物
        for obs in self.obstacles:
            if len(obs) >= 3:
                poly = MplPolygon(obs, closed=True, fill=True, alpha=0.7, color='gray')
                ax.add_patch(poly)
            elif len(obs) == 2:
                plt.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'k-', linewidth=2)
        
        # 绘制目标位置
        plt.plot(self.target_info[0], self.target_info[1], 'g*', markersize=15, label='Target')
        
        # 绘制车辆
        car_poly = self._get_vehicle_polygon()
        x, y = car_poly.exterior.xy
        plt.fill(x, y, 'r', alpha=0.5, label='Ego Vehicle')
        
        # 绘制雷达数据
        radar_data = self._simulate_radar()
        angles = np.arange(0, 360, self.angular_resolution)
        for i, dist in enumerate(radar_data):
            angle_rad = np.radians(angles[i] + np.degrees(self.vehicle_state[2]))
            x = self.vehicle_state[0] + dist * np.cos(angle_rad)
            y = self.vehicle_state[1] + dist * np.sin(angle_rad)
            plt.plot([self.vehicle_state[0], x], [self.vehicle_state[1], y], 'g-', alpha=0.3)
            plt.plot(x, y, 'go', markersize=3)
        
        # 设置图形属性
        ax.set_aspect('equal')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Parking Scenario')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    config = {
        'data_dir': 'C:\AI_Planner\RL\pygame_input_features_new_withinBEV_no_parallel_parking',
        'max_range': 15.0,
        'timestep': 0.1,
        'max_steps': 500,
        'render_mode': "human", # "human", # 'human'
        # 随机仿真环境代替json环境
        'scenario_mode': 'file',  # 'random' 或 'file'
        'world_size': 30.0,
        'min_obstacles': 5,
        'max_obstacles': 10,
        "manual": False,
    }

    env = ParkingEnv(config)

    # 初始化动作
    action = np.array([0.0, 0.0], dtype=np.float32)
    # 控制参数
    steer_step = 0.1
    throttle_step = 0.1

    if config["render_mode"] == "human":
        # 初始化Pygame视频系统
        pygame.init()
        pygame.display.set_caption("Parking Environment - Manual Control")
        screen = pygame.display.set_mode(env.screen_size)

    start = time.time()
    while True:
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        # 重置动作
        action = np.array([0.0, 0.0], dtype=np.float32)
        
        # # 初始渲染
        if config["render_mode"]=="human":
            env.render()  
        
        while not (terminated or truncated):
            if config["manual"]:
                # 处理键盘事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        pygame.quit()
                        exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            pygame.quit()
                            exit()
                
                # 获取按键状态 (持续按键检测)
                keys = pygame.key.get_pressed()
                
                # 转向控制
                if keys[pygame.K_LEFT]:
                    action[0] = max(-1.0, action[0] - steer_step)
                elif keys[pygame.K_RIGHT]:
                    action[0] = min(1.0, action[0] + steer_step)
                else:
                    # 无转向按键时缓慢回正
                    if action[0] > 0:
                        action[0] = max(0, action[0] - steer_step/2)
                    elif action[0] < 0:
                        action[0] = min(0, action[0] + steer_step/2)
                
                # 油门控制
                if keys[pygame.K_UP]:
                    action[1] = min(1.0, action[1] + throttle_step)
                elif keys[pygame.K_DOWN]:
                    action[1] = max(-1.0, action[1] - throttle_step)
                else:
                    # 无油门按键时缓慢减速
                    if action[1] > 0:
                        action[1] = max(0, action[1] - throttle_step/2)
                    elif action[1] < 0:
                        action[1] = min(0, action[1] + throttle_step/2)
            else:
                action = env.action_space.sample()

            
            # 执行动作
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if config["render_mode"]=="human":
                # 显示当前控制状态
                font = pygame.font.SysFont(None, 24)
                steer_text = font.render(f"Steering: {action[0]:.2f}", True, (0, 0, 255))
                throttle_text = font.render(f"Throttle: {action[1]:.2f}", True, (0, 0, 255))
                screen.blit(steer_text, (10, 220))
                screen.blit(throttle_text, (10, 250))
                pygame.display.flip()
                
                # 控制帧率
                pygame.time.Clock().tick(30)
        
        end = time.time()
        print(end-start)
        # print(f"Total reward: {total_reward}")
    
    env.close()
    if config["render_mode"]=="human":
        pygame.quit()