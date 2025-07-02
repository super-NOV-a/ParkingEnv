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

random.seed(123)

class ParkingEnv(Env):
    def __init__(self, config):
        # 配置参数
        self.data_dir = config.get('data_dir', 'scenarios')
        self.max_range = config.get('max_range', 10.0)
        self.angular_resolution = config.get('angular_resolution', 10)  # 默认10度提高性能
        self.rays_per_sector = config.get('rays_per_sector', 1)  # 默认1条射线提高性能
        self.dt = config.get('timestep', 0.1)
        self.max_steps = config.get('max_steps', 500)
        self.render_mode = config.get('render_mode', 'human')
        self.scenario_mode = config.get('scenario_mode', 'file')  # 'file' 或 'random'
        self.collision_threshold = config.get('collision_threshold', 0.5)  # 碰撞阈值
        
        # 随机场景参数
        self.world_size = config.get('world_size', 30.0)  # 世界大小（米）
        self.min_obstacles = config.get('min_obstacles', 3)
        self.max_obstacles = config.get('max_obstacles', 8)
        self.min_parking_size = config.get('min_parking_size', 3.0)  # 最小停车位尺寸
        self.max_parking_size = config.get('max_parking_size', 5.0)  # 最大停车位尺寸
        self.min_obstacle_size = config.get('min_obstacle_size', 1.0)  # 最小障碍物尺寸
        self.max_obstacle_size = config.get('max_obstacle_size', 4.0)  # 最大障碍物尺寸
        
        # 车辆参数
        self.wheelbase = 2.5  # 轴距 (米)
        self.max_steer = np.radians(30)  # 最大转向角 (弧度)
        self.max_speed = 2.0  # 最大速度 (米/秒) - 停车场景降低速度
        self.car_length = 4.7  # 车长 (米)
        self.car_width = 1.8   # 车宽 (米)
        
        # 空间定义
        num_bins = 360 // self.angular_resolution
        self.action_space = spaces.Box(
            low=np.array([-1, -1]), 
            high=np.array([1, 1]),  # [转向, 油门]
            dtype=np.float32
        )
        
        # 观测空间: 雷达数据 + [速度, 转向角] + [目标距离, 目标角度, 目标角度差]
        self.observation_space = spaces.Box(
            low=0,
            high=self.max_range,
            shape=(num_bins + 5,),  # 雷达数据 + [v, steer] + [目标距离, 目标角度, 目标角度差]
            dtype=np.float32
        )
        
        # 场景数据
        self.scenario_files = self._get_scenario_files()
        self.current_scenario = None
        self.ego_info = None
        self.target_info = None
        self.obstacles = []
        self.obstacle_geoms = []
        self.obstacle_tree = None
        
        # 车辆状态
        self.vehicle_state = None  # [x, y, yaw, v, steer]
        self.step_count = 0
        self.prev_dist = float('inf')  # 用于奖励计算
        self.last_radar = None  # 缓存最后一次雷达数据
        
        # 渲染相关
        self.screen = None
        self.clock = pygame.time.Clock()
        self.screen_size = (800, 800)  # 渲染窗口大小
        self.scale = 10  # 米到像素的缩放比例
        
        # 性能优化
        self.radar_cache = {}
        self.vehicle_poly_cache = None
        self.last_render_time = 0
        self.render_interval = 0.01  # 渲染间隔 (秒)
    
    def _get_scenario_files(self):
        """获取所有场景JSON文件"""
        files = []
        for f in os.listdir(self.data_dir):
            if f.endswith('.json'):
                files.append(os.path.join(self.data_dir, f))
        return files
    
    def _load_scenario(self, file_path):
        """从JSON文件加载场景"""
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # 提取坐标系原点信息
        nfm_origin = data['Frames']['0'].get("m_nfmOrigin", [0, 0])
        m_pathOrigin = data['Frames']['0']['PlanningRequest'].get("m_origin", [0, 0])
        
        # 提取自车信息并转换坐标系
        ego_data = data['Frames']['0']['PlanningRequest']['m_startPosture']['m_pose']
        ego_info = [
            ego_data[0] + m_pathOrigin[0] - nfm_origin[0],
            ego_data[1] + m_pathOrigin[1] - nfm_origin[1],
            self._normalize_angle(ego_data[2])
        ]
        
        # 提取目标信息并转换坐标系
        target_data = data['Frames']['0']['PlanningRequest']['m_targetArea']['m_targetPosture']['m_pose']
        target_info = [
            target_data[0] + m_pathOrigin[0] - nfm_origin[0],
            target_data[1] + m_pathOrigin[1] - nfm_origin[1],
            self._normalize_angle(target_data[2])
        ]
        
        # 提取障碍物信息
        obstacles = []
        for obj in data['Frames']['0']['NfmAggregatedPolygonObjects']:
            points = []
            if 'nfmPolygonObjectNodes' in obj.keys():
                for point in obj['nfmPolygonObjectNodes']:
                    # 转换坐标系并添加到点列表
                    x = point['m_x'] + m_pathOrigin[0] - nfm_origin[0]
                    y = point['m_y'] + m_pathOrigin[1] - nfm_origin[1]
                    points.append((x, y))
                obstacles.append(points)
        
        return ego_info, target_info, obstacles
    
    def _generate_random_scenario(self):
        """生成随机障碍物和停车位场景"""
        # 生成停车位（目标位置）
        parking_size = random.uniform(self.min_parking_size, self.max_parking_size)
        parking_orientation = random.uniform(0, 2 * math.pi)  # 随机朝向
        
        # 确保停车位在边界内
        padding = parking_size * 1.5
        target_x = random.uniform(padding, self.world_size - padding)
        target_y = random.uniform(padding, self.world_size - padding)
        self.target_info = [target_x, target_y, parking_orientation]
        
        # 生成自车初始位置（与停车位保持一定距离）
        min_start_dist = parking_size * 2
        max_start_dist = parking_size * 4
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(min_start_dist, max_start_dist)
        ego_x = target_x + dist * math.cos(angle)
        ego_y = target_y + dist * math.sin(angle)
        
        # 确保自车在边界内
        ego_x = np.clip(ego_x, padding, self.world_size - padding)
        ego_y = np.clip(ego_y, padding, self.world_size - padding)
        
        # 自车朝向随机方向
        ego_yaw = random.uniform(0, 2 * math.pi)
        self.ego_info = [ego_x, ego_y, ego_yaw]
        
        # 生成随机障碍物
        self.obstacles = []
        num_obstacles = random.randint(self.min_obstacles, self.max_obstacles)
        
        for _ in range(num_obstacles):
            # 尝试生成不重叠的障碍物
            for attempt in range(10):  # 最多尝试10次
                # 随机选择障碍物类型（矩形或多边形）
                obstacle_type = random.choice(['rectangle', 'polygon'])
                
                if obstacle_type == 'rectangle':
                    # 生成矩形障碍物
                    width = random.uniform(self.min_obstacle_size, self.max_obstacle_size)
                    height = random.uniform(self.min_obstacle_size, self.max_obstacle_size)
                    x = random.uniform(0, self.world_size)
                    y = random.uniform(0, self.world_size)
                    angle = random.uniform(0, 2 * math.pi)
                    
                    # 计算四个角点
                    half_w = width / 2
                    half_h = height / 2
                    corners = [
                        (-half_w, -half_h),
                        (-half_w, half_h),
                        (half_w, half_h),
                        (half_w, -half_h)
                    ]
                    
                    # 旋转并平移
                    obstacle = []
                    for cx, cy in corners:
                        rx = cx * math.cos(angle) - cy * math.sin(angle)
                        ry = cx * math.sin(angle) + cy * math.cos(angle)
                        obstacle.append((x + rx, y + ry))
                
                else:  # 多边形
                    num_sides = random.randint(3, 6)  # 3-6边形
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
                
                # 检查是否与自车或目标重叠
                poly = Polygon(obstacle)
                ego_point = Point(self.ego_info[0], self.ego_info[1])
                target_point = Point(self.target_info[0], self.target_info[1])
                
                min_dist_to_ego = poly.distance(ego_point)
                min_dist_to_target = poly.distance(target_point)
                
                # 确保障碍物不与自车或目标太近
                if min_dist_to_ego > self.car_length and min_dist_to_target > parking_size:
                    self.obstacles.append(obstacle)
                    break
        
        return self.ego_info, self.target_info, self.obstacles
    
    def _normalize_angle(self, angle):
        """将角度归一化到[-π, π]范围内"""
        return math.atan2(math.sin(angle), math.cos(angle))
    
    def _get_target_info_for_obs(self):
        """获取目标信息的极坐标表示（相对于车辆）"""
        # 车辆位置和朝向
        ego_x, ego_y, ego_yaw = self.vehicle_state[:3]
        
        # 目标位置和朝向
        target_x, target_y, target_yaw = self.target_info
        
        # 计算目标相对于车辆的位置向量
        dx = target_x - ego_x
        dy = target_y - ego_y
        
        # 计算距离
        distance = math.sqrt(dx**2 + dy**2)
        
        # 计算全局角度
        global_angle = math.atan2(dy, dx)
        
        # 计算相对于车辆朝向的角度
        relative_angle = self._normalize_angle(global_angle - ego_yaw)
        
        # 计算目标朝向与车辆朝向的差值
        yaw_diff = self._normalize_angle(target_yaw - ego_yaw)
        
        return distance, relative_angle, yaw_diff
    
    def step(self, action):
        """
        执行一个时间步
        """
        # 解析动作
        steer_cmd = np.clip(action[0], -1, 1)
        throttle_cmd = np.clip(action[1], -1, 1)
        
        # 应用转向滤波
        current_steer = self.vehicle_state[4]
        new_steer = current_steer * 0.7 + steer_cmd * self.max_steer * 0.3
        
        # 更新车辆状态 (简化自行车模型)
        x, y, yaw, v, _ = self.vehicle_state
        
        # 简化动力学模型
        new_v = np.clip(v + throttle_cmd * 1.0 * self.dt, -self.max_speed, self.max_speed)
        
        # 更新位置
        new_x = x + new_v * math.cos(yaw) * self.dt
        new_y = y + new_v * math.sin(yaw) * self.dt
        
        # 更新朝向 (考虑转向)
        if abs(new_steer) > 1e-5:
            turn_radius = self.wheelbase / math.tan(new_steer)
            angular_velocity = new_v / turn_radius
            new_yaw = yaw + angular_velocity * self.dt
        else:
            new_yaw = yaw
        
        new_yaw = self._normalize_angle(new_yaw)
        
        self.vehicle_state = np.array([new_x, new_y, new_yaw, new_v, new_steer])
        self.vehicle_poly_cache = None  # 清空车辆多边形缓存
        
        # 获取观测
        obs = self._get_observation()
        
        # 检查终止条件
        terminated, truncated = self._check_termination()
        
        # 计算奖励
        reward = self._calculate_reward(terminated, truncated)
        
        # 更新步数
        self.step_count += 1
        
        # 渲染 - 使用帧数控制而非时间间隔
        if self.render_mode == 'human' and self.step_count % 5 == 0:  # 每5步渲染一次
            self.render()

        return obs, reward, terminated, truncated, {}
    
    def reset(self, scenario_idx=None, seed=None, options=None):
        """重置环境"""
        # 根据场景模式加载或生成场景
        if self.scenario_mode == 'random':
            # 随机生成场景
            self.ego_info, self.target_info, self.obstacles = self._generate_random_scenario()
            self.current_scenario = "Random Scenario"
        else:
            # 从文件加载场景
            if scenario_idx is None:
                scenario_idx = np.random.randint(0, len(self.scenario_files))
            
            self.current_scenario = self.scenario_files[scenario_idx]
            self.ego_info, self.target_info, self.obstacles = self._load_scenario(self.current_scenario)
        
        # 创建障碍物几何对象和空间索引
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
        
        # 创建空间索引
        self.obstacle_tree = STRtree(self.obstacle_geoms)
        
        # 初始化车辆状态
        self.vehicle_state = np.array([
            self.ego_info[0],
            self.ego_info[1],
            self.ego_info[2],
            0.0,  # 初始速度
            0.0   # 初始转向角
        ])
        
        self.step_count = 0
        self.prev_dist = float('inf')
        self.radar_cache = {}
        self.vehicle_poly_cache = None
        self.last_radar = None  # 重置雷达缓存
        
        # 返回初始观测
        obs = self._get_observation()
        return obs, {}
    
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
        
        # 绘制雷达数据 - 关键修改：使用实际探测距离
        radar_data = self._simulate_radar()
        for i, dist in enumerate(radar_data):
            angle = math.radians(i * self.angular_resolution)
            # 使用实际距离而不是最大距离
            end_x = center_x + int(dist * self.scale * math.cos(angle))
            end_y = center_y - int(dist * self.scale * math.sin(angle))  # 注意Y轴方向
            
            # 根据距离设置颜色（近处红色，远处绿色）
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
        
        # 获取目标信息用于显示
        target_dist, target_angle, yaw_diff = self._get_target_info_for_obs()
        
        # 显示状态信息
        font = pygame.font.SysFont(None, 24)
        speed_text = font.render(f"Speed: {self.vehicle_state[3]:.2f} m/s", True, (0, 0, 0))
        steer_text = font.render(f"Steer: {math.degrees(self.vehicle_state[4]):.1f}°", True, (0, 0, 0))
        step_text = font.render(f"Step: {self.step_count}/{self.max_steps}", True, (0, 0, 0))
        dist_text = font.render(f"Dist: {target_dist:.2f} m", True, (0, 0, 0))
        angle_text = font.render(f"Angle: {math.degrees(target_angle):.1f}°", True, (0, 0, 0))
        yaw_diff_text = font.render(f"Yaw Diff: {math.degrees(yaw_diff):.1f}°", True, (0, 0, 0))
        scenario_text = font.render(f"Scenario: {os.path.basename(self.current_scenario)}", True, (0, 0, 0))
        
        self.screen.blit(speed_text, (10, 10))
        self.screen.blit(steer_text, (10, 40))
        self.screen.blit(step_text, (10, 70))
        self.screen.blit(dist_text, (10, 100))
        self.screen.blit(angle_text, (10, 130))
        self.screen.blit(yaw_diff_text, (10, 160))
        self.screen.blit(scenario_text, (10, 190))
        
        # 更新显示
        pygame.display.flip()
        self.clock.tick(30)
        
        if mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def _get_observation(self):
        """获取当前观测"""
        # 获取雷达数据
        radar_data = self._simulate_radar()
        self.last_radar = radar_data  # 缓存最后一次雷达数据
        
        # 获取目标信息
        target_dist, target_angle, yaw_diff = self._get_target_info_for_obs()
        
        # 组合观测: 雷达数据 + [速度, 转向角] + [目标距离, 目标角度, 目标角度差]
        state_info = np.array([
            self.vehicle_state[3] / self.max_speed,  # 归一化速度
            self.vehicle_state[4] / self.max_steer,  # 归一化转向角
            min(target_dist / self.max_range, 1.0),  # 归一化距离 (限制在[0,1])
            target_angle / math.pi,  # 归一化角度 [-1,1]
            yaw_diff / math.pi  # 归一化角度差 [-1,1]
        ])
        
        return np.concatenate([radar_data, state_info])
    
    def _simulate_radar(self):
        """精确的雷达模拟，计算每个扇区内到障碍物的最短距离"""
        state_hash = hash(tuple(self.vehicle_state[:3]))
        if state_hash in self.radar_cache:
            return self.radar_cache[state_hash]
        
        num_bins = 360 // self.angular_resolution
        radar_data = np.full(num_bins, self.max_range)
        
        ego_x, ego_y, ego_yaw = self.vehicle_state[:3]
        ego_point = Point(ego_x, ego_y)
        
        # 创建一个圆形查询区域
        query_circle = ego_point.buffer(self.max_range)
        
        # 查询在探测范围内的障碍物
        candidate_indices = self.obstacle_tree.query(query_circle)
        candidates = [self.obstacle_geoms[i] for i in candidate_indices]
        
        # 为每个扇区创建边界线
        sector_boundaries = []
        for i in range(num_bins):
            start_angle = math.radians(i * self.angular_resolution)
            end_angle = math.radians((i + 1) * self.angular_resolution)
            
            # 创建扇区边界线
            start_line = LineString([
                (ego_x, ego_y),
                (ego_x + self.max_range * math.cos(start_angle + ego_yaw), 
                ego_y + self.max_range * math.sin(start_angle + ego_yaw))
            ])
            
            end_line = LineString([
                (ego_x, ego_y),
                (ego_x + self.max_range * math.cos(end_angle + ego_yaw), 
                ego_y + self.max_range * math.sin(end_angle + ego_yaw))
            ])
            
            sector_boundaries.append((start_line, end_line))
        
        # 处理每个候选障碍物
        for geom in candidates:
            if not isinstance(geom, BaseGeometry):
                continue
                
            # 计算障碍物到车辆的最小距离
            min_dist = geom.distance(ego_point)
            if min_dist >= self.max_range:
                continue
                
            # 获取几何体的边界（对于点、线、多边形都适用）
            if isinstance(geom, Point):
                boundary = geom
            elif isinstance(geom, LineString):
                boundary = geom
            elif isinstance(geom, Polygon):
                boundary = geom.exterior
            else:
                boundary = geom
            
            # 处理每个扇区
            for bin_idx in range(num_bins):
                # 获取当前扇区的边界线
                start_line, end_line = sector_boundaries[bin_idx]
                
                # 创建扇区楔形
                sector_wedge = Polygon([
                    (ego_x, ego_y),
                    (start_line.coords[1]),
                    (end_line.coords[1]),
                    (ego_x, ego_y)
                ])
                
                # 检查障碍物是否与扇区相交
                if not boundary.intersects(sector_wedge):
                    continue
                    
                # 计算扇区内到障碍物的最短距离
                if boundary.within(sector_wedge):
                    # 如果障碍物完全在扇区内，直接计算距离
                    dist = boundary.distance(ego_point)
                else:
                    # 计算扇区内的交点
                    intersection = boundary.intersection(sector_wedge)
                    if intersection.is_empty:
                        continue
                        
                    # 找到扇区内最近的点
                    dist = intersection.distance(ego_point)
                
                # 更新该扇区的距离
                if dist < radar_data[bin_idx]:
                    radar_data[bin_idx] = dist
        
        self.radar_cache[state_hash] = radar_data
        return radar_data

    def _get_closest_distance(self, geometry, ego_point):
        """获取交点到自车点的最近距离"""
        if geometry.is_empty:
            return None
        elif isinstance(geometry, Point):
            return geometry.distance(ego_point)
        elif isinstance(geometry, LineString):
            return geometry.distance(ego_point)
        elif hasattr(geometry, 'geoms'):
            return min(
                (g.distance(ego_point) for g in geometry.geoms if isinstance(g, (Point, LineString))),
                default=None
            )
        else:
            return None
    
    def _calculate_ray_distance(self, ego_point, bin_idx, dx, dy):
        """计算单条射线的距离（用于多进程）"""
        ray_end = (ego_point.x + dx * self.max_range, 
                  ego_point.y + dy * self.max_range)
        ray = LineString([(ego_point.x, ego_point.y), ray_end])
        
        min_dist = self.max_range
        candidates = self.obstacle_tree.query(ray)
        
        for geom in candidates:
            if not isinstance(geom, BaseGeometry) or not ray.intersects(geom):
                continue
                
            inter = ray.intersection(geom)
            if inter.is_empty:
                continue
                
            if isinstance(inter, Point):
                dist = ego_point.distance(inter)
            elif isinstance(inter, MultiPoint):
                dist = min(ego_point.distance(pt) for pt in inter.geoms)
            elif isinstance(inter, LineString):
                # 取最近的点
                closest_point = inter.interpolate(inter.project(ego_point))
                dist = ego_point.distance(closest_point)
            else:
                dist = ego_point.distance(inter)
                
            if dist < min_dist:
                min_dist = dist
                # 提前终止：如果已经非常接近，停止检查其他障碍物
                if min_dist < 0.5:
                    break
        
        return bin_idx, min_dist
    
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
        if self.last_radar is not None and min(self.last_radar) < self.collision_threshold:
            return True, False  # 碰撞导致终止

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
        
        # 碰撞惩罚
        if terminated and self.last_radar is not None and min(self.last_radar) < self.collision_threshold:
            reward -= 50.0  # 碰撞惩罚
        
        # 速度奖励 (鼓励移动)
        reward += 0.1 * abs(self.vehicle_state[3])
        
        # 安全奖励 (避免近距离障碍)
        if self.last_radar is not None:
            min_radar = min(self.last_radar)
            if min_radar < 1.5:
                reward -= (1.5 - min_radar) * 5.0  # 近距离障碍惩罚
        
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


# 修改 __main__ 部分
if __name__ == "__main__":
    config = {
        'data_dir': 'C:\\AI_Planner\\RL\\pygame_input_features_new_withinBEV_no_parallel_parking',
        'max_range': 30.0,
        'angular_resolution': 5,
        'rays_per_sector': 1,
        'timestep': 0.1,
        'max_steps': 500,
        'render_mode': None, # 'human',
        'scenario_mode': 'random',  # 或 'file'
        'world_size': 40.0,
        'min_obstacles': 5,
        'max_obstacles': 10,
        'collision_threshold': 0.5
    }

    env = ParkingEnv(config)

    # 初始化动作
    action = np.array([0.0, 0.0], dtype=np.float32)
    # 控制参数
    steer_step = 0.1
    throttle_step = 0.1

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
        # env.render()
        
        while not (terminated or truncated):
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
            
            # 执行动作
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
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
    pygame.quit()