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
import time
import random
from lidar import Lidar2D
from vehicle import Vehicle

np.random.seed(123)
random.seed(123)


class ParkingEnv(Env):
    def __init__(self, config):
        # Configuration parameters
        self.data_dir = config.get('data_dir', 'scenarios')
        self.dt = config.get('timestep', 0.1)
        self.max_range = 30.0
        self.max_steps = config.get('max_steps', 500)
        self.render_mode = config.get('render_mode', 'human')
        self.scenario_mode = config.get('scenario_mode', 'random')
        self.collision_threshold = config.get('collision_threshold', 0.5)
        
        # 设置车辆尺寸
        self.wheelbase = 3.0    # 轴距
        self.front_hang = 1.0   # 前悬
        self.rear_hang = 1.0    # 后悬
        self.car_width = 2.0    # 车宽
        # 设置目标车位尺寸为5.5m×2.3m
        self.parking_length = 5.5  # 停车位长度
        self.parking_width = 2.3   # 停车位宽度
        self.car_length = self.wheelbase + self.front_hang + self.rear_hang
        
        # Random scenario parameters
        self.world_size = config.get('world_size', 30.0)
        self.min_obstacles = config.get('min_obstacles', 3)
        self.max_obstacles = config.get('max_obstacles', 8)
        self.min_obstacle_size = config.get('min_obstacle_size', 1.0)
        self.max_obstacle_size = config.get('max_obstacle_size', 10.0)
        
        # Vehicle dynamics # 添加方向状态变量
        self.direction = 1  # 1表示前进，-1表示后退

        # Vehicle dynamics
        self.max_steer = np.radians(30)
        self.max_speed = 5.0
        self.steer_filter_factor = 0.7  # 转向滤波因子

        self.vehicle = Vehicle(
            wheelbase=self.wheelbase,
            width=self.car_width,
            front_hang=self.front_hang,
            rear_hang=self.rear_hang,
            max_steer=self.max_steer,
            max_speed=self.max_speed,
            dt=self.dt,
            steer_filter=self.steer_filter_factor,
        )

        # Lidar configuration - 确保雷达在车辆中心
        lidar_config = {
            'range_min': 0.5,
            'max_range': config.get('max_range', 10.0),
            'angle_range': 360,
            'num_beams': 72,
            'noise': False,
            'std': 0.05,
            'angle_std': 0.5,
            'position_offset': (0, 0)  # 雷达在车辆中心
        }
        self.lidar = Lidar2D(lidar_config)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([0.0] * lidar_config['num_beams'] + [-1.0, -1.0, 0.0, -1.0, -1.0]),
            high=np.array([self.max_range] * lidar_config['num_beams'] + [1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Action space
        self.action_space = spaces.Box(
            low=np.array([-1, -1]), 
            high=np.array([1, 1]),
            dtype=np.float32
        )
        
        # Scenario data
        if self.scenario_mode == 'random':
            self.current_scenario = "Random Scenario" 
        elif self.scenario_mode == 'file':
            self.scenario_files = self._get_scenario_files()
        else:
            print("No such ", (self.scenario_mode), " mode")
        self.ego_info = None
        self.target_info = None
        self.obstacles = []
        self.obstacle_geoms = []
        
        # Vehicle state
        self.step_count = 0
        self.prev_dist = float('inf')
        self.vehicle_poly = None
        self.target_poly = None
        
        # Rendering
        self.screen = None
        self.clock = pygame.time.Clock()
        self.screen_size = (800, 800)
        self.scale = 10

    def _create_parking_polygon(self, center_x, center_y, orientation):
        """根据中心点和方向创建停车位矩形"""
        half_l = self.parking_length / 2
        half_w = self.parking_width / 2
        
        # 定义未旋转前的四个角点（相对中心）
        corners_local = [
            (-half_l, -half_w),
            (-half_l, half_w),
            (half_l, half_w),
            (half_l, -half_w)
        ]
        
        # 旋转并平移角点
        corners_global = []
        for x, y in corners_local:
            # 旋转
            rx = x * math.cos(orientation) - y * math.sin(orientation)
            ry = x * math.sin(orientation) + y * math.cos(orientation)
            # 平移
            corners_global.append((center_x + rx, center_y + ry))
        
        return Polygon(corners_global)

    def step(self, action):
        self.vehicle.state, self.vehicle.direction = self.vehicle.step(action)

        # 更新车辆多边形
        self.vehicle_poly = self.vehicle.get_shapely_polygon()

        # 获取观测
        obs = self._get_observation()

        # 终止判断
        terminated, truncated, collised = self._check_termination()

        # 计算奖励
        reward = self._calculate_reward(terminated, collised)

        self.step_count += 1

        if self.render_mode == 'human' and self.step_count % 5 == 0:
            self.render()

        return obs, reward, terminated, truncated, {}

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self, scenario_idx=None, seed=None, options=None):
        if seed is not None:
            self.seed(seed)  # 你前面已经实现了 seed()
        # Load or generate scenario
        if self.scenario_mode == 'random':
            self.ego_info, self.target_info, self.obstacles = self._generate_random_scenario()
            self.current_scenario = "Random Scenario"
        else:
            if scenario_idx is None:
                scenario_idx = np.random.randint(0, len(self.scenario_files))
            self.current_scenario = self.scenario_files[scenario_idx]
            self.ego_info, self.target_info, self.obstacles = self._load_scenario(self.current_scenario)
        
        # Create obstacle geometries
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

        # Update lidar obstacles
        self.lidar.update_obstacles(self.obstacle_geoms)
        
        # Initialize vehicle state
        self.vehicle.reset_state(self.ego_info[0], self.ego_info[1], self.ego_info[2])
        self.vehicle_poly = self.vehicle.get_shapely_polygon()
        # self.target_poly = Polygon(self.vehicle.create_polygon(
        #     self.target_info[0], self.target_info[1], self.target_info[2], target=True
        # ))
        
        self.step_count = 0
        self.prev_dist = float('inf')
        
        return self._get_observation(), {}

    def render(self, mode=None):
        """Optimized rendering with vehicle polygon caching"""
        if mode is None:
            mode = self.render_mode
        
        # Initialize Pygame
        if self.screen is None and mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Parking Environment")
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.close()
                return
        
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Calculate view center
        center_x, center_y = self.screen_size[0] // 2, self.screen_size[1] // 2
        
        # Draw obstacles
        for obs in self.obstacles:
            pygame_points = []
            for x, y in obs:
                dx = x - self.vehicle.state[0]
                dy = y - self.vehicle.state[1]
                rx = dx * math.cos(-self.vehicle.state[2]) - dy * math.sin(-self.vehicle.state[2])
                ry = dx * math.sin(-self.vehicle.state[2]) + dy * math.cos(-self.vehicle.state[2])
                screen_x = center_x + int(rx * self.scale)
                screen_y = center_y - int(ry * self.scale)
                pygame_points.append((screen_x, screen_y))
            
            if len(pygame_points) >= 3:
                pygame.draw.polygon(self.screen, (100, 100, 100), pygame_points)
            elif len(pygame_points) == 2:
                pygame.draw.line(self.screen, (100, 100, 100), pygame_points[0], pygame_points[1], 2)
        
        # 绘制目标位置和朝向 - 使用正确的停车位尺寸
        parking_length = self.parking_length
        parking_width = self.parking_width
        
        # 计算停车位角点（相对于目标中心）
        corners = [
            (-parking_length/2, -parking_width/2),
            (-parking_length/2, parking_width/2),
            (parking_length/2, parking_width/2),
            (parking_length/2, -parking_width/2)
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
            dx = px - self.vehicle.state[0]
            dy = py - self.vehicle.state[1]
            # 旋转以对齐车辆朝向
            vx = dx * math.cos(-self.vehicle.state[2]) - dy * math.sin(-self.vehicle.state[2])
            vy = dx * math.sin(-self.vehicle.state[2]) + dy * math.cos(-self.vehicle.state[2])
            # 转换到屏幕坐标
            screen_x = center_x + int(vx * self.scale)
            screen_y = center_y - int(vy * self.scale)
            rotated_corners.append((screen_x, screen_y))
        
        # 绘制停车位
        pygame.draw.polygon(self.screen, (0, 200, 0), rotated_corners, 2)
        
        # 添加车位朝向指示器（箭头）
        # 计算车位中心点
        center_dx = self.target_info[0] - self.vehicle.state[0]
        center_dy = self.target_info[1] - self.vehicle.state[1]
        center_rx = center_dx * math.cos(-self.vehicle.state[2]) - center_dy * math.sin(-self.vehicle.state[2])
        center_ry = center_dx * math.sin(-self.vehicle.state[2]) + center_dy * math.cos(-self.vehicle.state[2])
        target_center_x = center_x + int(center_rx * self.scale)
        target_center_y = center_y - int(center_ry * self.scale)
        
        # 计算箭头方向（车头方向）
        arrow_length = parking_length * 0.4 * self.scale
        arrow_end_x = target_center_x + int(math.cos(target_yaw - self.vehicle.state[2]) * arrow_length)
        arrow_end_y = target_center_y - int(math.sin(target_yaw - self.vehicle.state[2]) * arrow_length)
        
        # 绘制箭头
        pygame.draw.line(self.screen, (0, 255, 0), (target_center_x, target_center_y), 
                    (arrow_end_x, arrow_end_y), 3)
        
        # 绘制箭头头部（三角形）
        arrow_head_size = 10
        angle = math.atan2(arrow_end_y - target_center_y, arrow_end_x - target_center_x)
        head_point1 = (
            arrow_end_x - arrow_head_size * math.cos(angle - math.pi/6),
            arrow_end_y - arrow_head_size * math.sin(angle - math.pi/6)
        )
        head_point2 = (
            arrow_end_x - arrow_head_size * math.cos(angle + math.pi/6),
            arrow_end_y - arrow_head_size * math.sin(angle + math.pi/6)
        )
        pygame.draw.polygon(self.screen, (0, 255, 0), 
                        [(arrow_end_x, arrow_end_y), head_point1, head_point2])
        
        # 绘制雷达数据 - 修复雷达位置问题
        for i, dist in enumerate(self.lidar.ranges):
            # 雷达角度是相对于车辆朝向的局部角度
            local_angle = self.lidar.angles[i]
            
            # 计算雷达点在车辆坐标系中的位置（相对车辆中心）
            local_x = dist * math.cos(local_angle)
            local_y = dist * math.sin(local_angle)
            
            # 直接转换到屏幕坐标
            end_x = center_x + int(local_x * self.scale)
            end_y = center_y - int(local_y * self.scale)  # 注意：屏幕坐标Y轴向下
            
            color_intensity = min(255, int(255 * (dist / self.max_range)))
            ray_color = (255 - color_intensity, color_intensity, 0)
            
            # 从车辆中心（屏幕中心）到雷达点绘制直线
            pygame.draw.line(self.screen, ray_color, (center_x, center_y), (end_x, end_y), 1)
            
            if dist < self.max_range:
                pygame.draw.circle(self.screen, (0, 150, 0), (end_x, end_y), 3)
        
        # 绘制车辆
        if self.vehicle_poly:
            exterior = list(self.vehicle_poly.exterior.coords)
            pygame_points = []
            for x, y in exterior:
                dx = x - self.vehicle.state[0]
                dy = y - self.vehicle.state[1]
                rx = dx * math.cos(-self.vehicle.state[2]) - dy * math.sin(-self.vehicle.state[2])
                ry = dx * math.sin(-self.vehicle.state[2]) + dy * math.cos(-self.vehicle.state[2])
                screen_x = center_x + int(rx * self.scale)
                screen_y = center_y - int(ry * self.scale)
                pygame_points.append((screen_x, screen_y))
            pygame.draw.polygon(self.screen, (200, 0, 0), pygame_points, 2)
        
        # 绘制车辆方向箭头（红色）
        arrow_length = 2.0 * self.scale  # 箭头长度（屏幕像素）
        end_x = center_x + int(arrow_length)  # 车辆前方
        end_y = center_y
        pygame.draw.line(self.screen, (255, 0, 0), (center_x, center_y), (end_x, end_y), 3)
        
        # 绘制箭头头部（三角形）
        arrow_head_size = 8
        angle = 0  # 车辆坐标系中前方为0度
        head_point1 = (
            end_x - arrow_head_size * math.cos(angle - math.pi/6),
            end_y - arrow_head_size * math.sin(angle - math.pi/6)
        )
        head_point2 = (
            end_x - arrow_head_size * math.cos(angle + math.pi/6),
            end_y - arrow_head_size * math.sin(angle + math.pi/6)
        )
        pygame.draw.polygon(self.screen, (255, 0, 0), [(end_x, end_y), head_point1, head_point2])

        # Display info
        font = pygame.font.SysFont(None, 24)
        texts = [
            f"Speed: {self.vehicle.state[3]:.2f} m/s",
            f"Steer: {math.degrees(self.vehicle.state[4]):.1f}°",
            f"Step: {self.step_count}/{self.max_steps}",
            f"Dist: {self.prev_dist:.2f} m",
            f"Scenario: {os.path.basename(self.current_scenario)}"
        ]

        # === 新增目标信息 ===
        dist_to_target = math.hypot(self.target_info[0] - self.vehicle.state[0],
                                    self.target_info[1] - self.vehicle.state[1])
        angle_to_target = math.atan2(self.target_info[1] - self.vehicle.state[1],
                                     self.target_info[0] - self.vehicle.state[0])
        rel_angle = self._normalize_angle(angle_to_target - self.vehicle.state[2])
        heading_diff = self._normalize_angle(self.target_info[2] - self.vehicle.state[2])

        texts.extend([
            f"TargetDist: {dist_to_target:.2f} m",
            f"RelAngle: {math.degrees(rel_angle):.1f}°",
            f"HeadingDiff: {math.degrees(heading_diff):.1f}°"
        ])

        # === 渲染所有文本（包括新增的）===
        for i, text in enumerate(texts):
            surf = font.render(text, True, (0, 0, 0))
            self.screen.blit(surf, (10, 10 + i * 30))

        # Update display
        pygame.display.flip()
        self.clock.tick(30)
        
        if mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def _check_termination(self):
        """Improved termination check using polygon overlap"""
        # Check collision
        if self._check_collision():
            print("碰撞")
            return True, False, True
        
        # Calculate overlap with target area
        if self.target_poly and self.vehicle_poly:
            overlap = self.vehicle_poly.intersection(self.target_poly).area
            vehicle_area = self.vehicle_poly.area
            overlap_ratio = overlap / vehicle_area
            
            # Calculate orientation difference
            yaw_diff = abs(self._normalize_angle(
                self.vehicle.state[2] - self.target_info[2]
            ))
            
            # Check success conditions
            if overlap_ratio > 0.9 and yaw_diff < math.radians(10):
                print("成功")
                return True, False, False
        
        # Check step limit
        if self.step_count >= self.max_steps:
            return False, True, False
        
        return False, False, False
    
    def _check_collision(self):
        """Precise collision detection using Shapely"""
        if not self.vehicle_poly:
            return False
        
        for obstacle in self.obstacle_geoms:
            if self.vehicle_poly.intersects(obstacle):
                return True
        
        return False

    def _calculate_reward(self, terminated, collised):
        # 基本信息提取
        x, y, yaw = self.vehicle.state[:3]
        tx, ty, target_yaw = self.target_info

        # 计算相对位置信息
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)

        angle_to_target = math.atan2(dy, dx)
        rel_angle = self._normalize_angle(angle_to_target - yaw)
        heading_diff = self._normalize_angle(target_yaw - yaw)

        # 初始化 reward
        reward = 0.0

        # --- 基础奖励项 ---
        reward -= 0.2 * dist  # 鼓励靠近
        reward -= 0.1 * abs(rel_angle)  # 鼓励朝向目标
        reward -= 0.05 * abs(heading_diff)  # 鼓励姿态一致
        reward -= 0.01  # 每步惩罚

        if terminated:
            if collised:
                reward -= 5.0  # 撞击惩罚
            else:
                reward += 10.0  # 成功奖励

        return reward

    def _get_scenario_files(self):
        """仅在文件模式下获取场景JSON文件"""
        files = []
        for f in os.listdir(self.data_dir):
            if f.endswith('.json'):
                files.append(os.path.join(self.data_dir, f))
        return files
    
    def _load_scenario(self, file_path):
        """从JSON文件加载场景，过滤目标车位内的障碍物"""
        # 修复硬编码问题
        # file_path = "C:\AI_Planner\RL\pygame_input_features_new_withinBEV_no_parallel_parking/1713602139487329202.json"
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        nfm_origin = data['Frames']['0'].get("m_nfmOrigin", [0, 0])
        m_pathOrigin = data['Frames']['0']['PlanningRequest'].get("m_origin", [0, 0])
        
        # 提取自车信息（带坐标转换）
        ego_data = data['Frames']['0']['PlanningRequest']['m_startPosture']['m_pose']
        ego_info = [
            ego_data[0] + m_pathOrigin[0] - nfm_origin[0],
            ego_data[1] + m_pathOrigin[1] - nfm_origin[1],
            self._normalize_angle(ego_data[2])
        ]
        
        # 提取目标信息（带坐标转换）
        target_data = data['Frames']['0']['PlanningRequest']['m_targetArea']['m_targetPosture']['m_pose']
        target_info = [
            target_data[0] + m_pathOrigin[0] - nfm_origin[0],
            target_data[1] + m_pathOrigin[1] - nfm_origin[1],
            self._normalize_angle(target_data[2])
        ]
        
        # 构建目标车位坐标变换矩阵
        cos_t = np.cos(target_info[2])
        sin_t = np.sin(target_info[2])
        tx, ty = target_info[0], target_info[1]
        trans_matrix = np.array([
            [cos_t, sin_t, -tx * cos_t - ty * sin_t],
            [-sin_t, cos_t, tx * sin_t - ty * cos_t],
            [0, 0, 1]
        ])
        
        # 提取并过滤障碍物
        obstacles = []
        for obj in data['Frames']['0']['NfmAggregatedPolygonObjects']:
            if 'nfmPolygonObjectNodes' not in obj:
                continue
                
            polygon = []
            for point in obj['nfmPolygonObjectNodes']:
                # 应用坐标转换
                x = point['m_x'] + m_pathOrigin[0] - nfm_origin[0]
                y = point['m_y'] + m_pathOrigin[1] - nfm_origin[1]
                polygon.append([x, y])
            
            # 转换到目标车位坐标系
            polygon_arr = np.array(polygon).T
            homogenous = np.vstack([polygon_arr, np.ones(polygon_arr.shape[1])])
            transformed = trans_matrix @ homogenous
            target_coords = transformed[:2, :].T
            
            # 使用正确的停车位尺寸
            vehicle_x_min, vehicle_x_max = -self.parking_length/2, self.parking_length/2
            vehicle_y_min, vehicle_y_max = -self.parking_width/2, self.parking_width/2
            
            # 检查是否在目标车位边界内
            in_target = False
            for x, y in target_coords:
                if (vehicle_x_min <= x <= vehicle_x_max and 
                    vehicle_y_min <= y <= vehicle_y_max):
                    in_target = True
                    break
                    
            # 只保留目标车位外的障碍物
            if not in_target:
                obstacles.append(polygon)
        
        return ego_info, target_info, obstacles

    def _generate_random_scenario(self):
        """生成随机障碍物和停车位场景"""
        # 使用指定的长宽生成停车位
        parking_length = self.parking_length
        parking_width = self.parking_width
        
        # 计算停车位外接圆半径用于安全检测
        parking_diag = math.sqrt(parking_length**2 + parking_width**2)
        parking_radius = parking_diag / 2
        
        parking_orientation = random.uniform(0, 2 * math.pi)
        
        # 使用外接圆半径计算安全边界
        padding = parking_radius * 3  # 扩大安全边界
        target_x = random.uniform(padding, self.world_size - padding)
        target_y = random.uniform(padding, self.world_size - padding)
        self.target_info = [target_x, target_y, parking_orientation]
        
        # 计算车辆外接圆半径
        car_radius = math.sqrt(self.car_length**2 + self.car_width**2) / 2
        
        # 生成自车初始位置（使用外接圆半径计算安全距离）
        min_start_dist = (parking_radius + car_radius) * 2
        max_start_dist = (parking_radius + car_radius) * 4
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
                
                # 使用外接圆半径进行安全检测
                if min_dist_to_ego > car_radius and min_dist_to_target > parking_radius:
                    self.obstacles.append(obstacle)
                    break
        
        return self.ego_info, self.target_info, self.obstacles

    def _normalize_angle(self, angle):
        """角度归一化"""
        return math.atan2(math.sin(angle), math.cos(angle))

    def _get_observation(self):
        # 使用雷达扫描 - 确保雷达在车辆中心
        x, y, yaw = self.vehicle.state[:3]
        radar_data = self.lidar.scan(x, y, yaw)

        # 自车中心到目标点的向量
        dx = self.target_info[0] - x
        dy = self.target_info[1] - y

        # 目标在自车坐标系下的极坐标（距离+方向）
        distance = math.hypot(dx, dy)
        angle_to_target = math.atan2(dy, dx)
        relative_angle = self._normalize_angle(angle_to_target - yaw)

        # 目标朝向与车辆朝向的差距
        heading_diff = self._normalize_angle(self.target_info[2] - yaw)

        # 组合观测: 雷达 + 归一化速度/转向角 + 目标距离 + 相对角度 + 朝向差距
        state_info = np.array([
            self.vehicle.state[3] / self.max_speed,  # 归一化速度
            self.vehicle.state[4] / self.max_steer,  # 归一化转向角
            distance / self.world_size,  # 归一化距离
            relative_angle / np.pi,  # [-1, 1]
            heading_diff / np.pi  # [-1, 1]
        ])

        return np.concatenate([radar_data, state_info])

    def close(self):
        """关闭环境，释放资源"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None


if __name__ == "__main__":
    config = {
        'data_dir': 'C:\AI_Planner\RL\pygame_input_features_new_withinBEV_no_parallel_parking',
        'max_range': 15.0,
        'timestep': 0.1,
        'max_steps': 500,
        'render_mode': "human",  # "human", # 'human  'None, #
        'scenario_mode': 'random',  # 'random' 或 'file'
        'world_size': 30.0,
        'min_obstacles': 0,
        'max_obstacles': 1,
        "manual": True,
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
                        # 添加重置快捷键
                        elif event.key == pygame.K_r:
                            terminated = True
                
                # 获取按键状态 (持续按键检测)
                keys = pygame.key.get_pressed()

                # 键盘控制方向映射
                if keys[pygame.K_LEFT]:
                    action[0] = max(-1.0, action[0] + steer_step)
                elif keys[pygame.K_RIGHT]:
                    action[0] = min(1.0, action[0] - steer_step)
                else:
                    # 回正
                    if action[0] > 0:
                        action[0] = max(0, action[0] - steer_step / 2)
                    elif action[0] < 0:
                        action[0] = min(0, action[0] + steer_step / 2)

                # 油门控制
                if keys[pygame.K_UP]:
                    action[1] = min(1.0, action[1] + 2*throttle_step)
                elif keys[pygame.K_DOWN]:
                    action[1] = max(-1.0, action[1] - throttle_step)
                else:
                    # 无油门按键时缓慢减速
                    if action[1] > 0:
                        action[1] = max(0, action[1] - throttle_step)
                    elif action[1] < 0:
                        action[1] = min(0, action[1] + throttle_step)
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
                screen.blit(steer_text, (10, 250))
                screen.blit(throttle_text, (10, 280))
                pygame.display.flip()
                
                # 控制帧率
                pygame.time.Clock().tick(30)
        
        end = time.time()
        print(f"Episode time: {end-start:.2f} seconds")
        # print(f"Total reward: {total_reward}")
    
    env.close()
    if config["render_mode"]=="human":
        pygame.quit()
