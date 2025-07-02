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
        
        # 观测空间: 雷达数据 + [速度, 转向角]
        self.observation_space = spaces.Box(
            low=0,
            high=self.max_range,
            shape=(num_bins + 2,),  # 雷达数据 + [v, steer]
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
    
    def _normalize_angle(self, angle):
        """将角度归一化到[-π, π]范围内"""
        return math.atan2(math.sin(angle), math.cos(angle))
    
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
        done = self._check_termination()
        
        # 计算奖励
        reward = self._calculate_reward(done)
        
        # 更新步数
        self.step_count += 1
        
        # 渲染 - 使用帧数控制而非时间间隔
        if self.render_mode == 'human' and self.step_count % 5 == 0:  # 每5步渲染一次
            self.render()

        return obs, reward, done, {}
    
    def reset(self, scenario_idx=None):
        """重置环境"""
        # 选择场景
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
        
        # 返回初始观测
        return self._get_observation()

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
        
        # 绘制目标位置
        dx = self.target_info[0] - self.vehicle_state[0]
        dy = self.target_info[1] - self.vehicle_state[1]
        rx = dx * math.cos(-self.vehicle_state[2]) - dy * math.sin(-self.vehicle_state[2])
        ry = dx * math.sin(-self.vehicle_state[2]) + dy * math.cos(-self.vehicle_state[2])
        target_x = center_x + int(rx * self.scale)
        target_y = center_y - int(ry * self.scale)
        pygame.draw.circle(self.screen, (0, 255, 0), (target_x, target_y), 8)
        
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
        """获取当前观测"""
        # 获取雷达数据
        radar_data = self._simulate_radar()
        
        # 组合观测: 雷达数据 + [速度, 转向角]
        state_info = np.array([
            self.vehicle_state[3] / self.max_speed,  # 归一化速度
            self.vehicle_state[4] / self.max_steer,  # 归一化转向角
        ])
        
        return np.concatenate([radar_data, state_info])

    def _simulate_radar(self):
        """雷达模拟：每个扇区内发出多条射线，检测最近障碍物距离"""
        state_hash = hash(tuple(self.vehicle_state[:3]))
        if state_hash in self.radar_cache:
            return self.radar_cache[state_hash]

        num_bins = 360 // self.angular_resolution
        radar_data = np.full(num_bins, self.max_range)
        
        ego_x, ego_y, ego_yaw = self.vehicle_state[:3]
        ego_point = Point(ego_x, ego_y)

        # 构建障碍物几何体（以防缓存未命中）
        geometries = self.obstacle_geoms

        for bin_idx in range(num_bins):
            min_dist = self.max_range
            bin_center_deg = bin_idx * self.angular_resolution

            # 在每个扇区内发射 rays_per_sector 条射线
            for j in range(self.rays_per_sector):
                offset = (j + 1) / (self.rays_per_sector + 1)
                angle_deg = bin_center_deg + (offset - 0.5) * self.angular_resolution
                angle_global = math.radians(angle_deg) + ego_yaw

                dx = self.max_range * math.cos(angle_global)
                dy = self.max_range * math.sin(angle_global)
                ray = LineString([(ego_x, ego_y), (ego_x + dx, ego_y + dy)])

                for geom in geometries:
                    if not ray.intersects(geom):
                        continue
                    inter = ray.intersection(geom)
                    dist = self._get_closest_distance(inter, ego_point)
                    if dist is not None:
                        min_dist = min(min_dist, dist)

            radar_data[bin_idx] = min_dist

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
        """检查终止条件"""
        # 检查碰撞
        car_poly = self._get_vehicle_polygon()
        if self._check_collision(car_poly):
            return True
        
        # 检查是否到达目标
        dist_to_target = math.hypot(
            self.vehicle_state[0] - self.target_info[0],
            self.vehicle_state[1] - self.target_info[1]
        )
        yaw_diff = abs(self._normalize_angle(self.vehicle_state[2] - self.target_info[2]))
        
        if dist_to_target < 0.5 and yaw_diff < math.radians(10):
            return True
        
        # 检查步数限制
        if self.step_count >= self.max_steps:
            return True
        
        return False
    
    def _check_collision(self, vehicle_poly):
        """修复类型错误的碰撞检测"""
        candidates = self.obstacle_tree.query(vehicle_poly)
        
        # 确保所有候选都是有效几何体
        for geom in candidates:
            if not isinstance(geom, BaseGeometry):
                continue
            if vehicle_poly.intersects(geom):
                return True
        return False
    
    def _calculate_reward(self, done):
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
        if done and current_dist < 0.5:
            reward += 100.0  # 成功到达目标的奖励
        
        # 碰撞惩罚
        if done and self._check_collision(self._get_vehicle_polygon()):
            reward -= 50.0  # 碰撞惩罚
        
        # 速度奖励 (鼓励移动)
        reward += 0.1 * abs(self.vehicle_state[3])
        
        # 安全奖励 (避免近距离障碍)
        min_radar = min(self._simulate_radar())
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

if __name__ == "__main__":
    config = {
        'data_dir': 'C:\AI_Planner\RL\pygame_input_features_new_withinBEV_no_parallel_parking',
        'max_range': 30.0,
        'angular_resolution': 20,  # 降低分辨率
        'rays_per_sector': 1,     # 减少射线数
        'timestep': 0.1,
        'max_steps': 500,
        'render_mode': 'human',
    }

    env = ParkingEnv(config)

    while True:
        # 测试环境
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            # 使用随机策略
            action = env.action_space.sample()
            
            # 执行一步
            obs, reward, done, _ = env.step(action)
            print(obs)
            # total_reward += reward
            
        print(f"Total reward: {total_reward}")
    env.close()