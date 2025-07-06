import numpy as np
import math
from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPoint
from shapely.strtree import STRtree
from shapely.ops import unary_union, nearest_points  # 添加 nearest_points

class Lidar2D:
    def __init__(self, config):
        self.range_min = config.get('range_min', 0.5)
        self.range_max = config.get('max_range', 10.0)
        self.angle_range = math.radians(config.get('angle_range', 270))  # 视野范围 (弧度)
        self.num_beams = config.get('num_beams', 72)  # 射线数量
        self.noise = config.get('noise', True)
        self.std = config.get('std', 0.05)  # 距离噪声标准差
        self.angle_std = math.radians(config.get('angle_std', 0.5))  # 角度噪声标准差
        
        # 计算射线角度
        self.angle_min = -self.angle_range / 2
        self.angle_max = self.angle_range / 2
        self.angle_inc = self.angle_range / (self.num_beams - 1) if self.num_beams > 1 else 0
        self.angles = np.linspace(self.angle_min, self.angle_max, self.num_beams)
        
        # 初始化数据
        self.ranges = self.range_max * np.ones(self.num_beams)
        self.obstacle_tree = None
        self.obstacles = []

    def update_obstacles(self, obstacles):
        """更新障碍物列表和空间索引"""
        self.obstacles = obstacles
        self.obstacle_tree = STRtree(obstacles) if obstacles else None

    def scan(self, x, y, yaw):
        """执行一次扫描"""
        if not self.obstacle_tree:
            self.ranges = self.range_max * np.ones(self.num_beams)
            return self.ranges.copy()
        
        origin = Point(x, y)
        rays = []
        
        # 生成所有射线
        for i, angle in enumerate(self.angles):
            # 添加角度噪声
            actual_angle = angle + yaw
            if self.noise:
                actual_angle += np.random.normal(0, self.angle_std)
                
            # 计算射线终点
            end_x = x + self.range_max * math.cos(actual_angle)
            end_y = y + self.range_max * math.sin(actual_angle)
            ray = LineString([(x, y), (end_x, end_y)])
            rays.append(ray)
        
        # 批量查询可能相交的障碍物
        min_dists = self.range_max * np.ones(self.num_beams)
        
        # 处理每条射线 - 使用STRtree优化查询
        for i, ray in enumerate(rays):
            # 使用STRtree查询可能与当前射线相交的障碍物
            candidate_geoms = self.obstacle_tree.query(ray)

            if len(candidate_geoms) == 0:
                continue

            merged_obstacles = unary_union(candidate_geoms)

            # 计算交点
            intersection = ray.intersection(merged_obstacles)
            
            if intersection.is_empty:
                continue
                
            # 统一处理交点：找到最近的交点
            if intersection.geom_type in ['Point', 'LineString', 'Polygon', 'MultiPoint', 'MultiLineString']:
                # 使用 nearest_points 方法找到离原点最近的点
                closest_point = nearest_points(origin, intersection)[1]
                dist = origin.distance(closest_point)
            else:
                dist = self.range_max
                
            # 应用距离噪声
            if self.noise:
                dist += np.random.normal(0, self.std)
                
            # 确保在有效范围内
            if self.range_min < dist < self.range_max:
                min_dists[i] = dist
        
        self.ranges = min_dists
        return min_dists.copy()
