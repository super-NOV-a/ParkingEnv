# parking_env_pkg/scenario_manager.py
from __future__ import annotations
import json, os, random, math
from pathlib import Path
from typing import List, Tuple, Sequence, Optional

import numpy as np
from shapely.geometry import Polygon, LineString, Point

from .utils import _normalize_angle, parking_corners

Vector = Tuple[float, float]

class ScenarioManager:
    """
    加载 / 随机生成场景。与 Gym 环境零耦合，仅返回
    - ego_info:  (x, y, yaw)
    - target_info: (x, y, yaw)
    - obstacles:  List[List[Vector]]    # 多边形或折线
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_dir = Path(cfg.get("data_dir", "scenarios"))
        self.parking_length = cfg["parking_length"]
        self.parking_width = cfg["parking_width"]
        self.world_size = cfg["world_size"]
        self.car_length = cfg["car_length"]
        self.car_width = cfg["car_width"]
        self.min_obstacles = cfg["min_obstacles"]
        self.max_obstacles = cfg["max_obstacles"]
        self.min_obstacle_size = cfg["min_obstacle_size"]
        self.max_obstacle_size = cfg["max_obstacle_size"]

    # ---------- Public API -------------------------------------------------
    def init(self, *, seed: Optional[int] = None,
             scenario_idx: Optional[int] = None):
        """根据 cfg['scenario_mode'] 返回场景三元组"""
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        mode = self.cfg.get("scenario_mode", "random")
        if mode == "file":
            return self._load_from_file(scenario_idx)
        return self._generate_random()

    # ---------- File mode --------------------------------------------------
    def _load_from_file(self, idx: Optional[int]):
        files = self._scenario_files()
        if idx is None:
            idx = random.randint(0, len(files) - 1)
        file_path = files[idx]
        return (*self._parse_json(file_path), os.path.basename(file_path))

    def _scenario_files(self):
        return sorted(f for f in self.data_dir.glob("*.json"))

    def _parse_json(self, file_path):
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
            _normalize_angle(ego_data[2])
        ]
        
        # 提取目标信息（带坐标转换）
        target_data = data['Frames']['0']['PlanningRequest']['m_targetArea']['m_targetPosture']['m_pose']
        target_info = [
            target_data[0] + m_pathOrigin[0] - nfm_origin[0],
            target_data[1] + m_pathOrigin[1] - nfm_origin[1],
            _normalize_angle(target_data[2])
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

    # ---------- Random mode ------------------------------------------------
    def _generate_random(self, curr_level: int = 0):
        """
        curr_level 0→简单（近距离/少障碍） … n→困难（远距离/多障碍）
        """
        world = self.world_size
        margin = max(self.parking_length, self.car_length) * 0.5

        # ------- 1. 随机目标车位（全图均可） --------------------------
        tx = random.uniform(margin, world - margin)
        ty = random.uniform(margin, world - margin)
        tyaw = random.uniform(0, 2 * math.pi)
        self.target_info = (tx, ty, tyaw)
        target_poly = Polygon(parking_corners(*self.target_info, self.parking_length, self.parking_width))

        # ------- 2. ego 以“同象限+距离壳层”采样 ----------------------
        #   距离范围随 curr_level 线性放大
        min_d = 5.0 + curr_level * 1.5          # meters
        max_d = 10.0 + curr_level * 3.0
        for _ in range(100):                    # 尝试 100 次
            ang = random.uniform(0, 2 * math.pi)
            d   = random.uniform(min_d, max_d)
            ex  = np.clip(tx + d * math.cos(ang), margin, world - margin)
            ey  = np.clip(ty + d * math.sin(ang), margin, world - margin)
            eyaw = random.uniform(0, 2 * math.pi)
            ego_poly = Polygon(parking_corners(ex, ey, eyaw, self.car_length, self.car_width))
            # 与目标/边界/障碍均不碰撞
            if not ego_poly.intersects(target_poly):
                self.ego_info = (ex, ey, eyaw)
                break
        else:
            raise RuntimeError("无法放置 ego，扩大世界或减少障碍")

        # ------- 3. 生成障碍（数量随难度增加） ------------------------
        self.obstacles = []
        n_obs = random.randint(
            max(0, self.min_obstacles - curr_level),
            self.max_obstacles + curr_level
        )
        attempts = 0
        while len(self.obstacles) < n_obs and attempts < n_obs * 20:
            attempts += 1
            poly = self._random_obstacle_polygon(world, margin)
            if (not poly.intersects(target_poly) and
                not poly.intersects(ego_poly)):
                self.obstacles.append(list(poly.exterior.coords)[:-1])  # 去掉闭合点

        return self.ego_info, self.target_info, self.obstacles

    def _random_obstacle_polygon(self, world, margin):
        shape = random.choice(["rect", "poly"])
        if shape == "rect":
            w, h = random.uniform(1,3), random.uniform(1,3)
            cx = random.uniform(margin, world-margin)
            cy = random.uniform(margin, world-margin)
            ang = random.uniform(0, 2*math.pi)
            rect = [(-w/2,-h/2), (-w/2,h/2), (w/2,h/2), (w/2,-h/2)]
            pts = [(cx + px*math.cos(ang)-py*math.sin(ang),
                    cy + px*math.sin(ang)+py*math.cos(ang)) for px,py in rect]
            return Polygon(pts)
        else:
            n = random.randint(3,6)
            r = random.uniform(0.8,2.5)
            cx = random.uniform(margin, world-margin)
            cy = random.uniform(margin, world-margin)
            ang0 = random.uniform(0, 2*math.pi)
            pts = [(cx + r*math.cos(ang0+2*math.pi*i/n),
                    cy + r*math.sin(ang0+2*math.pi*i/n)) for i in range(n)]
            return Polygon(pts)

