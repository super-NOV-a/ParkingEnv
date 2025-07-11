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
             scenario_idx: Optional[int] = None,
             current_level: Optional[int] = 0,):
        """根据 cfg['scenario_mode'] 返回场景三元组"""
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        mode = self.cfg.get("scenario_mode", "random")
        if mode == "file":
            return self._load_from_file(scenario_idx)
        if mode == "empty": # 如果需要边框，可以在random时将occupy_prob设为0
            return self._generate_empty(current_level)
        return self._generate_random(current_level)

    # ---------- File mode --------------------------------------------------
    def _load_from_file(self, idx: Optional[int]):
        files = self._scenario_files()
        if not files:
            raise FileNotFoundError(
                f"No *.json scenario files found in {self.data_dir.resolve()}\n"
                "→ 请确认 data_dir 是否正确，或改用 scenario_mode='random'"
            )
        if idx is None:
            idx = random.randint(0, len(files) - 1)
        file_path = files[idx]
        return self._parse_json(file_path) # (*self._parse_json(file_path), os.path.basename(file_path))

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
        生成“主路 + 障碍区”场景，
        车位朝向在 {0°, 90°, 45°} 中随机选一种，gap 不变。
        """
        # ---------- 0. 读配置 / 常量 --------------------------
        cfg  = self.cfg
        W    = self.world_size
        pl, pw = self.parking_length, self.parking_width
        cl, cw = self.car_length,  self.car_width
        margin      = cfg.get("margin", 1.0)
        gap         = cfg.get("gap", 4.0)
        wall_thick  = cfg.get("wall_thickness", 0.2)
        occupy_prob = min(0.9, cfg.get("occupy_prob", 0.5) + 0.05 * curr_level)

        # ---------- 1. 主路位置（两栏 / 三栏随机） --------------
        layout_mode = random.choice(("two", "three"))
        if layout_mode == "two":
            half = W / 2
            road_side = cfg.get("road_side", random.choice(("left", "right")))
            if road_side == "left":
                road_ymin, road_ymax = 0.0, half
                obs_bands = [(half, W)]
            else:
                road_ymin, road_ymax = half, W
                obs_bands = [(0.0, half)]
        else:                                  # "three"
            third = W / 3
            road_ymin, road_ymax = third, 2 * third
            obs_bands = [(0.0, third), (2 * third, W)]

        # ---------- 2. 随机车位朝向 ---------------------------
        yaw = random.choice([0.0, math.pi / 2, math.pi / 4])

        # 旋转后包络尺寸（用于排布）
        bbox_x = abs(pl * math.cos(yaw)) + abs(pw * math.sin(yaw))
        bbox_y = abs(pl * math.sin(yaw)) + abs(pw * math.cos(yaw))

        # ---------- 3. 生成车位网格 ---------------------------
        parking_spots: list[tuple[Polygon, tuple[float, float, float]]] = []
        used_polys:    list[Polygon] = []

        for ymin, ymax in obs_bands:
            # 靠近主路那一边
            if ymax == road_ymin:                # 障碍区在主路下方
                row_y = ymax - bbox_y / 2 - margin
            else:                                # 障碍区在主路上方
                row_y = ymin + bbox_y / 2 + margin

            x_start = margin + bbox_x / 2
            x_end   = W - margin - bbox_x / 2
            step    = bbox_x + gap
            n_cols  = max(1, int((x_end - x_start) / step) + 1)

            for i in range(n_cols):
                tx = x_start + i * step
                ty = row_y
                spot_poly = Polygon(
                    parking_corners(tx, ty, yaw, pl, pw)
                )
                parking_spots.append((spot_poly, (tx, ty, yaw)))

        # ---------- 4. 目标车位 + 障碍 ------------------------
        target_poly, target_info = random.choice(parking_spots)
        obstacles: list[list[tuple[float, float]]] = []

        for sp, _ in parking_spots:
            if sp.equals(target_poly):
                continue
            if random.random() < occupy_prob:
                obstacles.append(list(sp.exterior.coords)[:-1])
                used_polys.append(sp)

        # ---------- 5. ego 采样（主路矩形内） ------------------
        for _ in range(100):
            ex = random.uniform(margin + cl / 2, W - margin - cl / 2)
            ey = random.uniform(road_ymin + cw / 2, road_ymax - cw / 2)
            eyaw = random.uniform(-math.pi, math.pi)
            ego_poly = Polygon(parking_corners(ex, ey, eyaw, cl, cw))
            if not any(ego_poly.intersects(p) for p in used_polys):
                ego_info = (ex, ey, eyaw)
                break
        else:
            raise RuntimeError("无法为 ego 找到合法初始位，扩大 world_size 或降低密度")

        # ---------- 6. 场景边墙（可选） ------------------------
        if wall_thick > 0:
            walls = [
                [(0, 0), (W, 0), (W, wall_thick), (0, wall_thick)],
                [(0, W - wall_thick), (W, W - wall_thick), (W, W), (0, W)],
                [(0, 0), (wall_thick, 0), (wall_thick, W), (0, W)],
                [(W - wall_thick, 0), (W, 0), (W, W), (W - wall_thick, W)]
            ]
            obstacles.extend(walls)

        # ---------- 7. 返回 ----------------------------------
        self.ego_info, self.target_info, self.obstacles = (
            ego_info, target_info, obstacles)
        return self.ego_info, self.target_info, self.obstacles

    def _generate_random_legacy(self, curr_level: int = 0):
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

    def _generate_empty(self, curr_level: int = 0):
        """
        curr_level, 根据level调整环境大小, 0→简单（近距离） … n→困难（远距离）
        """
        world = self.world_size
        margin = max(self.parking_length, self.car_length) * 0.5

        # ------- 1. 随机目标车位（全图均可） --------------------------
        tx = random.uniform(margin, world - margin)
        ty = random.uniform(margin, world - margin)
        tyaw = random.uniform(0, 2 * math.pi)
        self.target_info = (tx, ty, tyaw)

        # ------- 2. ego 以“同象限+距离壳层”采样 ----------------------
        #   距离范围随 curr_level 线性放大
        min_d = 2.0 + curr_level * 1.5          # meters
        max_d = 10.0 + curr_level * 3.0     # 最远40米
        ang = random.uniform(0, 2 * math.pi)
        d   = random.uniform(min_d, max_d)
        ex  = np.clip(tx + d * math.cos(ang), margin, world - margin)
        ey  = np.clip(ty + d * math.sin(ang), margin, world - margin)
        eyaw = random.uniform(0, 2 * math.pi)
        self.ego_info = (ex, ey, eyaw)

        return self.ego_info, self.target_info, []


