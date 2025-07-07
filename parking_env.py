"""Unified Parking Environment (continuous & discrete)
=====================================================
This file merges *parking_env_continuous.py* and *parking_env_discrete.py* into
one cohesive module.  The class **ParkingEnv** selects its control mode
(`continuous` or `discrete`) from the inbound *config* dict and instantiates the
corresponding vehicle model.

Common helper routines that do not depend on environment state have been moved
to *utils.py* to avoid duplication.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame
from gymnasium import Env, spaces
from shapely.geometry import LineString, Point, Polygon

from lidar import Lidar2D
from utils import _normalize_angle, parking_corners
from vehicle import VehicleContinuous, VehicleDiscrete

# -----------------------------------------------------------------------------
# Helper types
# -----------------------------------------------------------------------------
Vector = Tuple[float, float]

np.random.seed(123)
random.seed(123)


class ParkingEnv(Env):
    """Parking task that supports *continuous* or *discrete* control.

    Parameters
    ----------
    config : dict
        *control_mode*: ``"continuous"`` or ``"discrete"`` (default
        ``"continuous"``)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "ParkingEnv"}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        cfg = config or {}

        # ----- global & scenario settings --------------------------------
        self.control_mode: str = cfg.get("control_mode", "continuous")
        self.data_dir: str = cfg.get("data_dir", "scenarios")
        self.dt: float = cfg.get("timestep", 0.1)
        self.max_steps: int = cfg.get("max_steps", 500)
        self.render_mode: str = cfg.get("render_mode", "human")
        self.scenario_mode: str = cfg.get("scenario_mode", "random")

        # ----- vehicle / parking geometry --------------------------------
        self.wheelbase: float = 3.0
        self.front_hang: float = 1.0
        self.rear_hang: float = 1.0
        self.car_width: float = 2.0
        self.parking_length: float = 5.5
        self.parking_width: float = 2.3
        self.car_length: float = self.wheelbase + self.front_hang + self.rear_hang

        # ----- world & random obstacles ----------------------------------
        self.world_size: float = cfg.get("world_size", 30.0)
        self.min_obstacles: int = cfg.get("min_obstacles", 3)
        self.max_obstacles: int = cfg.get("max_obstacles", 8)
        self.min_obstacle_size: float = cfg.get("min_obstacle_size", 1.0)
        self.max_obstacle_size: float = cfg.get("max_obstacle_size", 10.0)

        # ----- vehicle limits -------------------------------------------
        self.max_steer = math.radians(30.0)
        self.max_speed: float = cfg.get("max_speed", 3.0)
        self.steer_filter_factor = 0.7

        # -----------------------------------------------------------------
        # Vehicle instantiation depending on control mode
        # -----------------------------------------------------------------
        if self.control_mode == "discrete":
            self.vehicle = VehicleDiscrete(
                wheelbase=self.wheelbase,
                width=self.car_width,
                front_hang=self.front_hang,
                rear_hang=self.rear_hang,
                max_steer=self.max_steer,
                max_speed=self.max_speed,
                dt=self.dt,
            )
            self.action_space = spaces.Discrete(9)
        else:
            self.vehicle = VehicleContinuous(
                wheelbase=self.wheelbase,
                width=self.car_width,
                front_hang=self.front_hang,
                rear_hang=self.rear_hang,
                max_steer=self.max_steer,
                max_speed=self.max_speed,
                dt=self.dt,
                steer_filter=self.steer_filter_factor,
            )
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        # -----------------------------------------------------------------
        # Lidar & observation space
        # -----------------------------------------------------------------
        lidar_cfg = {
            "range_min": 0.5,
            "max_range": cfg.get("lidar_max_range", 30.0),
            "angle_range": 360,
            "num_beams": 72,
            "noise": False,
            "std": 0.05,
            "angle_std": 0.5,
            "position_offset": (0.0, 0.0),
        }
        self.lidar_max_range = lidar_cfg["max_range"]
        self.lidar = Lidar2D(lidar_cfg)

        state_low = [-1.0, -1.0, 0.0, -1.0, -1.0]
        state_high = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.observation_space = spaces.Box(
            low=np.array([0.0] * lidar_cfg["num_beams"] + state_low, dtype=np.float32),
            high=np.array([self.lidar_max_range] * lidar_cfg["num_beams"] + state_high, dtype=np.float32),
            dtype=np.float32,
        )

        # -----------------------------------------------------------------
        # Scenario bookkeeping & render buffers
        # -----------------------------------------------------------------
        self._reset_scenario_buffers()

        # Pygame render buffers – created lazily on first render
        self.screen = None
        self.clock = pygame.time.Clock()
        self.screen_size = (800, 800)
        self.scale = 10  # world‑to‑screen scaling factor

    # ---------------------------------------------------------------------
    # Gym required API
    # ---------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self, *, seed: Optional[int] = None, options=None, scenario_idx: Optional[int] = None):
        if seed is not None:
            self.seed(seed)
        self._load_or_generate_scenario(scenario_idx)
        self.vehicle.reset_state(*self.ego_info)
        self.vehicle_poly = self.vehicle.get_shapely_polygon()

        self.step_count = 0
        self.prev_dist = float("inf")
        return self._get_observation(), {}

    def step(self, action):
        # Vehicle kinematics
        self.vehicle.state, _ = self.vehicle.step(action)
        self.vehicle_poly = self.vehicle.get_shapely_polygon()

        # Obs & reward
        obs = self._get_observation()
        terminated, truncated, collised = self._check_termination()
        reward = self._calculate_reward(terminated, collised)
        self.step_count += 1

        # Render every 5 steps if requested
        if self.render_mode == "human" and self.step_count % 5 == 0:
            self.render()

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Scenario initialisation helpers
    # ------------------------------------------------------------------
    def _reset_scenario_buffers(self):
        self.ego_info: Optional[Tuple[float, float, float]] = None
        self.target_info: Optional[Tuple[float, float, float]] = None
        self.obstacles: List = []
        self.obstacle_geoms: List[Polygon | LineString | Point] = []
        self.vehicle_poly: Optional[Polygon] = None
        self.target_poly: Optional[Polygon] = None
        self.step_count: int = 0
        self.prev_dist: float = float("inf")

    def _load_or_generate_scenario(self, scenario_idx: Optional[int]):
        if self.scenario_mode == "file":
            if not hasattr(self, "_scenario_files"):
                self._scenario_files = self._get_scenario_files()
            if scenario_idx is None:
                scenario_idx = random.randint(0, len(self._scenario_files) - 1)
            path = self._scenario_files[scenario_idx]
            self.ego_info, self.target_info, self.obstacles = self._load_scenario(path)
            self.current_scenario = os.path.basename(path)
        else:
            self.ego_info, self.target_info, self.obstacles = self._generate_random_scenario()
            self.current_scenario = "Random Scenario"

        # Build shapely objects + lidar occupancy
        self.target_poly = Polygon(parking_corners(*self.target_info, self.parking_length, self.parking_width))
        self.obstacle_geoms = []
        for obs in self.obstacles:
            if len(obs) >= 3:
                poly = Polygon(obs)
                self.obstacle_geoms.append(poly if poly.is_valid else poly.buffer(0))
            elif len(obs) == 2:
                self.obstacle_geoms.append(LineString(obs))
            elif len(obs) == 1:
                self.obstacle_geoms.append(Point(obs[0]))
        self.lidar.update_obstacles(self.obstacle_geoms)

    # ------------------------------------------------------------------
    # Reward & Observation
    # ------------------------------------------------------------------
    def _calculate_reward(self, terminated: bool, collised: bool) -> float:
        x, y, yaw = self.vehicle.state[:3]
        tx, ty, tyaw = self.target_info

        dist = math.hypot(tx - x, ty - y)
        # angle_to_target = math.atan2(ty - y, tx - x)
        # rel_angle = _normalize_angle(angle_to_target - yaw)
        # heading_diff = _normalize_angle(tyaw - yaw)

        # --- 纯距离奖励：距离越小，奖励越高 --------------------------
        # 把距离 d 映射到[0,1] 然后计算 0.1/x -1
        reward = (self.world_size/dist) * 0.1 - 1
        # clip 保证落在 [-1, 1]
        reward = min(10.0, reward)
        return reward


    def _get_observation(self):
        x, y, yaw = self.vehicle.state[:3]
        radar_data = self.lidar.scan(x, y, yaw)

        dx, dy = self.target_info[0] - x, self.target_info[1] - y
        distance = math.hypot(dx, dy)
        angle_to_target = math.atan2(dy, dx)
        relative_angle = _normalize_angle(angle_to_target - yaw)
        heading_diff = _normalize_angle(self.target_info[2] - yaw)

        state_info = np.array([
            self.vehicle.state[3] / self.max_speed,
            self.vehicle.state[4] / self.max_steer,
            distance / self.world_size,
            relative_angle / math.pi,
            heading_diff / math.pi,
        ], dtype=np.float32)

        return np.concatenate([radar_data.astype(np.float32), state_info])

    # ------------------------------------------------------------------
    # Termination checks
    # ------------------------------------------------------------------
    def _check_collision(self) -> bool:
        return self.vehicle_poly is not None and any(self.vehicle_poly.intersects(o) for o in self.obstacle_geoms)

    def _check_termination(self):
        # 1) collision
        if self._check_collision():
            # print("碰撞")
            return True, False, True

        # 2) parked successfully
        if self.target_poly and self.vehicle_poly:
            overlap = self.vehicle_poly.intersection(self.target_poly).area
            if overlap / self.vehicle_poly.area > 0.9:
                yaw_diff = abs(_normalize_angle(self.vehicle.state[2] - self.target_info[2]))
                if yaw_diff < math.radians(10):
                    # ======= 修改前 =======
                    # return True, False, False    # 会触发 reset
                    # ======= 修改后 =======
                    return True, False, False     # 继续同一回合

        # 3) out of bounds (too far from target)
        x, y, _ = self.vehicle.state[:3]
        if math.hypot(self.target_info[0] - x, self.target_info[1] - y) > self.world_size * 1.2:
            # print("出界")
            return False, True, True

        # 4) max steps
        if self.step_count >= self.max_steps:
            # print("到最大时长")
            return False, True, False

        return False, False, False

    # ------------------------------------------------------------------
    # Scenario IO helpers (unchanged from original)
    # ------------------------------------------------------------------
    def _get_scenario_files(self):
        return [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".json")]

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

    def _get_parking_corners(self, x, y, yaw):
        """根据车位中心位置和朝向，生成车位四个角点（顺时针）"""
        half_w = self.parking_width / 2
        l_f = self.parking_length / 2
        l_r = -self.parking_length / 2

        # 局部坐标下四角点
        corners = [
            (l_r, -half_w),
            (l_r, half_w),
            (l_f, half_w),
            (l_f, -half_w),
        ]

        # 旋转 + 平移到世界坐标
        rotated = []
        for cx, cy in corners:
            rx = cx * math.cos(yaw) - cy * math.sin(yaw)
            ry = cx * math.sin(yaw) + cy * math.cos(yaw)
            rotated.append((x + rx, y + ry))
        return rotated

    # ------------------------------------------------------------------
    # Random scenario generator (minimal changes)
    # ------------------------------------------------------------------
    def _generate_random_scenario(self):
        world = self.world_size
        margin = max(self.parking_length, self.car_length) * 0.5

        # 四个象限：(xmin, xmax, ymin, ymax)
        quadrants = {
            0: (0, world/2, 0, world/2),      # 左下
            1: (world/2, world, 0, world/2),  # 右下
            2: (0, world/2, world/2, world),  # 左上
            3: (world/2, world, world/2, world)  # 右上
        }

        def sample_in(qid):
            x0, x1, y0, y1 = quadrants[qid]
            x = random.uniform(x0 + margin, x1 - margin)
            y = random.uniform(y0 + margin, y1 - margin)
            yaw = random.uniform(0, 2 * math.pi)
            return (x, y, yaw)

        # 随机选择象限
        ego_q = random.randint(0, 3)
        other_qs = [q for q in range(4) if q != ego_q]
        target_q = random.choice(other_qs)

        self.ego_info = sample_in(ego_q)
        self.target_info = sample_in(target_q)

        # 构造目标车位 polygon 用于后续障碍排除
        target_poly = Polygon(self._get_parking_corners(*self.target_info))

        # 障碍物生成
        self.obstacles = []
        max_attempts = 10
        for _ in range(random.randint(self.min_obstacles, self.max_obstacles)):
            for _ in range(max_attempts):
                obs_q = random.choice(other_qs)
                x0, x1, y0, y1 = quadrants[obs_q]
                shape_type = random.choice(["rect", "poly"])

                if shape_type == "rect":
                    w, h = random.uniform(1.0, 3.0), random.uniform(1.0, 3.0)
                    cx = random.uniform(x0 + margin, x1 - margin)
                    cy = random.uniform(y0 + margin, y1 - margin)
                    angle = random.uniform(0, 2 * math.pi)
                    rect = [(-w/2, -h/2), (-w/2, h/2), (w/2, h/2), (w/2, -h/2)]
                    obs = [(cx + px * math.cos(angle) - py * math.sin(angle),
                            cy + px * math.sin(angle) + py * math.cos(angle)) for px, py in rect]
                else:
                    n = random.randint(3, 6)
                    r = random.uniform(1.0, 3.0)
                    cx = random.uniform(x0 + margin, x1 - margin)
                    cy = random.uniform(y0 + margin, y1 - margin)
                    ang0 = random.uniform(0, 2 * math.pi)
                    obs = [(cx + r * math.cos(ang0 + 2 * math.pi * i / n),
                            cy + r * math.sin(ang0 + 2 * math.pi * i / n)) for i in range(n)]

                poly = Polygon(obs)
                if not poly.intersects(target_poly):
                    self.obstacles.append(obs)
                    break  # 成功生成该障碍，进入下一个障碍循环

        return self.ego_info, self.target_info, self.obstacles
    # def _generate_random_scenario(self):
    #     parking_diag = math.hypot(self.parking_length, self.parking_width)
    #     parking_radius = parking_diag / 2.0
    #     padding = parking_radius * 3.0

    #     target_x = random.uniform(padding, self.world_size - padding)
    #     target_y = random.uniform(padding, self.world_size - padding)
    #     target_yaw = random.uniform(0, 2 * math.pi)
    #     self.target_info = (target_x, target_y, target_yaw)

    #     car_radius = math.hypot(self.car_length, self.car_width) / 2.0
    #     min_d = (parking_radius + car_radius) * 2.0
    #     max_d = min_d * 2.0
    #     angle = random.uniform(0, 2 * math.pi)
    #     dist = random.uniform(min_d, max_d)
    #     ego_x = np.clip(target_x + dist * math.cos(angle), padding, self.world_size - padding)
    #     ego_y = np.clip(target_y + dist * math.sin(angle), padding, self.world_size - padding)
    #     ego_yaw = random.uniform(0, 2 * math.pi)
    #     self.ego_info = (ego_x, ego_y, ego_yaw)

    #     # Obstacles
    #     obstacles: List[List[Vector]] = []
    #     num_obs = random.randint(self.min_obstacles, self.max_obstacles)
    #     for _ in range(num_obs):
    #         for _attempt in range(10):
    #             kind = random.choice(["rect", "poly"])
    #             if kind == "rect":
    #                 w, h = random.uniform(self.min_obstacle_size, self.max_obstacle_size), random.uniform(
    #                     self.min_obstacle_size, self.max_obstacle_size
    #                 )
    #                 x = random.uniform(0, self.world_size)
    #                 y = random.uniform(0, self.world_size)
    #                 ang = random.uniform(0, 2 * math.pi)
    #                 half_w, half_h = w / 2, h / 2
    #                 rect = [
    #                     (-half_w, -half_h),
    #                     (-half_w, half_h),
    #                     (half_w, half_h),
    #                     (half_w, -half_h),
    #                 ]
    #                 obs = [
    #                     (
    #                         x + cx * math.cos(ang) - cy * math.sin(ang),
    #                         y + cx * math.sin(ang) + cy * math.cos(ang),
    #                     )
    #                     for cx, cy in rect
    #                 ]
    #             else:  # polygon 3‑6 sides
    #                 n = random.randint(3, 6)
    #                 r = random.uniform(self.min_obstacle_size, self.max_obstacle_size)
    #                 x = random.uniform(r, self.world_size - r)
    #                 y = random.uniform(r, self.world_size - r)
    #                 ang0 = random.uniform(0, 2 * math.pi)
    #                 obs = [
    #                     (x + r * math.cos(ang0 + 2 * math.pi * i / n), y + r * math.sin(ang0 + 2 * math.pi * i / n))
    #                     for i in range(n)
    #                 ]

    #             if (
    #                 Polygon(obs).distance(Point(self.ego_info[0], self.ego_info[1])) > car_radius
    #                 and Polygon(obs).distance(Point(self.target_info[0], self.target_info[1])) > parking_radius
    #             ):
    #                 obstacles.append(obs)
    #                 break
    #     self.obstacles = obstacles
    #     return self.ego_info, self.target_info, self.obstacles

    # ------------------------------------------------------------------
    # Rendering (unchanged except for _normalize_angle import)
    # ------------------------------------------------------------------
    def render(self, mode: Optional[str] = None):
        mode = mode or self.render_mode
        if self.screen is None and mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Parking Environment")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.close()
                return

        # clear
        self.screen.fill((255, 255, 255))
        center_x, center_y = self.screen_size[0] // 2, self.screen_size[1] // 2

        def world_to_screen(wx: float, wy: float):
            dx, dy = wx - self.vehicle.state[0], wy - self.vehicle.state[1]
            theta = -self.vehicle.state[2]
            rx = dx * math.cos(theta) - dy * math.sin(theta)
            ry = dx * math.sin(theta) + dy * math.cos(theta)
            return center_x + int(rx * self.scale), center_y - int(ry * self.scale)

        def draw_polygon(points, color, width: int = 0):
            pygame.draw.polygon(self.screen, color, [world_to_screen(x, y) for x, y in points], width)

        def draw_arrow_head(tip, angle, color, size):
            """tip 点朝 angle(弧度) 画一个等腰三角形箭头"""
            x, y = tip
            p1 = (x - size * math.cos(angle - math.pi / 6),
                y - size * math.sin(angle - math.pi / 6))
            p2 = (x - size * math.cos(angle + math.pi / 6),
                y - size * math.sin(angle + math.pi / 6))
            pygame.draw.polygon(self.screen, color, [tip, p1, p2])

        # obstacles
        for obs in self.obstacles:
            if len(obs) >= 3:
                draw_polygon(obs, (100, 100, 100))
            elif len(obs) == 2:
                pygame.draw.line(self.screen, (100, 100, 100), world_to_screen(*obs[0]), world_to_screen(*obs[1]), 2)

        # target spot
        l, w = self.parking_length, self.parking_width
        draw_polygon(parking_corners(*self.target_info, l, w), (0, 200, 0), 2)

        # 目标车位希望的车头方向（绿色）
        tx, ty, tyaw = self.target_info
        spot_len = self.parking_length * 0.4 * self.scale   # 箭头长度 ≈ 40% 车位长
        start = world_to_screen(tx, ty)
        # tyaw 需减去车辆当前 yaw 才是屏幕坐标系下的相对角
        rel_yaw = tyaw - self.vehicle.state[2]
        end = (start[0] + int(math.cos(rel_yaw) * spot_len),
            start[1] - int(math.sin(rel_yaw) * spot_len))
        pygame.draw.line(self.screen, (0, 200, 0), start, end, 3)
        # draw_arrow_head(end, rel_yaw, (0, 200, 0), 10)

        # draw lidar rays
        for ang, dist in zip(self.lidar.angles, self.lidar.ranges):
            lx, ly = dist * math.cos(ang), dist * math.sin(ang)
            end = (center_x + int(lx * self.scale), center_y - int(ly * self.scale))
            intensity = min(255, int(255 * dist / self.lidar_max_range))
            ray_color = (255 - intensity, intensity, 0)
            pygame.draw.line(self.screen, ray_color, (center_x, center_y), end, 1)
            if dist < self.lidar_max_range:
                pygame.draw.circle(self.screen, (0, 150, 0), end, 3)

        # vehicle body
        if self.vehicle_poly:
            draw_polygon(list(self.vehicle_poly.exterior.coords), (200, 0, 0), 2)

        # 车辆朝向（红色）
        veh_len = 2.0 * self.scale               # 箭头长度 ≈ 2 m
        center_sx, center_sy = center_x, center_y
        end_x = center_sx + int(veh_len)         # 因为屏幕坐标已随车身对齐
        pygame.draw.line(self.screen, (255, 0, 0),
                        (center_sx, center_sy), (end_x, center_sy), 3)
        draw_arrow_head((end_x, center_sy), 0.0, (255, 0, 0), 8)

        # HUD
        font = pygame.font.SysFont(None, 24)
        tx, ty, tyaw = self.target_info
        info = [
            f"Speed: {self.vehicle.state[3]:.2f} m/s",
            f"Steer: {math.degrees(self.vehicle.state[4]):.1f}°",
            f"Step: {self.step_count}/{self.max_steps}",
            f"Scenario: {self.current_scenario}",
        ]
        dist_to_target = math.hypot(tx - self.vehicle.state[0], ty - self.vehicle.state[1])
        rel_angle = _normalize_angle(math.atan2(ty - self.vehicle.state[1], tx - self.vehicle.state[0]) - self.vehicle.state[2])
        heading_diff = _normalize_angle(tyaw - self.vehicle.state[2])
        info += [
            f"TargetDist: {dist_to_target:.2f} m",
            f"RelAngle: {math.degrees(rel_angle):.1f}°",
            f"HeadingDiff: {math.degrees(heading_diff):.1f}°",
        ]
        for i, txt in enumerate(info):
            self.screen.blit(font.render(txt, True, (0, 0, 0)), (10, 10 + i * 24))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
