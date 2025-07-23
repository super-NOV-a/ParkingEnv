from __future__ import annotations
import os
"""
ScenarioManager – difficulty‑aware version (2025‑07‑14)
=======================================================
This rewrite *replaces* the previous canvas file and keeps **all external
interfaces unchanged** while adding:

1. **Difficulty scaling** (`current_level` >= 0)
   * `gap` shrinks linearly with level → tighter slots
   * `occupy_prob` rises linearly with level → more obstacles
   * Parameters are configurable via `cfg` (see doc‑string in `__init__`).
2. **World boundary walls** – optional rectangles of thickness
   `wall_thickness` (m) around the square world; on by default.

Public usage remains:
    >>> mgr = ScenarioManager(cfg)
    >>> ego, target, obstacles = mgr.init(seed=0, current_level=3)

Return types:
    ego      – tuple(x, y, yaw)
    target   – tuple(x, y, yaw)
    obstacles – list[list[(x, y)]]
"""

import math
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from .utils import _normalize_angle
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Basic aliases
# ---------------------------------------------------------------------------
Vector = Tuple[float, float]
_EgoInfo = Tuple[float, float, float]
_TargetInfo = Tuple[float, float, float]
_ObstacleList = List[List[Vector]]

# ---------------------------------------------------------------------------
# Geometry helpers (kept local – zero extra deps)
# ---------------------------------------------------------------------------

def parking_corners(cx: float, cy: float, yaw: float, length: float, width: float) -> List[Vector]:
    """Return the 4 corners (CCW) of a rectangle centred at (cx,cy)."""
    hl, hw = 0.5 * length, 0.5 * width
    cos_t, sin_t = math.cos(yaw), math.sin(yaw)
    local = [( hl,  hw), ( hl, -hw), (-hl, -hw), (-hl,  hw)]
    return [
        (cx + u * cos_t - v * sin_t, cy + u * sin_t + v * cos_t)
        for u, v in local
    ]


def _poly_inside_world(poly: Polygon, world: float, margin: float) -> bool:
    """Axis‑aligned bounding check that entire polygon fits in world."""
    minx, miny, maxx, maxy = poly.bounds
    bound = (margin, margin, world - margin, world - margin)
    return minx >= bound[0] and miny >= bound[1] and maxx <= bound[2] and maxy <= bound[3]


def _dedup_overlap(slots: List[Tuple[Polygon, Tuple[float, float, float]]]) -> List[Tuple[Polygon, Tuple[float, float, float]]]:
    """Remove geometrically overlapping slots (simple area intersection test)."""
    accepted: List[Tuple[Polygon, Tuple[float, float, float]]] = []
    for poly, pose in slots:
        if any(poly.intersects(p) and poly.intersection(p).area > 1e-6 for p, _ in accepted):
            continue
        accepted.append((poly, pose))
    return accepted

# ---------------------------------------------------------------------------
# ScenarioManager
# ---------------------------------------------------------------------------
@dataclass
class ScenarioManager:
    cfg: Dict

    # ---------------------------------------------------------------------
    # Construction & configurable parameters
    # ---------------------------------------------------------------------
    def __post_init__(self):
        self.world_size: float = self.cfg["world_size"]  # square edge length
        # slot & vehicle sizes ------------------------------------------------
        self.parking_length: float = self.cfg["parking_length"]
        self.parking_width: float = self.cfg["parking_width"]
        self.car_length: float = self.cfg.get("car_length", self.parking_length * 0.9)
        self.car_width: float = self.cfg.get("car_width", self.parking_width * 0.9)

        # default scene parameters ------------------------------------------
        self.gap_base: float = self.cfg.get("gap_base", 4.5)  # m (easy)
        self.gap_step: float = self.cfg.get("gap_step", 0.4)  # m per level
        self.gap_min: float  = self.cfg.get("gap_min", 0.5)   # m (hard cap)

        self.occupy_base: float = self.cfg.get("occupy_prob_base", 0.05)
        self.occupy_step: float = self.cfg.get("occupy_prob_step", 0.09)
        self.occupy_max: float  = self.cfg.get("occupy_prob_max", 0.95)

        self.margin: float = self.cfg.get("margin", 5)
        self.wall_thickness: float = self.cfg.get("wall_thickness", 0.1)  # 0 → off

        self.scenario_mode = self.cfg.get("scenario_mode", "random").lower()
        assert self.scenario_mode in {"random", "file", "empty", "box", "random_box", "parking"}

        # caches for renderer access ----------------------------------------
        # self.ego_info: Optional[_EgoInfo] = None
        # self.target_info: Optional[_TargetInfo] = None
        # self.obstacles: _ObstacleList = []

        self.data_dir = Path(self.cfg.get("data_dir", "scenarios"))
        self.energy = self.cfg.get("energy", False)
        self.random_file_init = self.cfg.get("random_file_init", False)
        if self.energy:
            self.energy_data_dir = Path(self.cfg.get("energy_data_dir", "scenarios"))
        self._file_cache: List[Path] = []

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def init(
        self,
        *,
        seed: Optional[int] = None,
        scenario_idx: Optional[int] = None,
        current_level: int = 0,
        energy: bool=False,
    ) -> Tuple[_EgoInfo, _TargetInfo, _ObstacleList]:
        """Entry point – returns a triple of (ego, target, obstacles)."""
        if seed is not None:
            random.seed(seed); np.random.seed(seed)

        if self.scenario_mode == "parking":
            return self._generate_parking(current_level), None
        elif self.scenario_mode == "file":
            return self._load_from_file(scenario_idx, self.energy)
        elif self.scenario_mode == "empty":
            return self._generate_empty(current_level), None    # None 用于占位
        elif self.scenario_mode == "box":
            return self._generate_random_box(current_level), None
        elif self.scenario_mode == "random_box":
            prob = np.random.random()
            if prob <= 0.3:         # box   百分之三十
                return self._generate_random_box(current_level), None
            elif prob <= 0.8:       # random百分之五十
                return self._generate_random(current_level), None
            # elif prob <= 0.9:       # file   百分之十
            #     return self._load_from_file(scenario_idx, self.energy)
            else:                   # empty  百分之十
                return self._generate_empty(current_level), None
        else:
            return self._generate_random(current_level), None


    # ---------------------------------------------------------------------
    # Difficulty helpers
    # ---------------------------------------------------------------------
    def _effective_gap(self, level: int) -> float:
        return max(self.gap_min, self.gap_base - level * self.gap_step)

    def _effective_occupy(self, level: int) -> float:
        return min(self.occupy_max, self.occupy_base + level * self.occupy_step)
    
    def _effective_lane_width(self, level: int) -> float:
        return min(10, 10 - level * 0.4)      # 从5m到1m

    def _effective_rear_margin(self, level: int) -> float:
        return min(3, 3 - level * 0.25)      # 从2m到0.5m
    
    # ---------------------------------------------------------------------
    # Random scenario generator (main road + parking strip)
    # ---------------------------------------------------------------------
    def _generate_random(self, level: int) -> Tuple[_EgoInfo, _TargetInfo, _ObstacleList]:
        W        = self.world_size
        main_h   = W / 3.0                 # main road strip height
        park_h   = W - main_h              # parking strip
        margin   = self.margin

        # Difficulty‑dependent params --------------------------------------
        gap = self._effective_gap(level)
        occupy_prob = self._effective_occupy(level)

        # 1) Ego pose (random in main road) --------------------------------
        ego_x = random.uniform(margin, W - margin)
        ego_y = random.uniform(margin, main_h - margin)
        ego_yaw = random.uniform(-math.pi, math.pi)
        ego_info: _EgoInfo = (ego_x, ego_y, ego_yaw)

        # 2) Global random yaw for all parking slots -----------------------
        slot_yaw =  0# random.uniform(-math.pi, math.pi)
        du = self.parking_length + gap     # local u step
        dv = self.parking_width + gap      # local v step

        step_x = abs(du * math.cos(slot_yaw)) + abs(dv * math.sin(slot_yaw))
        step_y = abs(du * math.sin(slot_yaw)) + abs(dv * math.cos(slot_yaw))

        # 3) Grid anchors inside parking strip ----------------------------
        y0   = main_h + margin + 0.5 * step_y
        yMax = W       - margin - 0.5 * step_y
        if y0 > yMax:
            raise RuntimeError("Parking strip too tight – decrease gap or slot size")
        n_rows = int((yMax - y0) // step_y) + 1

        x0   = margin + 0.5 * step_x
        xMax = W      - margin - 0.5 * step_x
        n_cols = int((xMax - x0) // step_x) + 1
        if n_cols < 1:
            raise RuntimeError("World width too small – enlarge world_size")

        # 4) Build slot polygons ------------------------------------------
        slots: List[Tuple[Polygon, Tuple[float, float, float]]] = []
        for r in range(n_rows):
            row_y = y0 + r * step_y
            for c in range(n_cols):
                cx = x0 + c * step_x
                cy = row_y
                poly = Polygon(parking_corners(cx, cy, slot_yaw, self.parking_length, self.parking_width))
                if _poly_inside_world(poly, W, margin):
                    slots.append((poly, (cx, cy, slot_yaw)))

        slots = _dedup_overlap(slots)
        if not slots:
            raise RuntimeError("No slots fit – adjust parameters or world size")

        # 5) Target slot – choose among first row (closest to main road) ---
        first_row_y = y0
        first_row_slots = [s for s in slots if abs(s[1][1] - first_row_y) < 1e-6]
        target_poly, target_pose = random.choice(first_row_slots)

        # 6) Obstacles – remaining slots with probability -----------------
        obstacles: _ObstacleList = []
        rng = random.random
        for poly, pose in slots:
            if poly == target_poly:
                continue
            if rng() < occupy_prob:
                obstacles.append(list(poly.exterior.coords)[:-1])

        # 7) World boundary walls (optional) ------------------------------
        if self.wall_thickness > 0.0:
            t = self.wall_thickness
            # Bottom, Top, Left, Right
            walls = [
                [(0, 0), (W, 0), (W, t), (0, t)],
                [(0, W - t), (W, W - t), (W, W), (0, W)],
                [(0, 0), (t, 0), (t, W), (0, W)],
                [(W - t, 0), (W, 0), (W, W), (W - t, W)]
            ]
            obstacles.extend(walls)

        return ego_info, target_pose, obstacles

    # ------------------------------------------------------------------
    # File loader – unchanged from previous implementation (kept minimal)
    # ------------------------------------------------------------------
    def _scan_files(self) -> List[Path]:
        if not self._file_cache and self.data_dir.exists():
            self._file_cache = sorted(self.data_dir.glob("*.json"))
        return self._file_cache

    def _load_from_file(self, idx: Optional[int], energy:bool):
        files = self._scan_files()
        if not files:
            raise FileNotFoundError("No *.json scenario found in " + str(self.data_dir))
        if idx is None:
            idx = random.randrange(len(files))
        path = files[idx % len(files)]
        if not energy:
            return self._parse_json(path), None
            # energy_path
        file_path = files[idx]
        energy_file_path = os.path.join(self.energy_data_dir, os.path.basename(file_path))
        ## file_path
        return self._parse_json(file_path), self._load_nodes_from_json(energy_file_path) # (*self._parse_json(file_path), os.path.basename(file_path))

    def _load_nodes_from_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            nodes_data = json.load(f)

        all_nodes = []
        for node_info in nodes_data:
            node = SimpleNamespace(
                x=node_info['x'],
                y=node_info['y'],
                yaw=node_info['yaw'],
                depth=node_info['depth']
            )
            all_nodes.append(node)
        return all_nodes

    def _parse_json(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
       
        nfm_origin = data['Frames']['0'].get("m_nfmOrigin", [0, 0])
        m_pathOrigin = data['Frames']['0']['PlanningRequest'].get("m_origin", [0, 0])
       
        # 提取自车信息（带坐标转换）
        # ── 1. ego pose (rear-axle) → world坐标 ───────────────────────────
        ego_data = data['Frames']['0']['PlanningRequest']['m_startPosture']['m_pose']
        ego_x = ego_data[0] + m_pathOrigin[0] - nfm_origin[0]
        ego_y = ego_data[1] + m_pathOrigin[1] - nfm_origin[1]
        ego_yaw = _normalize_angle(ego_data[2])

        # 🚗  rear-axle → geometric centre
        ego_x, ego_y = _shift_forward(ego_x, ego_y, ego_yaw)
        if self.random_file_init:
            ego_x += np.random.uniform(-2, 2)
            ego_y += np.random.uniform(-2, 2)
            ego_yaw += np.random.uniform(-0.2, 0.2)     # 十几度
        ego_info = [ego_x, ego_y, ego_yaw]
 
         # 构建目标车位坐标变换矩阵
        cos_e = np.cos(ego_info[2])
        sin_e = np.sin(ego_info[2])
        ex, ey = ego_info[0], ego_info[1]
        trans_matrix_ego = np.array([
            [cos_e, sin_e, -ex * cos_e - ey * sin_e],
            [-sin_e, cos_e, ex * sin_e - ey * cos_e],
            [0, 0, 1]
        ])
 
        # 提取目标信息（带坐标转换）
        # ── 2. target slot pose (rear-axle) → world ──────────────────────
        tgt = data['Frames']['0']['PlanningRequest']['m_targetArea']['m_targetPosture']['m_pose']
        tgt_x = tgt[0] + m_pathOrigin[0] - nfm_origin[0]
        tgt_y = tgt[1] + m_pathOrigin[1] - nfm_origin[1]
        tgt_yaw = _normalize_angle(tgt[2])
 
        # 🅿️  rear-axle → slot geometric centre
        tgt_x, tgt_y = _shift_forward(tgt_x, tgt_y, tgt_yaw)
        target_info = [tgt_x, tgt_y, tgt_yaw]
       
        # 构建目标车位坐标变换矩阵
        cos_t = np.cos(target_info[2])
        sin_t = np.sin(target_info[2])
        tx, ty = target_info[0], target_info[1]
        trans_matrix = np.array([
            [cos_t, sin_t, -tx * cos_t - ty * sin_t],
            [-sin_t, cos_t, tx * sin_t - ty * cos_t],
            [0, 0, 1]
        ])
        W = self.world_size
        # 提取并过滤障碍物
        # ───────────────────────── 3) 外圈世界墙体 ───────────────────────
        obstacles: _ObstacleList = []
        if self.wall_thickness > 0.0:
            t = self.wall_thickness
            half_W = self.world_size / 2.0

            # 让 target 几何中心处于世界正方形中心
            tx, ty = target_info[0], target_info[1]          # 目标车位中心
            xmin, xmax = tx - half_W, tx + half_W
            ymin, ymax = ty - half_W, ty + half_W
 
            Wll = [
                # 下边墙
                [(xmin, ymin), (xmax, ymin), (xmax, ymin + t), (xmin, ymin + t)],
                # 上边墙
                [(xmin, ymax - t), (xmax, ymax - t), (xmax, ymax), (xmin, ymax)],
                # 左边墙
                [(xmin, ymin), (xmin + t, ymin), (xmin + t, ymax), (xmin, ymax)],
                # 右边墙
                [(xmax - t, ymin), (xmax, ymin), (xmax, ymax), (xmax - t, ymax)],
            ]
            obstacles.extend(Wll)
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
 
            # 转换到ego车辆坐标系
            transformed_ego = trans_matrix_ego @ homogenous
            ego_coords = transformed_ego[:2, :].T
           
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
 
            # 检查是否在其实车辆边界内
            for x, y in ego_coords:
                if (vehicle_x_min <= x <= vehicle_x_max and
                    vehicle_y_min <= y <= vehicle_y_max):
                    in_target = True
                    break
                   
            # 只保留目标车位外的障碍物
            if not in_target:
                obstacles.append(polygon)
       
        return ego_info, target_info, obstacles

    # ------------------------------------------------------------------
    # Empty generator – unchanged (kept for compatibility)
    # ------------------------------------------------------------------
    def _generate_empty(self, level: int = 0):
        """
        生成“空白场地”的场景。

        - 整个 world (edge=W) 已由四周固定墙体包围
        """
        W = self.world_size
        margin = self.margin
        tx = random.uniform(margin, W - margin)
        ty = random.uniform(margin, W - margin)
        tyaw = random.uniform(0, 2 * math.pi)
        target_info = (tx, ty, tyaw)

        # ego distance grows with level
        min_d = level * 1.5
        max_d = 10.0 + level * 3.0
        ang = random.uniform(0, 2 * math.pi)
        d = random.uniform(min_d, min_d + max_d)
        ex = min(max(tx + d * math.cos(ang), margin), W - margin)
        ey = min(max(ty + d * math.sin(ang), margin), W - margin)
        eyaw = random.uniform(0, 2 * math.pi)
        ego_info = (ex, ey, eyaw)

        obstacles: _ObstacleList = []
        if self.wall_thickness > 0.0:
            t = self.wall_thickness
            Wll = [
                [(0, 0), (W, 0), (W, t), (0, t)],
                [(0, W - t), (W, W - t), (W, W), (0, W)],
                [(0, 0), (t, 0), (t, W), (0, W)],
                [(W - t, 0), (W, 0), (W, W), (W - t, W)]
            ]
            obstacles.extend(Wll)
        return ego_info, target_info, obstacles

    def _generate_random_box(self, level: int = 0):
        """
        Random-box scene:
        1. 整个 world 用四周固定墙体包围；
        2. 目标车位中心随机，姿态随机；
        3. 在车位 local 坐标系 (u,v) 中，沿 ±u/±v 方向最多选 3 条边生成
        与车位平行的墙体（矩形），墙-车位最近距离 ≥ 0.5 m。
        """

        # ───────────────────────── 0) 参数 & 快捷量 ──────────────────────
        W      = self.world_size
        t      = max(self.wall_thickness, 0.05)          # 最薄也给 0.05
        GAP    = 3 - 0.2*level                           # 车位边 ↔ 墙内侧 距离。最大为3，最小为1
        margin = self.margin

        pl, pw = self.parking_length, self.parking_width
        hl, hw = pl / 2, pw / 2                          # half length / width

        # ───────────────────────── 1) 随机目标车位 pose ──────────────────
        tx  = random.uniform(margin + hl, W - margin - hl)
        ty  = random.uniform(margin + hw, W - margin - hw)
        tyaw = random.uniform(0, 2 * math.pi)
        cos_y, sin_y = math.cos(tyaw), math.sin(tyaw)
        target_info = (tx, ty, tyaw)

        # ───────────────────────── 2) 随机 ego pose ─────────────────────
        half_W = self.world_size / 2.0
        t      = max(self.wall_thickness, 0.05)      # 世界外墙厚度
        xmin, xmax = tx - half_W + t + margin, tx + half_W - t - margin
        ymin, ymax = ty - half_W + t + margin, ty + half_W - t - margin

        d_min = 8   # 远离5米即可生成时避开障碍
        d_max = half_W - t - margin

        for _try in range(20):                               # 最多尝试 200 次
            # 随机角度 + 距离采样
            d   = random.uniform(d_min, d_max)
            ang = random.uniform(0, 2 * math.pi)
            ex  = tx + d * math.cos(ang)
            ey  = ty + d * math.sin(ang)

            # 若落到 world 墙内侧外再试
            if not (xmin <= ex <= xmax and ymin <= ey <= ymax):
                continue

            break
        else:
            raise RuntimeError("无法为 ego 找到合法初始位，调整 world_size 或降低 level")

        eyaw = random.uniform(0, 2 * math.pi)
        ego_info = (ex, ey, eyaw)

        # ───────────────────────── 3) 外圈世界墙体 ───────────────────────
        obstacles: _ObstacleList = []
        if self.wall_thickness > 0.0:
            t = self.wall_thickness
            half_W = self.world_size / 2.0

            # 让 target 几何中心处于世界正方形中心
            tx, ty = target_info[0], target_info[1]          # 目标车位中心
            xmin, xmax = tx - half_W, tx + half_W
            ymin, ymax = ty - half_W, ty + half_W

            Wll = [
                # 下边墙
                [(xmin, ymin), (xmax, ymin), (xmax, ymin + t), (xmin, ymin + t)],
                # 上边墙
                [(xmin, ymax - t), (xmax, ymax - t), (xmax, ymax), (xmin, ymax)],
                # 左边墙
                [(xmin, ymin), (xmin + t, ymin), (xmin + t, ymax), (xmin, ymax)],
                # 右边墙
                [(xmax - t, ymin), (xmax, ymin), (xmax, ymax), (xmax - t, ymax)],
            ]
            obstacles.extend(Wll)

        # ───────────────────────── 4) 车位局部包围墙 ─────────────────────
        # 长边 / 短边方向上墙体应延伸多少： + GAP 余量
        span_u = hl  + GAP - 2
        span_v = hw  + GAP - 1.3   #  前后 GAP

        # 四条候选墙：以车位中心为原点的 (u,v) 坐标
        local_walls = {
            "front": [                               # +u 方向（墙长沿 v）
                [hl + GAP,         -span_v],
                [hl + GAP + t,     -span_v],
                [hl + GAP + t,      span_v],
                [hl + GAP,          span_v],
            ],
            "rear": [                                # -u 方向
                [-hl - GAP - t,   -span_v],
                [-hl - GAP,       -span_v],
                [-hl - GAP,        span_v],
                [-hl - GAP - t,    span_v],
            ],
            "left": [                                # +v 方向（墙长沿 u）
                [-span_u,  hw + GAP],
                [ span_u,  hw + GAP],
                [ span_u,  hw + GAP + t],
                [-span_u,  hw + GAP + t],
            ],
            "right": [                               # -v 方向
                [-span_u, -hw - GAP - t],
                [ span_u, -hw - GAP - t],
                [ span_u, -hw - GAP],
                [-span_u, -hw - GAP],
            ],
        }

        # 将 local (u,v) → world (x,y)
        def local_to_world(u, v):
            return (
                tx + u * cos_y - v * sin_y,
                ty + u * sin_y + v * cos_y,
            )

        # ────────────────── 根据 level 决定要几条墙 (k = 0‥3) ──────────────────
        alpha = max(0, min(level, 10)) / 10.0      # 归一化到 0-1
        p3   = alpha                               # 线性：level=10 → p3=1
        rest = 1.0 - p3
        p2   = alpha * rest                        # 给 2 墙一个钟形概率
        p0 = p1 = (rest - p2) / 2.0                # 剩下均分给 0/1 墙

        r = random.random()
        if r < p0:
            k = 0
        elif r < p0 + p1:
            k = 1
        elif r < p0 + p1 + p2:
            k = 2
        else:
            k = 3
        available = ["front", "rear", "left", "right"]
        selected = []

        while available and len(selected) < k:
            edge = random.choice(available)
            selected.append(edge)
            # 互斥规则：选了 front 就移除 rear，反之亦然
            if edge == "front" and "rear" in available:
                available.remove("rear")
            if edge == "rear" and "front" in available:
                available.remove("front")
            available.remove(edge)  # 当前边已选，用掉

        # 生成墙体
        for edge in selected:
            wall_pts = [local_to_world(u, v) for u, v in local_walls[edge]]
            obstacles.append(wall_pts)

        return ego_info, target_info, obstacles

    def _generate_parking(self, level: int = 0):
        """生成【两排垂直车位 + 中间行车道】泊车场景，并支持：
        1) 车辆朝向在原本 ±y 方向基础上随机再偏转 ≤10°；
        2) 占位障碍不仅限于矩形，可随机生成三角形 / 六边形等形状，
            其最大外接尺寸不超过对应车位 0.9 倍。
        """

        # ===================== 基本参数 ===============================
        W       = self.world_size                   # 横向长度 (y 轴)
        t       = max(self.wall_thickness, 0.05)    # 墙厚
        margin  = self.margin                       # 边距

        pl      = self.parking_length               # 车位纵深 (x)
        pw      = self.parking_width                # 车位横宽 (y)

        gap_y   = self._effective_gap(level)        # 相邻车位间隔
        occupy  = self._effective_occupy(level)     # 障碍概率

        lane_w  = self._effective_lane_width(level)                 # 行车道宽
        rear_m  = self._effective_rear_margin(level)                # 上下留白
        L       = 2 * pl + lane_w + rear_m                          # 纵向总宽 (x)

        center_x   = L / 2.0
        left_row_x = center_x - (lane_w / 2.0 + pl / 2.0)
        right_row_x = center_x + (lane_w / 2.0 + pl / 2.0)

        # ===================== 沿 y 轴排布车位 ========================
        y0     = margin + pw / 2.0
        y_max  = W - margin - pw / 2.0
        step_y = pw + gap_y
        n_rows = int((y_max - y0) // step_y) + 1

        slots = []          # (row_tag, (cx, cy))
        for i in range(n_rows):
            cy = y0 + i * step_y
            slots.append(("upper", (left_row_x,  cy)))
            slots.append(("lower", (right_row_x, cy)))

        # ===================== 目标车位 ===============================
        target_row, (tx, ty) = random.choice(slots)
        head_out  = random.random() < 0.7                          # 70% 车头朝行车道
        tyaw  = 0 if target_row == "upper" else math.pi
        if not head_out:                       # 车头朝相反方向
            tyaw += math.pi               # 反向
        target_info = (tx, ty, tyaw)

        # ===================== 障碍物（多形状） =======================
        def gen_shape(cx, cy, yaw, depth, width):
            """在局部坐标生成形状并旋转、平移到世界坐标"""
            shape_type = random.choice(["rect", "triangle", "hex"])
            dx, dy = width * 0.45, depth * 0.45  # 保证外接框 ≤ 0.9 × 车位
            if shape_type == "rect":
                # 仍用矩形
                local = Polygon([(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)])
            elif shape_type == "triangle":
                # 等腰三角形，底朝行车道
                local = Polygon([(-dx, -dy), (dx, -dy), (0.0, dy)])
            else:  # hexagon
                pts = [(dx * math.cos(a), dy * math.sin(a))
                    for a in [math.radians(k) for k in range(0, 360, 60)]]
                local = Polygon(pts)

            poly = affinity.rotate(local, math.degrees(yaw), origin=(0, 0))
            poly = affinity.translate(poly, xoff=cx, yoff=cy)
            return list(poly.exterior.coords)[:-1]

        obstacles = []
        for row, (sx, sy) in slots:
            if (sx, sy) == (tx, ty):
                continue  # 目标位留空
            if random.random() < occupy:
                syaw_base = -math.pi/2 if row == "upper" else math.pi/2
                syaw      = syaw_base + math.radians(random.uniform(-10, 10))
                obstacles.append(gen_shape(sx, sy, syaw, pl, pw))

        # ===================== 世界边界墙 =============================
        if t > 0.0:
            walls = [
                [(0,     0),  (L,     0),  (L,     t),  (0,     t)],   # 下
                [(0,   W-t),  (L,   W-t),  (L,     W),  (0,     W)],   # 上
                [(0,     0),  (t,     0),  (t,     W),  (0,     W)],   # 左
                [(L-t,   0),  (L,     0),  (L,     W),  (L-t,   W)],   # 右
            ]
            obstacles.extend(walls)

        # ===================== 车辆初始位 =============================
        lane_x_min = left_row_x + pl / 2.0 + 0.2
        lane_x_max = right_row_x - pl / 2.0 - 0.2

        spawn_upper = random.random() < 0.5
        spawn_x = random.uniform(lane_x_min, lane_x_max)
        if spawn_upper:
            spawn_y   = min(W - margin - pw, y_max + gap_y / 2.0)
            spawn_yaw = -math.pi/2
        else:
            spawn_y   = max(margin + pw, y0 - gap_y / 2.0)
            spawn_yaw =  math.pi/2
        spawn_yaw += math.radians(random.uniform(-level, level))         # ≤±10°

        ego_info = (spawn_x, spawn_y, spawn_yaw)

        return ego_info, target_info, obstacles


def _shift_forward(x: float, y: float, yaw: float, dx: float=1.4):
    """沿 yaw 正方向把 (x, y) 平移 dx。"""
    return x + dx * np.cos(yaw), y + dx * np.sin(yaw)
