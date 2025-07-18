from __future__ import annotations
import math, random, numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque

import gymnasium as gym

from .utils import _normalize_angle
from vehicles.vehicle_continuous import VehicleContinuous
from vehicles.vehicle_disc_accel import VehicleDiscAccel
from vehicles.vehicle_arc import VehicleArc
from lidar import Lidar2D
from .scenario_manager import ScenarioManager
from .render import PygameRenderer
from .energy_core import get_energy

Vector = Tuple[float, float]

_VEHICLE_REGISTRY = {
    "continuous": (
        VehicleContinuous,
        lambda cls: gym.spaces.Box(low=np.array([-1., -1.]),
                                   high=np.array([1., 1.]),
                                   dtype=np.float32)),
    "disc_accel": (
        VehicleDiscAccel,
        lambda cls: gym.spaces.Discrete(cls.N_ACTIONS)),
    "arc": (
        VehicleArc,
        lambda cls: gym.spaces.MultiDiscrete([cls.N_STEER, cls.N_ARC])),
}
_DEFAULT_TYPE = "continuous"


class ParkingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "ParkingEnv"}

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.energy       = cfg.get("energy", False)      # 是否使用能量场
        self.energy_nodes = None
        # ======================== 基本仿真参数 ========================
        self.dt         = cfg.get("timestep",   0.1)     # 时间步长
        self.max_steps  = cfg.get("max_steps",  500)     # 最大步数
        self.render_mode = cfg.get("render_mode", "human")

        # ======================== 车辆参数配置 ========================
        self.wheelbase   = 3.0
        self.front_hang  = 1.0
        self.rear_hang   = 1.0
        self.car_width   = 2.0
        self.car_length  = self.wheelbase + self.front_hang + self.rear_hang
        self.parking_length, self.parking_width = 5., 2.
        self.max_steer   = math.radians(30.0)
        self.max_speed   = cfg.get("max_speed", 3.0)

        # ======================== 车辆模型选择 ========================
        self.vehicle_type = cfg.get("vehicle_type", _DEFAULT_TYPE)
        VehicleCls, space_fn = _VEHICLE_REGISTRY[self.vehicle_type]
        self.vehicle = VehicleCls(
            wheelbase=self.wheelbase,
            width=self.car_width,
            front_hang=self.front_hang,
            rear_hang=self.rear_hang,
            max_steer=self.max_steer,
            max_speed=self.max_speed,
            dt=self.dt
        )
        self.action_space = space_fn(VehicleCls)

        # 离散控制空间尺寸（供动作编码使用）
        if self.vehicle_type == "arc":
            self.N_STEER = VehicleArc.N_STEER
            self.N_ARC   = VehicleArc.N_ARC
        else:
            self.N_STEER = self.N_ARC = 1

        # ======================== 传感器（激光雷达）配置 ========================
        lidar_cfg = dict(
            range_min=0.5,
            max_range=cfg.get("lidar_max_range", 30.0),
            angle_range=360,
            num_beams=72,
            noise=False,
            std=0.05,
            angle_std=0.5,
            position_offset=(0.0, 0.0)
        )
        self.lidar = Lidar2D(lidar_cfg)

        # ======================== 状态空间与观测空间 ========================
        state_low  = [-1., -1., 0., -1., -1., -1., -1.,  -1., -1.]
        state_high = [ 1.,  1., 1.,  1.,  1.,  1.,  1.,   1.,  1.]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.] * lidar_cfg["num_beams"] + state_low, np.float32),
            high=np.array([lidar_cfg["max_range"]] * lidar_cfg["num_beams"] + state_high, np.float32),
            dtype=np.float32
        )

        # ======================== 场景生成配置 ========================
        cfg_defaults = dict(
            world_size=30.0,
            scenario_mode="random",
            min_obstacles=3,
            max_obstacles=8,
            min_obstacle_size=1.0,
            max_obstacle_size=10.0,
            row_count_range=(1, 4),
            ego_target_max_dist=30,
            wall_thickness=0.15
        )
        self.cfg = {**cfg_defaults, **cfg,
                    "parking_length": self.parking_length,
                    "parking_width":  self.parking_width,
                    "car_length":     self.car_length,
                    "car_width":      self.car_width}
        self.scenario = ScenarioManager(self.cfg)
        self.is_file = (self.cfg.get("scenario_mode", "random").lower() == "file")
        self.renderer = PygameRenderer() if self.render_mode == "human" else None

        # ======================== 难度等级管理 ========================
        self.dist_levels  = np.linspace(2.0, 0.25, 11)
        self.angle_levels = np.radians(np.linspace(36, 3, 11))
        self.level = int(np.clip(cfg.get("difficulty_level", 0), 0, 10))
        self.success_dist  = float(self.dist_levels[self.level])
        self.success_angle = float(self.angle_levels[self.level])

        # ======================== 碰撞检测缓存 ========================
        self.vehicle_poly = self.vehicle.get_shapely_polygon()
        self._radar: Optional[np.ndarray] = None
        self._collision_thresholds: np.ndarray = np.empty(self.lidar.num_beams, dtype=np.float32)
        self._init_vehicle_collision_thresholds()

        # ======================== 动作编码查找表 ========================
        if self.N_STEER > 1:
            self._steer_norm_lut = np.linspace(-1.0, 1.0, self.N_STEER, dtype=np.float32)
        if self.N_ARC > 1:
            self._arc_norm_lut = np.linspace(-1.0, 1.0, self.N_ARC, dtype=np.float32)

        # ======================== 观测缓存 ========================
        self._state_buf = np.empty(9, dtype=np.float32)
        self._obs_buf   = np.empty(self.lidar.num_beams + 9, dtype=np.float32)
        self._prev_action = np.array([self.N_STEER // 2, self.N_ARC // 2], dtype=np.int32)

        # ======================== 学习/评估统计 ========================
        self._success_history = deque(maxlen=500)
        self.episode_success  = False
        self.episode_reward   = 0.0

        # ======================== 环境缓冲区清空 ========================
        self._reset_internal_buffers()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None, scenario_idx: Optional[int] = None):
        # ------ 难度自适应 ------
        if hasattr(self, "episode_success"):
            self._success_history.append(self.episode_success)
            self._adjust_difficulty()

        self.episode_success = False
        self.episode_reward  = 0.0
        self._radar          = None

        if seed is not None:
            self.seed(seed)
        if self.energy:
            (self.ego_info, self.target_info, self.obstacles) ,jr_pkgs = \
                self.scenario.init(seed=seed, scenario_idx=scenario_idx, current_level=self.level, energy=self.energy)
            self.energy_nodes = jr_pkgs
        else:
            (self.ego_info, self.target_info, self.obstacles), _ = \
                self.scenario.init(seed=seed, scenario_idx=scenario_idx, current_level=self.level)
        self.current_scenario = scenario_idx
        self.vehicle.reset_state(*self.ego_info)
        self.step_count   = 0
        self.prev_dist    = float("inf")
        self.vehicle_poly = self.vehicle.get_shapely_polygon()
        self._update_obstacle_geometries()
        self._prev_action = np.array([self.N_STEER // 2, self.N_ARC // 2], dtype=np.int32)
        return self._get_observation(), {}

    def step(self, action):
        # 记录当前动作索引 (array)
        curr_act = np.asarray(action, dtype=np.int32).copy()

        self.vehicle.state, _ = self.vehicle.step(action)
        self.vehicle_poly = self.vehicle.get_shapely_polygon()
        self._radar = None

        obs   = self._get_observation()
        term, trunc, coll = self._check_termination()
        reward = self._calc_reward(term, trunc, coll, self.energy)

        if term and not coll:
            self.episode_success = True
        self.episode_reward += reward
        self.step_count      += 1

        if self.renderer and self.step_count % 5 == 0:
            self.render()

        # ⚠️ 更新 prev_action 供下一步观测使用
        self._prev_action = curr_act
        return obs, reward, term, trunc, {"collision": coll}

    # ------------------------------------------------------------------
    # Difficulty adjustment
    # ------------------------------------------------------------------
    def _adjust_difficulty(self):
        if len(self._success_history) < self._success_history.maxlen:
            return
        success_rate = sum(self._success_history) / len(self._success_history)
        if success_rate > 0.7 and self.level < 10:
            self.level += 1
            self._success_history.clear()
            print(f"[Difficulty ↑] level={self.level}  success_rate={success_rate:.2%}")
        elif success_rate < 0.6 and self.level > 0:
            self.level -= 1
            self._success_history.clear()
            print(f"[Difficulty ↓] level={self.level}  success_rate={success_rate:.2%}")
        self.success_dist  = float(self.dist_levels[self.level])
        self.success_angle = float(self.angle_levels[self.level])
    
    # ------------------------------------------------------------------
    # Cached lidar scan
    # ------------------------------------------------------------------
    def _scan_radar(self, x: float, y: float, yaw: float) -> np.ndarray:
        """懒加载激光数据，避免同一帧重复扫描"""
        if self._radar is None:
            self._radar = self.lidar.scan(x, y, yaw)
        return self._radar

    # ------------------------------------------------------------------
    # Observation / Reward / Termination
    # ------------------------------------------------------------------
    def _get_observation(self):
        x, y, yaw = self.vehicle.get_pose_center()
        radar = self._scan_radar(x, y, yaw)
        self._obs_buf[:self.lidar.num_beams] = radar / self.lidar.max_range  # (radar - self._collision_thresholds)

        dx, dy = self.target_info[0] - x, self.target_info[1] - y
        dist   = math.hypot(dx, dy)
        rel_ang = _normalize_angle(math.atan2(dy, dx) - yaw)
        heading = _normalize_angle(self.target_info[2] - yaw)

        heading_sin, heading_cos = math.sin(heading), math.cos(heading)
        bearing_sin, bearing_cos = math.sin(rel_ang), math.cos(rel_ang)
        prev_s, prev_a = self._encode_prev_action()

        self._state_buf[:] = [
            self.vehicle.state[3] / self.max_speed,
            self.vehicle.state[4] / self.max_steer,
            min(dist, 10.0) / 10.0,
            bearing_sin, bearing_cos,
            heading_sin, heading_cos,
            prev_s, prev_a,
        ]
        self._obs_buf[self.lidar.num_beams:] = self._state_buf
        return self._obs_buf
        
    def _calc_reward(self, terminated: bool, trunc: bool, collided: bool, energy: bool) -> float:
        if energy and self.is_file:
            energy = get_energy(self.energy_nodes, self.vehicle.state[:3])  
            energy_reward = (energy - 500) / (1000 * 300)
            if terminated and not collided:
                n_switch = self.vehicle.switch_count
                return max(1.0 - 0.05 * n_switch, 0.3) + energy_reward
            return energy_reward
        if terminated and not collided:
            n_switch = self.vehicle.switch_count
            return max(0.05 * (20 - n_switch), 0.3)*(1+0.1*self.level)   # 对于更高难度鼓励智能体
        if trunc or collided:
            return -0.05
        return 0.0

    def _check_termination(self):
        x, y, yaw = self.vehicle.get_pose_center()
        tx, ty, tyaw = self.target_info

        # ---- 成功 —— 距离 + 朝向 ----
        if math.hypot(tx - x, ty - y) < self.success_dist and \
           abs(_normalize_angle(yaw - tyaw)) < self.success_angle:
            return True, False, False

        # ---- 碰撞 ----
        if np.any(self._scan_radar(x, y, yaw) < self._collision_thresholds):
            return True, False, True   # 碰撞不结束的话需要强制车辆移动到碰撞前才对，不然会强制一直扣分

        # ---- 其他失败 ----
        if math.hypot(tx - x, ty - y) > self.cfg["world_size"]:
            return False, True, False  # Out of bounds
        if self.step_count >= self.max_steps:
            return False, True, False  # Timeout
        return False, False, False

    def _encode_prev_action(self) -> Tuple[float, float]:
        s = self._steer_norm_lut[self._prev_action[0]] if self.N_STEER > 1 else 0.0
        a = self._arc_norm_lut[self._prev_action[1]] if self.N_ARC > 1 else 0.0
        return s, a

    # ------------------------------------------------------------------
    # Collision threshold (pre‑computed once)
    # ------------------------------------------------------------------
    def _init_vehicle_collision_thresholds(self):
        """
        计算每条激光束与车辆外轮廓的最近交点距离，结果保存在
        self._collision_thresholds (shape = [num_beams], 单位: 米)。

        用途：在 _check_termination() 中快速判定是否车体-障碍碰撞
        ——只需比较当前雷达距离 < 该阈值即可。
        """
        # ------------------------------------------------------------
        # 1. 构造车辆轮廓线段集合 (N,4):  x1,y1,x2,y2
        # ------------------------------------------------------------
        max_range=self.cfg.get("lidar_max_range", 30.0)
        cx, cy, _ = self.vehicle.get_pose_center()        # 雷达原点
        v_poly = self.vehicle_poly                        # 已由 reset() 生成
        coords = list(v_poly.exterior.coords)
        segs = np.asarray(
            [[x1, y1, x2, y2] for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:])],
            dtype=np.float32,
        )
        if segs.shape[0] == 0:  # 理论不会发生，安全检查
            self._collision_thresholds = np.full(self.lidar.num_beams, max_range, np.float32)
            return

        # ------------------------------------------------------------
        # 2. 计算所有激光束方向的单位向量
        # ------------------------------------------------------------
        # self.lidar.angles ∈ [-π, π]，与 scan() 内部保持一致
        dir_cos = np.cos(self.lidar.angles, dtype=np.float32)
        dir_sin = np.sin(self.lidar.angles, dtype=np.float32)

        # ------------------------------------------------------------
        # 3. 射线-线段批量求交，优先走 Numba 快路径
        # ------------------------------------------------------------
        from lidar import USING_NUMBA
        if USING_NUMBA:
            # 动态导入函数以避免在无 Numba 环境下报错
            from lidar import _ray_trace_batch  # type: ignore
            dists = _ray_trace_batch(
                float(cx), float(cy),
                dir_cos, dir_sin,
                segs,
                float(max_range),
            )
        else:
            # --- 纯 Python 回退 ------------------------------------------------
            r_max = float(max_range)
            dists = np.full(self.lidar.num_beams, r_max, dtype=np.float32)
            for i in range(self.lidar.num_beams):
                cx_dir, sx_dir = dir_cos[i], dir_sin[i]
                best = r_max
                for x1, y1, x2, y2 in segs:
                    denom = cx_dir * (y2 - y1) - sx_dir * (x2 - x1)  # r × s
                    if abs(denom) < 1e-12:
                        continue  # 平行
                    t = ((x1 - cx) * sx_dir - (y1 - cy) * cx_dir) / denom
                    if t < 0.0 or t > 1.0:
                        continue  # 不在线段内
                    u = ((x1 - cx) * (y2 - y1) - (y1 - cy) * (x2 - x1)) / denom
                    if u <= 0.0:
                        continue  # 射线反向
                    if u < best:
                        best = u
                dists[i] = best

        # ------------------------------------------------------------
        # 4. 可以写入阈值并加一点裕度 (105%)，防止浮点误差漏检
        # ------------------------------------------------------------
        self._collision_thresholds = (dists).astype(np.float32)

    # ------------------------------------------------------------------
    # Obstacle helpers
    # ------------------------------------------------------------------
    def _reset_internal_buffers(self):
        self.ego_info = self.target_info = None
        self.obstacles: List[List[Tuple[float, float]]] = []
        self.obstacle_geoms = []
        self.current_scenario = ""
        self.step_count       = 0

    def _update_obstacle_geometries(self):
        from shapely.geometry import Polygon, LineString, Point
        self.obstacle_geoms = []
        for obs in self.obstacles:
            if len(obs) >= 3:
                self.obstacle_geoms.append(Polygon(obs))
            elif len(obs) == 2:
                self.obstacle_geoms.append(LineString(obs))
            elif len(obs) == 1:
                self.obstacle_geoms.append(Point(obs[0]))
        self.lidar.update_obstacles(self.obstacle_geoms)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def close(self):
        if self.renderer:
            self.renderer.render = lambda *a, **k: None

    def render(self):
        if not self.renderer:  # rgb_array 模式 return raw array
            return
        x, y, yaw = self.vehicle.get_pose_center()
        self.renderer.render(
            vehicle_poly=self.vehicle_poly,
            vehicle_state=(x, y, yaw, self.vehicle.state[3], self.vehicle.state[4]),
            target_info=self.target_info,
            obstacles=self.obstacles,
            lidar=self.lidar,
            lidar_max_range=self.lidar.max_range,
            step=self.step_count,
            max_steps=self.max_steps,
            scenario_name=self.current_scenario,
            parking_length=self.parking_length,
            parking_width=self.parking_width,
        )
