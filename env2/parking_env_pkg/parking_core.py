# parking_env_pkg/parking_core.py
from __future__ import annotations
import math, random, numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque

import gymnasium as gym
from shapely.geometry import Polygon

from .utils import _normalize_angle, parking_corners

from vehicles.vehicle_continuous import VehicleContinuous
from vehicles.vehicle_disc_accel import VehicleDiscAccel
from vehicles.vehicle_arc import VehicleArc

from lidar import Lidar2D
from .scenario_manager import ScenarioManager
from .render import PygameRenderer

Vector = Tuple[float, float]

_VEHICLE_REGISTRY = {
    "continuous": (VehicleContinuous,
                   lambda cls: gym.spaces.Box(low=np.array([-1., -1.]),
                                              high=np.array([1., 1.]),
                                              dtype=np.float32)),
    "disc_accel": (VehicleDiscAccel,
                   lambda cls: gym.spaces.Discrete(cls.N_ACTIONS)),
    "arc": (VehicleArc,
            lambda cls: gym.spaces.MultiDiscrete([cls.N_STEER, cls.N_ARC])),
}
_DEFAULT_TYPE = "continuous"

class ParkingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "ParkingEnv"}

    # ------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------
    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        # ---------- basic params ---------------------------------
        self.dt = cfg.get("timestep", 0.1)
        self.max_steps = cfg.get("max_steps", 500)
        self.render_mode = cfg.get("render_mode", "human")

        # ---------- geometry -------------------------------------
        self.wheelbase = 3.0; self.front_hang = 1.0
        self.rear_hang = 1.0; self.car_width = 2.0
        self.car_length = self.wheelbase + self.front_hang + self.rear_hang
        self.parking_length, self.parking_width = 5.5, 2.3

        # ---------- vehicle --------------------------------------
        self.max_steer = math.radians(30.0)
        self.max_speed = cfg.get("max_speed", 3.0)
        self.vehicle_type = cfg.get("vehicle_type", _DEFAULT_TYPE)
        VehicleCls, space_fn = _VEHICLE_REGISTRY[self.vehicle_type]
        self.vehicle = VehicleCls(wheelbase=self.wheelbase, width=self.car_width,
                                  front_hang=self.front_hang, rear_hang=self.rear_hang,
                                  max_steer=self.max_steer, max_speed=self.max_speed,
                                  dt=self.dt)
        self.action_space = space_fn(VehicleCls)

        # ---------- lidar ----------------------------------------
        lidar_cfg = dict(range_min=0.5,
                         max_range=cfg.get("lidar_max_range", 30.0),
                         angle_range=360, num_beams=72,
                         noise=False, std=0.05, angle_std=0.5,
                         position_offset=(0.0, 0.0))
        self.lidar_max_range = lidar_cfg["max_range"]
        self.lidar = Lidar2D(lidar_cfg)
        state_low  = [-1., -1., 0., -1., -1.]
        state_high = [ 1.,  1., 1.,  1.,  1.]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.] * lidar_cfg["num_beams"] + state_low, np.float32),
            high=np.array([self.lidar_max_range] * lidar_cfg["num_beams"] + state_high, np.float32),
            dtype=np.float32)
        self.lidar_data = None

        # ---------- scenario & render ----------------------------
        cfg_defaults = dict(world_size=30.0,
                            scenario_mode="random",
                            min_obstacles=3, max_obstacles=8,
                            min_obstacle_size=1.0, max_obstacle_size=10.0,
                            row_count_range=(1, 4), occupy_prob=0.2,
                            ego_target_max_dist=30, gap=4.0,
                            wall_thickness=0.15)
        self.cfg = {**cfg_defaults, **cfg,
                    "parking_length": self.parking_length,
                    "parking_width": self.parking_width,
                    "car_length": self.car_length,
                    "car_width": self.car_width}
        self.scenario = ScenarioManager(self.cfg)
        self.renderer = PygameRenderer() if self.render_mode == "human" else None

        # ---------- difficulty table -----------------------------
        # 0 ←───→ 10
        self.dist_levels  = np.linspace(2.0, 0.25, 11)            # m
        self.angle_levels = np.radians(np.linspace(36, 3, 11))    # rad
        self.level        = int(np.clip(cfg.get("difficulty_level", 0), 0, 10))
        self.success_dist  = float(self.dist_levels[self.level])
        self.success_angle = float(self.angle_levels[self.level])

        # ---------- runtime state --------------------------------
        self._success_history = deque(maxlen=1000)   # 成功/失败（bool）
        self.episode_success  = False
        self.episode_reward   = 0.0                  # 仍可保留奖励统计
        self._reset_internal_buffers()

    # ------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None,
              scenario_idx: Optional[int] = None):
        # --- 记录上一局成功与否，并据此调整难度 ---
        if hasattr(self, "episode_success"):
            self._success_history.append(self.episode_success)
            self._adjust_difficulty()

        # --- 开始新一局 ---
        self.episode_reward = 0.0
        self.episode_success = False

        if seed is not None:
            self.seed(seed)
        (self.ego_info,
         self.target_info,
         self.obstacles) = self.scenario.init(seed=seed, scenario_idx=scenario_idx)
        self.current_scenario = scenario_idx
        self.vehicle.reset_state(*self.ego_info)
        self.vehicle_poly = self.vehicle.get_shapely_polygon()
        self.prev_dist  = float("inf")
        self.step_count = 0
        self._update_obstacle_geometries()
        return self._get_observation(), {}

    def step(self, action):
        self.vehicle.state, _ = self.vehicle.step(action)
        self.vehicle_poly = self.vehicle.get_shapely_polygon()

        obs   = self._get_observation()
        term, trunc, coll = self._check_termination()
        reward = self._calc_reward(term, coll)

        # 成功标志：本局终止且未碰撞
        if term and not coll:
            self.episode_success = True

        self.episode_reward += reward
        self.step_count += 1
        if self.renderer and self.step_count % 5 == 0:
            self.render()
        return obs, reward, term, trunc, {"collision": coll}

    # ------------------------------------------------------------
    # 难度调节
    # ------------------------------------------------------------
    def _adjust_difficulty(self):
        """根据最近 1000 局成功率调整 level → success_dist / angle"""
        if len(self._success_history) < self._success_history.maxlen:
            return  # 样本不足
        success_rate = sum(self._success_history) / len(self._success_history)

        if success_rate > 0.5 and self.level < 10:
            self.level += 1
        elif success_rate < 0.3 and self.level > 0:
            self.level -= 1

        # 更新阈值
        self.success_dist  = float(self.dist_levels[self.level])
        self.success_angle = float(self.angle_levels[self.level])

        print(f"[Difficulty] level={self.level:2d} | "
              f"dist ≤ {self.success_dist:.3f} m | "
              f"angle ≤ {math.degrees(self.success_angle):.2f}° "
              f"(success_rate={success_rate:.2%})")

    # ------------------------------------------------------------
    # 其余辅助函数（未改变或仅微调注释）
    # ------------------------------------------------------------
    def _reset_internal_buffers(self):
        self.ego_info = self.target_info = None
        self.obstacles: List[List[Tuple[float, float]]] = []
        self.obstacle_geoms, self.vehicle_poly = [], None
        self.current_scenario = ""
        self.step_count = 0; self.prev_dist = float("inf")

    def _update_obstacle_geometries(self):
        from shapely.geometry import Polygon, LineString, Point
        self.obstacle_geoms = []
        for obs in self.obstacles:
            if len(obs) >= 3: self.obstacle_geoms.append(Polygon(obs))
            elif len(obs) == 2: self.obstacle_geoms.append(LineString(obs))
            elif len(obs) == 1: self.obstacle_geoms.append(Point(obs[0]))
        self.lidar.update_obstacles(self.obstacle_geoms)

    # ---------------- Reward / obs / termination ---------------
    def _calc_reward(self, terminated: bool, collised: bool) -> float:
        if terminated and not collised:
            n = self.vehicle.switch_count
            return max(1.0 - 0.1 * n, 0.3)
        return 0.0

    def _get_observation(self):
        x, y, yaw = self.vehicle.get_pose_center()
        radar = self.lidar.scan(x, y, yaw) / self.lidar_max_range
        dx, dy = self.target_info[0] - x, self.target_info[1] - y
        dist   = math.hypot(dx, dy)
        rel_ang = _normalize_angle(math.atan2(dy, dx) - yaw)
        heading = _normalize_angle(self.target_info[2] - yaw)
        state = np.array([self.vehicle.state[3] / self.max_speed,
                          self.vehicle.state[4] / self.max_steer,
                          dist / self.cfg["world_size"],
                          rel_ang / math.pi,
                          heading / math.pi], np.float32)
        return np.concatenate([radar.astype(np.float32), state])

    def _check_termination(self):
        """返回 (终止?, 超时?, 碰撞?)"""
        x, y, yaw = self.vehicle.get_pose_center()
        tx, ty, tyaw = self.target_info
        dist = math.hypot(tx - x, ty - y)
        angle_diff = abs(_normalize_angle(yaw - tyaw))

        # ---- 成功判定（动态阈值）----
        if dist < self.success_dist and angle_diff < self.success_angle:
            return True, False, False

        # ---- 失败相关判定 ----------
        if self.vehicle_poly and any(self.vehicle_poly.intersects(o)
                                     for o in self.obstacle_geoms):
            return True, False, True           # 碰撞
        if math.hypot(tx - x, ty - y) > self.cfg["world_size"]*1.2:
            return False, True, True           # 出界
        if self.step_count >= self.max_steps:
            return False, True, False          # 超时
        return False, False, False

    # ---------------- Misc --------------------------------------
    def seed(self, seed=None):
        random.seed(seed); np.random.seed(seed)

    def close(self):
        if self.renderer:
            self.renderer.render = lambda *a, **k: None

    def render(self):
        x, y, yaw = self.vehicle.get_pose_center()
        self.renderer.render(
            vehicle_poly=self.vehicle_poly,
            vehicle_state=(x, y, yaw, self.vehicle.state[3], self.vehicle.state[4]),
            target_info=self.target_info,
            obstacles=self.obstacles,
            lidar=self.lidar,
            lidar_max_range=self.lidar_max_range,
            step=self.step_count,
            max_steps=self.max_steps,
            scenario_name=self.current_scenario,
            parking_length=self.parking_length,
            parking_width=self.parking_width,
        )
