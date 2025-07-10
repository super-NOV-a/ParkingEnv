# parking_env_pkg/parking_core.py
from __future__ import annotations
import math, random, numpy as np
from typing import Dict, Optional, Tuple, List, Sequence

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

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        # ---------- basic params ---------------------------------------
        self.dt = cfg.get("timestep", 0.1)
        self.max_steps = cfg.get("max_steps", 500)
        self.render_mode = cfg.get("render_mode", "human")
        # ---------- geometry ------------------------------------------
        self.wheelbase = 3.0; self.front_hang = 1.0
        self.rear_hang = 1.0; self.car_width = 2.0
        self.car_length = self.wheelbase + self.front_hang + self.rear_hang
        self.parking_length, self.parking_width = 5.5, 2.3
        # ---------- vehicle -------------------------------------------
        self.max_steer = math.radians(30.0)
        self.max_speed = cfg.get("max_speed", 3.0)
        self.vehicle_type = cfg.get("vehicle_type", _DEFAULT_TYPE)
        VehicleCls, space_fn = _VEHICLE_REGISTRY[self.vehicle_type]
        self.vehicle = VehicleCls(wheelbase=self.wheelbase, width=self.car_width,
                                  front_hang=self.front_hang, rear_hang=self.rear_hang,
                                  max_steer=self.max_steer, max_speed=self.max_speed,
                                  dt=self.dt)
        self.action_space = space_fn(VehicleCls)
        # ---------- lidar ---------------------------------------------
        lidar_cfg = {"range_min": 0.5,
                     "max_range": cfg.get("lidar_max_range", 30.0),
                     "angle_range": 360, "num_beams": 72,
                     "noise": False, "std": 0.05, "angle_std": 0.5,
                     "position_offset": (0.0, 0.0)}
        self.lidar_max_range = lidar_cfg["max_range"]
        self.lidar = Lidar2D(lidar_cfg)
        state_low = [-1., -1., 0., -1., -1.]
        state_high = [1., 1., 1., 1., 1.]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.] * lidar_cfg["num_beams"] + state_low, np.float32),
            high=np.array([self.lidar_max_range] * lidar_cfg["num_beams"] + state_high, np.float32),
            dtype=np.float32)
        # ---------- scenario & render ---------------------------------
        cfg_defaults = {"world_size": 30.0, "scenario_mode": "random",
                        "min_obstacles": 3, "max_obstacles": 8,
                        "min_obstacle_size": 1.0, "max_obstacle_size": 10.0}
        self.cfg = {**cfg_defaults, **cfg,
                    "parking_length": self.parking_length,
                    "parking_width": self.parking_width,
                    "car_length": self.car_length,
                    "car_width": self.car_width,}
        self.scenario = ScenarioManager(self.cfg)
        self.renderer = PygameRenderer() if self.render_mode == "human" else None
        # ---------- runtime state -------------------------------------
        self._reset_internal_buffers()


    # ==================================================================
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None,
              scenario_idx: Optional[int] = None):
        if seed is not None: self.seed(seed)
        (self.ego_info,
         self.target_info,
         self.obstacles) = self.scenario.init(seed=seed, scenario_idx=scenario_idx)
        self.current_scenario = scenario_idx
        # self._reset_internal_buffers()
        self.vehicle.reset_state(*self.ego_info)
        self.vehicle_poly = self.vehicle.get_shapely_polygon()
        self.prev_dist = float("inf"); self.step_count = 0
        self._update_obstacle_geometries()
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.vehicle.state, _ = self.vehicle.step(action)
        self.vehicle_poly = self.vehicle.get_shapely_polygon()
        obs = self._get_observation()
        term, trunc, coll = self._check_termination()
        reward = self._calc_reward(term, coll)
        self.step_count += 1
        if self.renderer and self.step_count % 5 == 0:
            self.render()
        return obs, reward, term, trunc, {}

    def seed(self, seed=None):
        random.seed(seed); np.random.seed(seed)

    def close(self):
        if self.renderer: self.renderer.render = lambda *a, **k: None

    # ==================================================================
    # Internals
    # ------------------------------------------------------------------
    def _reset_internal_buffers(self):
        self.ego_info = self.target_info = None
        self.obstacles: List[List[Vector]] = []
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

    # ------------------------------------------------------------------
    def _calc_reward(self, terminated, collised):
        x, y, _ = self.vehicle.get_pose_center()
        tx, ty, _ = self.target_info
        dist = math.hypot(tx - x, ty - y)
        scale = self.cfg["world_size"] * 0.5
        r = 2.0 * math.exp(-dist / scale) - 1.0
        if terminated: r += 1.0 if not collised else -1.0
        return r

    def _get_observation(self):
        x, y, yaw = self.vehicle.get_pose_center()
        radar = self.lidar.scan(x, y, yaw) / self.lidar_max_range
        dx, dy = self.target_info[0] - x, self.target_info[1] - y
        dist = math.hypot(dx, dy)
        rel_ang = _normalize_angle(math.atan2(dy, dx) - yaw)
        heading = _normalize_angle(self.target_info[2] - yaw)
        state = np.array([self.vehicle.state[3] / self.max_speed,
                          self.vehicle.state[4] / self.max_steer,
                          dist / self.cfg["world_size"],
                          rel_ang / math.pi,
                          heading / math.pi], np.float32)
        return np.concatenate([radar.astype(np.float32), state])

    # ------------------------------------------------------------------
    def _check_termination(self):
        # 1. collision
        if self.vehicle_poly and any(self.vehicle_poly.intersects(o)
                                     for o in self.obstacle_geoms):
            return True, False, True
        # 2. in slot
        if self.vehicle_poly and self._in_parking_slot():
            return True, False, False
        # 3. out of bound
        x, y, _ = self.vehicle.get_pose_center()
        if math.hypot(self.target_info[0]-x, self.target_info[1]-y) > \
           self.cfg["world_size"]*1.2:
            return False, True, True
        # 4. max steps
        if self.step_count >= self.max_steps:
            return False, True, False
        return False, False, False

    def _in_parking_slot(self):
        from shapely.geometry import Polygon
        slot = Polygon(
            parking_corners(*self.target_info,
                            self.parking_length, self.parking_width))
        inter = self.vehicle_poly.intersection(slot).area
        if inter / self.vehicle_poly.area > 0.9:
            diff = abs(_normalize_angle(self.vehicle.state[2]-self.target_info[2]))
            return diff < math.radians(10)
        return False

    # ------------------------------------------------------------------
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
