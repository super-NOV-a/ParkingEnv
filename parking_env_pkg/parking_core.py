from __future__ import annotations
import math, random, numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque

import gymnasium as gym

from .utils import _normalize_angle
from vehicles.vehicle_continuous import VehicleContinuous
from vehicles.vehicle_disc_accel import VehicleDiscAccel
from vehicles.vehicle_arc import VehicleArc
from vehicles.vehicle_incremental import VehicleIncremental
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
    "incremental": (
        VehicleIncremental,
        lambda cls: gym.spaces.Box(low=np.array([-1., -1.]),
                                   high=np.array([1., 1.]),
                                   dtype=np.float32)),
}
_DEFAULT_TYPE = "continuous"


class ParkingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "ParkingEnv"}

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        # ======================== 能量场 ========================
        self.energy       = cfg.get("energy", False)      # 是否使用能量场
        self.energy_nodes = None
        self.energy_last = 0
        self.energy_total = 0                             #初始能量
        # ======================== 基本仿真参数 ========================
        self.dt         = cfg.get("timestep",   0.1)     # 时间步长
        self.max_steps  = cfg.get("max_steps",  500)     # 最大步数
        self.render_mode = cfg.get("render_mode", "human")

        # ======================== 车辆参数配置 ========================
        self.wheelbase   = 3.0
        self.front_hang  = 0.925
        self.rear_hang   = 1.025    # 车长4.95
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
            num_beams=cfg.get("lidar_beams", 72),
            noise=False,
            std=0.05,
            angle_std=0.5,
            position_offset=(0.0, 0.0)
        )
        self.lidar = Lidar2D(lidar_cfg)

        # ======================== 状态空间与观测空间 ========================
        # 观测维度从 9 → 11，加入 direction 与 switch_frac
        state_low  = [-1., -1., 0., -1., -1., -1., -1.,  -1., -1., -1., 0.]
        state_high = [ 1.,  1., 1.,  1.,  1.,  1.,  1.,   1.,  1.,  1., 1.]

        self.observation_space = gym.spaces.Box(
            low=np.array([-1.] * lidar_cfg["num_beams"] + state_low, np.float32),   # 雷达距离减碰撞阈值
            high=np.array([1.] * lidar_cfg["num_beams"] + state_high, np.float32),
            dtype=np.float32
        )

        # ======================== 场景生成配置 ========================
        cfg_defaults = dict(
            world_size=30.0,
            scenario_mode="random",
            wall_thickness=0.15
        )
        self.cfg = {**cfg_defaults, **cfg,
                    "parking_length": self.parking_length,
                    "parking_width":  self.parking_width,
                    "car_length":     self.car_length,
                    "car_width":      self.car_width}
        self.scenario = ScenarioManager(self.cfg)
        self.scenario_mode = self.cfg.get("scenario_mode", "random").lower()
        self.is_file = (self.scenario_mode == "file")
        self.renderer = PygameRenderer() if self.render_mode == "human" else None

        # ======================== 难度等级管理 ========================
        self.dist_levels  = np.linspace(2.0, 1, 11)
        self.angle_levels = np.radians(np.linspace(36, 18, 11))
        # self.dist_levels  = np.linspace(2.0, 0.25, 11)
        # self.angle_levels = np.radians(np.linspace(36, 3, 11))
        self.level = int(np.clip(cfg.get("difficulty_level", 0), 0, 10))
        self.success_dist  = float(self.dist_levels[self.level])
        self.success_angle = float(self.angle_levels[self.level])

        # ======================== 碰撞检测缓存 ========================
        self.vehicle_poly = self.vehicle.get_shapely_polygon()
        # self._radar: Optional[np.ndarray] = None         # 原始射线距离
        self._clearance: Optional[np.ndarray] = None     # radar - threshold
        self._collision_thresholds: np.ndarray = np.empty(self.lidar.num_beams, dtype=np.float32)
        self._init_vehicle_collision_thresholds()

        # ======================== 离散动作编码查找表 ========================
        if self.N_STEER > 1:
            self._steer_norm_lut = np.linspace(-1.0, 1.0, self.N_STEER, dtype=np.float32)
        if self.N_ARC > 1:
            self._arc_norm_lut = np.linspace(-1.0, 1.0, self.N_ARC, dtype=np.float32)

        # ======================== 观测缓存 ========================
        self._state_buf = np.empty(len(state_low), dtype=np.float32)
        self._obs_buf   = np.empty(self.lidar.num_beams + len(state_low), dtype=np.float32)
        # self._prev_action = np.array([self.N_STEER // 2, self.N_ARC // 2], dtype=np.int32)
        # 2) _prev_action 用 float 保存
        self._prev_action = np.zeros(2, dtype=np.float32)

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
        self._clearance = None

        if seed is not None:
            self.seed(seed)
        if self.energy:
            (self.ego_info, self.target_info, self.obstacles) ,jr_pkgs = \
                self.scenario.init(seed=seed, scenario_idx=scenario_idx, current_level=self.level, energy=self.energy)
            self.energy_nodes = jr_pkgs
 
            #   能量场初始化
            self.energy_total = 0
 
            ego_centor_x, ego_centor_y, ego_centor_yaw = self.ego_info[:3]
            ego_rear_x, ego_rear_y = _shift_forward(ego_centor_x, ego_centor_y, ego_centor_yaw)  # 目前ego是中心点，处理成后轴中心点
            # target_centor_x, target_centor_y, target_centor_yaw = self.target_info[:3]
            # target_rear_x, target_rear_y = _shift_forward(target_centor_x, target_centor_y, target_centor_yaw)  # 目前ego是中心点，处理成后轴中心点
            target_rear_x, target_rear_y, target_centor_yaw = self.target_rear
           
            self.energy_last = get_energy(self.energy_nodes, [ego_rear_x, ego_rear_y, ego_centor_yaw])
            self.energy_target = get_energy(self.energy_nodes, [target_rear_x, target_rear_y, target_centor_yaw])
            self.energy_offset = self.energy_target - self.energy_last
            # print("--------------energy_last:", self.energy_last)
            # print("---------------------energy_target :", self.energy_target )
            # print("-----------------------energy_offset :", self.energy_offset )
        else:
            (self.ego_info, self.target_info, self.obstacles), _ = \
                self.scenario.init(seed=seed, scenario_idx=scenario_idx, current_level=self.level)
            
        self.current_scenario = scenario_idx

        # === 1) 车位后轴中心（仅一次，全局复用） ===
        tx_c, ty_c, tyaw = self.target_info
        rear_dx = -1.4     # ⇐ 你的车辆几何中心 ↔ 后轴中心平移量
        tr_x, tr_y = _shift_forward(tx_c, ty_c, tyaw, dx=rear_dx)
        self.target_rear = (tr_x, tr_y, tyaw)

        # ego_info 仍然是几何中心 —— 先平移到后轴
        rear_x, rear_y = _shift_forward(
            self.ego_info[0], self.ego_info[1], self.ego_info[2]
        )
        self.vehicle.reset_state(rear_x, rear_y, self.ego_info[2])
        self.step_count   = 0
        self.prev_dist    = float("inf")
        self.vehicle_poly = self.vehicle.get_shapely_polygon()
        self._update_obstacle_geometries()
        self._prev_action = np.array([self.N_STEER // 2, self.N_ARC // 2], dtype=np.int32)
        self._reset_reward_state()

        return self._get_observation(), {}

    def step(self, action):
        self._clearance = None      # 在懒加载中重新读写
        self.vehicle.state, _ = self.vehicle.step(action)
        self.vehicle_poly = self.vehicle.get_shapely_polygon()

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
        self._prev_action = np.asarray(action, dtype=np.float32)  # 不再 cast int32
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
        """
        懒加载净空距（雷达距离减去车身轮廓阈值）。
        结果缓存到 ``self._clearance``，同一帧仅计算一次。
        """
        if self._clearance is None:
            radar = self.lidar.scan(x, y, yaw)                       # 原始射线距离
            self._clearance = radar - self._collision_thresholds     # 可正可负
        # return self._clearance

    # ------------------------------------------------------------------
    # Observation / Reward / Termination
    # ------------------------------------------------------------------
    def _get_observation(self):
        x, y, yaw = self.vehicle.get_pose_center()
        self._scan_radar(x, y, yaw)     ## 缓存供碰撞检测
        self._obs_buf[:self.lidar.num_beams] = self._clearance / self.lidar.max_range

        # dx, dy = self.target_info[0] - x, self.target_info[1] - y
        dx, dy = self.target_rear[0] - x, self.target_rear[1] - y
        dist   = math.hypot(dx, dy)
        rel_ang = _normalize_angle(math.atan2(dy, dx) - yaw)
        # heading = _normalize_angle(self.target_info[2] - yaw)
        heading = _normalize_angle(self.target_rear[2] - yaw)

        heading_sin, heading_cos = math.sin(heading), math.cos(heading)
        bearing_sin, bearing_cos = math.sin(rel_ang), math.cos(rel_ang)
        prev_s, prev_a = self._encode_prev_action()

        direction = 1.0 if self.vehicle.state[3] >= 0 else -1.0
        switch_frac = min(self.vehicle.switch_count / 20.0, 1.0)
        self._state_buf[:] = [
            self.vehicle.state[3] / self.max_speed,     # speed
            self.vehicle.state[4] / self.max_steer,     # steer
            min(dist, 30.0) / 30.0,                     # dist to target
            bearing_sin, bearing_cos,                   # relative target position sin\cos
            heading_sin, heading_cos,                   # relative target heading sin\cos
            prev_s, prev_a,                             # last_action
            direction, switch_frac,                     # direction 
        ]
        self._obs_buf[self.lidar.num_beams:] = self._state_buf
        return self._obs_buf
    
    def _calc_reward(self, terminated: bool, trunc: bool, collided: bool, need_energy: bool) -> float:
        if self.scenario_mode == "parking":
            return self._calc_reward_parking(terminated, trunc, collided)
        if need_energy and self.is_file:
            rear_x, rear_y, rear_yaw = self.vehicle.state[:3]
            energy = get_energy(self.energy_nodes, [rear_x, rear_y, rear_yaw])
            # energy_reward = (energy - self.energy_last) / (500 * 3)
            energy_reward = (energy - self.energy_last) / (self.energy_offset) / 3
            self.energy_last =  energy
            self.energy_total += energy_reward

            if terminated and not collided:
                n_switch = self.vehicle.switch_count
                return max(0.05 * (20 - n_switch), 0.3)*(1+0.1*self.level) + energy_reward
            return energy_reward
        # 终点且无碰撞
        if terminated and not collided:
            # 统计换挡次数， ≤2 次不扣，之后每次-0.05   # 保底 0.3，避免负值
            reward  = max(1.0 - 0.05 * max(0, self.vehicle.switch_count - 2), 0.3)                    
            return reward * (1 + 0.1 * self.level)         # 难度加成
        # 对于更高难度鼓励智能体,且减少换向惩罚
        # if trunc or collided:
        #     return -0.05
        return 0.0
    
    def _calc_reward_parking(self, terminated: bool, trunc: bool, collision: bool) -> float:
        """
        • shaping 总和 ≤0.5 ： r_shape = 0.5·(φ - φ_prev)
        • success (terminated & !collision):
            base = max(0.2, 1 - 0.1·switch_cnt)
        • collision → 0
        """
        # ---------- shaping -------------------------------------------------
        rx, ry, ryaw = self.vehicle.state[:3]
        phi = self._calc_shaping(rx, ry, ryaw)
        r_shape = (phi - getattr(self, "_phi_prev", phi)) * 0.5
        self._phi_prev = phi           # 更新缓存

        # ---------- 终止情况 -----------------------------------------------
        if collision or trunc:
            return -0.3                                     # 直接 0

        if terminated:                                     # 成功
            switch_cnt = max(0, getattr(self, "switch_cnt", 0)-2)   # 小于等于两次不扣
            success_reward = max(0.3, 1.0 - 0.1 * switch_cnt) * (1 + 0.1 * self.level)
            return success_reward

        # ---------- 普通步 --------------------------------------------------
        return r_shape

    def _calc_shaping(self, x: float, y: float, yaw: float) -> float:
        """
        φ = φ_dist + φ_yaw ∈ [0,1]
        • φ_dist : 0.5·exp(-k·dist)          （dist=0 ⇒0.5，dist→∞ ⇒0）
        • φ_yaw  : 距离≤8 m 时 0–0.5，yaw_err=0 ⇒0.5
        """
        dx = self.target_rear[0] - x
        dy = self.target_rear[1] - y
        dist = math.hypot(dx, dy)

        # ---- 距离势能 0–0.5 -------------------------------------
        # 距离奖励曲线参数
        # _K_DIST    = 0.1       # tanh 斜率，10m->0.18 50m->0.003
        phi_dist = 0.01 * (-dist + 50)

        # ---- 姿态势能 0–0.5（仅近距离） --------------------------
        if dist <= 20.0:
            yaw_err = abs(_normalize_angle(self.target_rear[2] - yaw))
            phi_yaw = 0.5 * (1.0 - yaw_err / math.pi)
        else:
            phi_yaw = 0.0

        return phi_dist + phi_yaw          # 上限 1.0

    def _check_termination(self):
        x, y, yaw = self.vehicle.get_pose_center()
        # tx, ty, tyaw = self.target_info
        tx, ty, tyaw = self.target_rear

        # ---- 成功 —— 距离 + 朝向 ----
        if math.hypot(tx - x, ty - y) < self.success_dist and \
           abs(_normalize_angle(yaw - tyaw)) < self.success_angle:
            return True, False, False

        # ---- 碰撞 ----
        if self._clearance is None:            # 极端情况：外部先调用 _check_termination()
            self._scan_radar(x, y, yaw)
        if np.any(self._clearance <= 0.0):
            return True, False, True   # 碰撞不结束的话需要强制车辆移动到碰撞前才对，不然会强制一直扣分

        # ---- 其他失败 ----
        # if math.hypot(tx - x, ty - y) > self.cfg["world_size"]:
        #     return False, True, True  # Out of bounds 有围墙就不需要判断了
        if self.step_count >= self.max_steps:
            return False, True, False  # Timeout
        return False, False, False

    def _encode_prev_action(self):
        # 离散模式保持旧逻辑
        if getattr(self.vehicle, "N_STEER", 1) > 1:      # 离散 steering  # VehicleArc
            s = self._steer_norm_lut[int(self._prev_action[0])]
            a = self._arc_norm_lut[int(self._prev_action[1])]
        else:                                            # 连续 / 增量   continuous / incremental
            s, a = float(self._prev_action[0]), float(self._prev_action[1])
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

    def _reset_reward_state(self):
        """在 reset() 末尾调用，初始化势函数 φ 与已换挡次数。"""
        rx, ry, ryaw = self.vehicle.state[:3]
        self._phi_prev = self._calc_shaping(rx, ry, ryaw)

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

def _shift_forward(x: float, y: float, yaw: float, dx: float=-1.4):
    """沿 yaw 正方向把 (x, y) 平移 dx。"""
    return x + dx * np.cos(yaw), y + dx * np.sin(yaw)
