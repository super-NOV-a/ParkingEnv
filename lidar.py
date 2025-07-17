"""lidar_numba.py — Numba-accelerated 2‑D lidar module
(完整代码，已加入 __main__ 基准测试)
"""
from __future__ import annotations
import math
from typing import List
import time

import numpy as np
from shapely.geometry import Polygon, LineString, Point

# ------------------------------------------------------------------
# 0. Numba 可选导入
# ------------------------------------------------------------------
try:
    import numba as nb
    USING_NUMBA = True
except ModuleNotFoundError:  # 无 Numba → 回退纯 Python
    USING_NUMBA = False
    print("[Lidar] numba 未安装，使用纯 Python 路径 (较慢)。")

# ------------------------------------------------------------------
# 1. Numba 射线‑线段求交核心
# ------------------------------------------------------------------
if USING_NUMBA:

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _ray_trace_batch(x0: float, y0: float,
                         dir_cos: np.ndarray, dir_sin: np.ndarray,
                         segs: np.ndarray, r_max: float) -> np.ndarray:
        """批量射线与线段相交最近距离 (Numba)
        使用标准 2‑D 叉积公式：
            r = (cx, sx)      — 射线方向 (单位向量)
            s = (x2-x1, y2-y1) — 线段向量
            denom = r × s;  若 |denom|<eps ⇒ 平行无交
            t = (q-p) × r / denom   (q = seg 起点, p = ray 原点)
            u = (q-p) × s / denom
            条件: 0≤t≤1 且 u>0
        返回距离 = u (因 |r| = 1)。
        """
        n_beam = dir_cos.shape[0]
        out = np.full(n_beam, r_max, dtype=np.float32)
        for i in nb.prange(n_beam):
            cx = dir_cos[i]
            sx = dir_sin[i]
            best = r_max
            for s in range(segs.shape[0]):
                x1, y1, x2, y2 = segs[s]
                denom = cx * (y2 - y1) - sx * (x2 - x1)  # r × s
                if abs(denom) < 1e-12:
                    continue  # 平行
                t = ((x1 - x0) * sx - (y1 - y0) * cx) / denom  # (q-p) × r / denom
                if t < 0.0 or t > 1.0:
                    continue  # 不在线段内
                u = ((x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)) / denom  # (q-p) × s / denom
                if u <= 0.0:
                    continue  # 射线反向
                if u < best:
                    best = u
            out[i] = best
        return out

# ------------------------------------------------------------------
# 2. Lidar2D 类（接口与原版兼容）
# ------------------------------------------------------------------
class Lidar2D:
    def __init__(self, cfg: dict):
        self.min_range = cfg.get('range_min', 0.2)
        self.max_range = cfg.get('max_range', 10.0)
        self.num_beams = cfg.get('num_beams', 72)
        ang_range_deg = cfg.get('angle_range', 360.0)
        self.noise = cfg.get('noise', False)
        self.std = cfg.get('std', 0.05)
        self.angle_std = cfg.get('angle_std', 0.5 * math.pi / 180)

        self.angles = np.linspace(-math.radians(ang_range_deg) / 2,
                                  math.radians(ang_range_deg) / 2,
                                  self.num_beams, dtype=np.float32)
        self.ranges = np.full(self.num_beams, self.max_range, dtype=np.float32)
        self._segments = np.empty((0, 4), dtype=np.float32)  # (N,4) x1,y1,x2,y2

    # --------------------------------------------------------------
    def update_obstacles(self, obstacles: List):
        """将障碍物离散为线段集合供 Numba 使用"""
        segs: List[List[float]] = []
        for ob in obstacles:
            if isinstance(ob, Polygon):
                coords = list(ob.exterior.coords)
                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i + 1]
                    segs.append([x1, y1, x2, y2])
            elif isinstance(ob, LineString):
                coords = list(ob.coords)
                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i + 1]
                    segs.append([x1, y1, x2, y2])
            elif isinstance(ob, Point):
                # 以 8 边形近似点
                r = 0.05
                cx, cy = ob.x, ob.y
                pts = [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 9)]
                for i in range(8):
                    x1, y1 = pts[i]
                    x2, y2 = pts[i + 1]
                    segs.append([x1, y1, x2, y2])
        self._segments = np.asarray(segs, dtype=np.float32) if segs else np.empty((0, 4), dtype=np.float32)

    # --------------------------------------------------------------
    def scan(self, x: float, y: float, yaw: float) -> np.ndarray:
        if self._segments.shape[0] == 0:
            self.ranges.fill(self.max_range)
            return self.ranges.copy()

        angs = self.angles + yaw
        if self.noise and self.angle_std > 0:
            angs += np.random.normal(0.0, self.angle_std, size=self.num_beams)
        dir_c = np.cos(angs, dtype=np.float32)
        dir_s = np.sin(angs, dtype=np.float32)

        if USING_NUMBA:
            dists = _ray_trace_batch(x, y, dir_c, dir_s, self._segments, self.max_range)
        else:
            dists = np.full(self.num_beams, self.max_range, dtype=np.float32)
            for i in range(self.num_beams):
                cx, sx = dir_c[i], dir_s[i]
                best = self.max_range
                for x1, y1, x2, y2 in self._segments:
                    denom = (x2 - x1) * sx - (y2 - y1) * cx
                    if abs(denom) < 1e-12:
                        continue
                    t = ((x1 - x) * sx - (y1 - y) * cx) / denom
                    if t < 0.0 or t > 1.0:
                        continue
                    u = ((x1 - x) * (y2 - y1) - (y1 - y) * (x2 - x1)) / denom
                    if u <= 0.0:
                        continue
                    if u < best:
                        best = u
                dists[i] = best
        if self.noise and self.std > 0:
            dists += np.random.normal(0.0, self.std, size=self.num_beams)
        np.clip(dists, self.min_range, self.max_range, out=dists)
        self.ranges[:] = dists.astype(np.float32)
        return self.ranges.copy()

# ------------------------------------------------------------------
# 3. Benchmark (仅在直接运行本文件时执行)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import random
    print("=== Lidar2D Numba Benchmark ===")

    # 生成随机障碍（矩形多边形）
    random.seed(42)
    np.random.seed(42)
    world = 30.0
    n_obs = 100
    obstacles = []
    for _ in range(n_obs):
        w, h = random.uniform(0.5, 2.0), random.uniform(0.5, 2.0)
        cx = random.uniform(0, world)
        cy = random.uniform(0, world)
        ang = random.uniform(0, 2 * math.pi)
        rect = [(-w/2, -h/2), (-w/2, h/2), (w/2, h/2), (w/2, -h/2)]
        poly = [(
            cx + px * math.cos(ang) - py * math.sin(ang),
            cy + px * math.sin(ang) + py * math.cos(ang)
        ) for px, py in rect]
        obstacles.append(Polygon(poly))

    cfg = {
        'max_range': 15.0,
        'num_beams': 180,
        'angle_range': 360,
        'noise': False,
    }

    # --- 1. Numba 路径 -------------------------------------------
    lidar_fast = Lidar2D(cfg)
    lidar_fast.update_obstacles(obstacles)
    t0 = time.perf_counter()
    for _ in range(1_000):
        _ = lidar_fast.scan(15.0, 15.0, 0.0)
    t_fast = time.perf_counter() - t0
    print(f"Numba  path: {t_fast*1000:.2f} ms / 1000 scans")

    # --- 2. 纯 Python 路径 ---------------------------------------
    # 暂时关闭 Numba 标志，并重新实例化
    USING_NUMBA_ORIG = USING_NUMBA
    USING_NUMBA = False
    lidar_py = Lidar2D(cfg)  # 创建新对象以捕获标志
    lidar_py.update_obstacles(obstacles)
    t0 = time.perf_counter()
    for _ in range(1_000):
        _ = lidar_py.scan(15.0, 15.0, 0.0)
    t_py = time.perf_counter() - t0
    print(f"Python path: {t_py*1000:.2f} ms / 1000 scans")

    # 恢复标志
    USING_NUMBA = USING_NUMBA_ORIG

    speedup = t_py / t_fast if t_fast > 0 else float('inf')
    print(f"Speed‑up ≈ {speedup:.1f}× (Numba vs Python)")
