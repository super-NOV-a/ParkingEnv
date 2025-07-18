#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_action_from_json_continuous.py (rev2)
--------------------------------------------
1. 先用 **离散弧长 + 离散转向角** (δ, s) 贪婪分段拟合 ground‑truth 轨迹；
2. 再把每段 (δ, s) 解析为连续 **转角增量 dδ + 加速度 a** 序列；
3. 比较三条轨迹：原始 / (δ,s) 重构 / (dδ,a) 重构。

⚠️ 修复 v1 在误差计算时的 shape 不匹配
   `traj_arc` 比 `path` 要短，直接使用 `len(path)` 会触发 broadcasting 错误。
   现分别计算 `min_len_arc` 与 `min_len_prim`，保证数组同长再做减法。
"""

import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# -------------------- helpers --------------------


def normalize_angle(a: float) -> float:
    """Wrap angle to (-π, π]"""
    return (a + math.pi) % (2 * math.pi) - math.pi


# -------------------- kinematics --------------------


def step_arc(state: np.ndarray, delta_abs: float, s: float, L: float = 3.0):
    """Single-track bicycle: drive arc length s with constant steering angle delta_abs."""
    x, y, th = state
    if abs(delta_abs) < 1e-6:  # straight
        x += s * math.cos(th)
        y += s * math.sin(th)
    else:
        R = L / math.tan(delta_abs)
        dth = s / R
        x += R * (math.sin(th + dth) - math.sin(th))
        y -= R * (math.cos(th + dth) - math.cos(th))
        th += dth
    return np.array([x, y, normalize_angle(th)])


def step_inc(state5: np.ndarray, delta_inc: float, a: float, dt: float, vmax: float = 3.0, L: float = 3.0):
    """Primitive step: steering increment + constant accel during dt."""
    x, y, th, v, delta_abs = state5
    delta_new = delta_abs + delta_inc
    v_new = np.clip(v + a * dt, -vmax, vmax)
    s = (v + v_new) * 0.5 * dt  # trapezoid integration
    pose = step_arc(np.array([x, y, th]), delta_new, s, L)
    return np.hstack([pose, v_new, delta_new])


# -------------------- utilities --------------------


def cumulative_dist(path_xy: np.ndarray):
    seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    return np.hstack([0.0, np.cumsum(seg)])


# -------------------- extract (δ,s) --------------------


def extract_arc_actions(
    path: np.ndarray,
    delta_space: np.ndarray,
    s_space: np.ndarray,
    L: float = 3.0,
) -> List[Tuple[float, float]]:
    """Greedy segmentation choosing δ, s from discrete sets."""
    cum = cumulative_dist(path[:, :2])
    arcs, idx = [], 0
    while idx < len(path) - 1:
        st = path[idx]
        best = None  # (delta, s, err, next_idx)
        for s in s_space:
            if abs(s) < 1e-9:
                continue
            target_dist = cum[idx] + abs(s)
            next_idx = np.searchsorted(cum, target_dist)
            if next_idx >= len(path):
                continue
            tgt = path[next_idx]
            err_best, cand = float("inf"), None
            for d in delta_space:
                pred = step_arc(st, d, s, L)
                err = np.linalg.norm(pred[:2] - tgt[:2])
                if err < err_best:
                    cand, err_best = (d, s), err
            if best is None or err_best < best[2]:
                best = (cand[0], cand[1], err_best, next_idx)
        if best is None:
            # fallback: drive straight to the end
            rem = cum[-1] - cum[idx]
            s_fallback = np.clip(rem, -abs(s_space).max(), abs(s_space).max())
            best = (0.0, s_fallback, 0.0, len(path) - 1)
        arcs.append((best[0], best[1]))
        idx = best[3]
    return arcs


# -------------------- (δ,s) → (dδ,a) --------------------


def arc_to_primitives(
    arc_actions: List[Tuple[float, float]],
    dt: float = 0.1,
    n_split: int = 10,
    vmax: float = 3.0,
) -> List[Tuple[float, float]]:
    """Expand each (δ, s) chunk into n primitives of (dδ, a)."""
    prim, v, delta_prev = [], 0.0, 0.0
    for delta_abs, s_target in arc_actions:
        n = n_split
        while True:
            a = 2.0 * (s_target - n * v * dt) / (dt ** 2 * n ** 2)
            v_end = v + a * dt * n
            if abs(v_end) > vmax:
                n += 1  # subdivide further
                continue
            break
        ddelta = (delta_abs - delta_prev) / n
        for _ in range(n):
            prim.append((ddelta, a))
            v = np.clip(v + a * dt, -vmax, vmax)
            delta_prev += ddelta
        delta_prev = delta_abs  # snap to target to avoid drift
    return prim


# -------------------- rollouts --------------------


def rollout_arc(start_state: np.ndarray, arc_actions, L: float = 3.0):
    traj, st = [start_state], start_state
    for d, s in arc_actions:
        st = step_arc(st, d, s, L)
        traj.append(st)
    return np.vstack(traj)


def rollout_primitive(start_state: np.ndarray, prim_actions, dt: float, vmax: float = 3.0, L: float = 3.0):
    st = np.hstack([start_state, 0.0, 0.0])  # x,y,th,v,delta_abs
    traj = [st[:3]]
    for dδ, a in prim_actions:
        st = step_inc(st, dδ, a, dt, vmax, L)
        traj.append(st[:3])
    return np.vstack(traj)


# -------------------- synthetic demo path --------------------


def demo_path(a: float = 50.0, N: int = 3000):
    t = np.linspace(0, 2 * np.pi, N)
    x, y = a * np.sin(t), a * np.sin(t) * np.cos(t)
    dtp = t[1] - t[0]
    th = np.arctan2(np.gradient(y, dtp), np.gradient(x, dtp))
    return np.column_stack([x, y, th])


# -------------------- main --------------------


def main():
    p = argparse.ArgumentParser("离散弧长拟合 & 原子动作反解")
    p.add_argument("--input", default="", help="groundtruth JSON")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--split", type=int, default=10)
    args = p.parse_args()

    # Ground‑truth path
    if args.input and Path(args.input).is_file():
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        path = np.array([[d["x"], d["y"], d["theta"]] for d in data])
    else:
        print("[INFO] demo ∞‑curve")
        path = demo_path()

    # Discrete action sets (can be replaced by real policy)
    delta_space = np.radians([-30, -24, -18, -12, -8, -5, -2, 0, 2, 5, 8, 12, 18, 24, 30])
    s_space = np.array([1.0, 0.25, -0.25, -1.0])
    vmax = 3.0

    # Stage‑1: discrete segmentation
    arc_acts = extract_arc_actions(path, delta_space, s_space)

    # Stage‑2: primitive expansion
    prim_acts = arc_to_primitives(arc_acts, dt=args.dt, n_split=args.split, vmax=vmax)

    # Rollouts
    traj_arc = rollout_arc(path[0], arc_acts)
    traj_prim = rollout_primitive(path[0], prim_acts, args.dt, vmax=vmax)

    # --- error metrics ---
    min_len_arc = min(len(path), len(traj_arc))
    min_len_prim = min(len(path), len(traj_prim))

    err_arc = np.linalg.norm(traj_arc[:min_len_arc, :2] - path[:min_len_arc, :2], axis=1)
    err_prim = np.linalg.norm(traj_prim[:min_len_prim, :2] - path[:min_len_prim, :2], axis=1)

    print(f"(δ,s)  mean/max: {err_arc.mean():.3f}/{err_arc.max():.3f} m")
    print(f"(dδ,a) mean/max: {err_prim.mean():.3f}/{err_prim.max():.3f} m")

    # --- visualization ---
    plt.figure(figsize=(8, 6))
    plt.plot(path[:, 0], path[:, 1], label="groundtruth")
    plt.plot(traj_arc[:, 0], traj_arc[:, 1], label="(δ,s) recon")
    plt.plot(traj_prim[:, 0], traj_prim[:, 1], label="(dδ,a) recon")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Trajectory comparison")
    plt.show()


if __name__ == "__main__":
    main()
