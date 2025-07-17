#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_action_from_json.py  ——  (δ,a) 版本
"""

import json, math, argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ----------------- 基础 -----------------
def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

# ----------------- 单轨模型 -----------------
def step_arc(state, delta, s, L=3.0):
    x, y, th = state
    if abs(delta) < 1e-6:
        x += s * math.cos(th)
        y += s * math.sin(th)
    else:
        R   = L / math.tan(delta)
        dth = s / R
        x  += R * (math.sin(th + dth) - math.sin(th))
        y  -= R * (math.cos(th + dth) - math.cos(th))
        th += dth
    return np.array([x, y, normalize_angle(th)])

def step_acc(state, delta, a, dt, vmax=3.0, L=3.0):
    x, y, th, v = state
    v_next = np.clip(v + a * dt, -vmax, vmax)
    s      = (v + v_next) * 0.5 * dt
    pose   = step_arc(np.array([x, y, th]), delta, s, L)
    return np.hstack([pose, v_next])

# ----------------- 连续 → (δ,s) -----------------
def extract_actions_arc(path, delta_space, s_space, L=3.0):
    acts, st = [], path[0]
    for tgt in path[1:]:
        best, err_best = None, float("inf")
        for d in delta_space:
            for s in s_space:
                pred = step_arc(st, d, s, L)
                err  = np.linalg.norm(pred[:2]-tgt[:2])
                if err < err_best: best, err_best = (d,s), err
        acts.append(best)
        st = step_arc(st, *best, L)
    return acts

# ----------------- (δ,s) → N*(δ,a) -----------------
def arc_to_acc_actions(
    arc_actions: List[Tuple[float,float]],
    accel_space: np.ndarray,
    dt: float,
    v0: float,
    n_split: int,
    vmax: float,
) -> List[Tuple[float,float]]:
    """
    输出 [(delta_abs, a), ...] 长度 = len(arc_actions)*n_split
    每块拆 n_split 步，方向角恒为 delta_abs
    a 通过仿真搜索选取，使距离最接近 s_target 且不超速
    """
    acc_acts, v = [], v0
    for delta_abs, s_target in arc_actions:
        best_a, best_err = None, float("inf")
        # 搜索 a
        for a in accel_space:
            v_sim, s_sum = v, 0.0
            for _ in range(n_split):
                v_next = np.clip(v_sim + a*dt, -vmax, vmax)
                s_sum += (v_sim + v_next)*0.5*dt
                v_sim  = v_next
            err = abs(s_sum - s_target)
            if err < best_err:
                best_a, best_err = a, err
        if best_a is None: best_a = 0.0
        # 生成 n_split 步
        for _ in range(n_split):
            acc_acts.append((delta_abs, best_a))
            v = np.clip(v + best_a*dt, -vmax, vmax)
    return acc_acts

# ----------------- 回放 -----------------
def rollout_acc(
    start_state, acc_acts, dt, vmax=3.0, L=3.0
):
    states=[]
    st = np.hstack([start_state,0.0])   # x,y,th,v
    for delta_abs, a in acc_acts:
        st = step_acc(st, delta_abs, a, dt, vmax, L)
        states.append(st)
    return np.vstack(states)

def rollout_arc(start_state, arc_acts, L=3.0):
    states=[start_state]
    st = start_state
    for d,s in arc_acts:
        st = step_arc(st, d, s, L)
        states.append(st)
    return np.vstack(states)

# ----------------- 示例轨迹 -----------------
def demo_path(a=20.0, N=2000):
    t = np.linspace(0, 2*math.pi, N)
    x, y = a*np.sin(t), a*np.sin(t)*np.cos(t)
    dtp  = t[1]-t[0]
    th   = np.arctan2(np.gradient(y,dtp), np.gradient(x,dtp))
    return np.column_stack([x,y,th])

# ----------------- CLI -----------------
def main():
    p = argparse.ArgumentParser("δ,a 反解")
    p.add_argument("--input", default="", help="连续轨迹 JSON")
    p.add_argument("--split", type=int, default=10)
    p.add_argument("--dt", type=float, default=0.2)
    args = p.parse_args()

    # 路径
    if args.input and Path(args.input).is_file():
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        path = np.array([[d["x"],d["y"],d["theta"]] for d in data])
    else:
        print("[INFO] demo ∞‑curve")
        path = demo_path()

    # 离散集合
    delta_space = np.radians([-30,-24,-18,-12,-8,-5,-2,0,2,5,8,12,18,24,30])
    s_space     = np.array([1.0,0.25,-0.25,-1.0])
    accel_space = np.arange(-1.0, 1.05, 0.05)   # 更细
    vmax = 3.0

    arc_acts = extract_actions_arc(path, delta_space, s_space)
    acc_acts = arc_to_acc_actions(
        arc_acts, accel_space, args.dt, v0=0.0, n_split=args.split, vmax=vmax
    )

    traj_arc = rollout_arc(path[0], arc_acts)
    traj_acc = rollout_acc(path[0], acc_acts, args.dt, vmax)

    # 误差（acc 采样每 split 步对齐）
    err_arc = np.linalg.norm(traj_arc[:len(path),:2]-path[:len(traj_arc),:2], axis=1)
    err_acc = np.linalg.norm(
        traj_acc[::args.split,:2][:len(path)] - path[:len(traj_acc[::args.split]),:2], axis=1
    )
    print(f"Arc mean/max: {err_arc.mean():.3f}/{err_arc.max():.3f} m")
    print(f"Acc mean/max: {err_acc.mean():.3f}/{err_acc.max():.3f} m")

    # 绘图
    plt.figure(figsize=(9,7))
    plt.plot(path[:,0], path[:,1], label="continuous")
    plt.plot(traj_arc[:,0], traj_arc[:,1], label="(δ,s) recon")
    plt.plot(traj_acc[:,0], traj_acc[:,1], label="(δ,a) recon")
    plt.axis("equal"); plt.grid(True); plt.legend(); plt.show()

if __name__ == "__main__":
    main()
