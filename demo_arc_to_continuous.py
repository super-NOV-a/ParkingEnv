#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_arc_to_continuous.py  (简版)
=================================
把 VehicleArc 的离散 (steer_idx, arc_idx) 序列
用 VehicleIncremental 的连续动作 [d_steer_norm, acc_norm] 近似重现。
"""
from __future__ import annotations
import math, argparse, random
from typing import List, Tuple, Sequence, Dict

import numpy as np
import matplotlib.pyplot as plt

from vehicles.vehicle_arc import VehicleArc, STEER_CHOICES, ARC_CHOICES
from vehicles.vehicle_incremental import VehicleIncremental   # 新增
try:
    from shapely.geometry import Polygon
    from matplotlib.patches import Polygon as MplPoly
    have_shapely = True
except ModuleNotFoundError:
    pass

# ------------------------------------------------------------
def _solve_const_acc(v0: float, s: float, n: int, dt: float, vmax: float, amax: float):
    """给定位移 s、时间 n*dt、初速 v0，求常加速度。若超限返回 None"""
    a = 2*(s - v0*dt*n)/(dt*dt*n*n)
    v_end = v0 + a*dt*n
    if abs(v_end) > vmax + 1e-6 or abs(a) > amax + 1e-6:
        return None
    return a

# ------------------------------------------------------------------ #
def _arc_to_cont_cmds(
        veh: VehicleIncremental,
        steer_idx: int,
        arc_idx: int,
):
    """
    固定把一次离散弧长(≈1 s)拆成 N_fixed = round(1.0/dt) 个连续动作。
    以 dt=0.2 → N_fixed = 5。
    每一步同时做方向盘增量和加/减速，使总位移逼近 s_target。
    ------------------------------------------------------------------
    输出 : list[(d_steer_norm, acc_norm)]  —— 长度恒为 N_fixed
    """
    # ---------- 固定步数 ----------
    N_fixed = max(1, int(round(1.0 / veh.dt)))   # e.g. 1/0.2 = 5

    # ---------- 目标量 ----------
    target_steer = STEER_CHOICES[steer_idx]       # rad
    s_target     = ARC_CHOICES[arc_idx]           # m (正前负后)
    sign_s       = 1.0 if s_target >= 0 else -1.0

    # ---------- 方向盘分布（余弦缓动） ----------
    delta_steer = target_steer - veh.state[4]
    cos_prof    = 0.5 * (1 - np.cos(np.linspace(0, math.pi, N_fixed, dtype=np.float32)))
    steer_steps = delta_steer * cos_prof / (cos_prof.sum() + 1e-9) * N_fixed
    d_steer_norm = steer_steps / (veh.max_steer_rate * veh.dt)
    d_steer_norm = np.clip(d_steer_norm, -1.0, 1.0)

    # ---------- 计算常加速度 a，使 N_fixed 步位移≈ s_target ----------
    v0   = veh.state[3]                 # 当前速度
    dt   = veh.dt
    N    = N_fixed
    a_req = 2 * (s_target - v0 * dt * N) / (dt * dt * N * (N + 1))  # 推导见注释
    a_req = np.clip(a_req, -veh.max_acc, veh.max_acc)
    acc_norm = float(a_req / veh.max_acc)
    acc_profile = np.full(N_fixed, acc_norm, dtype=np.float32)

    # ---------- 打包 ----------
    cmds = list(zip(d_steer_norm.tolist(), acc_profile.tolist()))
    return cmds


# ------------------------------------------------------------
def replay_action_sequence(
        actions: Sequence[Tuple[int, int]],
        start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt: float = 0.1,
        *,
        verbose: bool = False,          # << 新增
):
    """
    回放离散动作序列，并用连续模型拟合。
    若 verbose=True，则逐帧打印：
        [ARC ] step k    action=(s_idx,a_idx)    pose=(x,y,yaw°)
        [CONT] micro j   a=(d_steer_norm,acc_norm) pose=(x,y,yaw°)
    """
    base = dict(wheelbase=3.0, width=2.0,
                front_hang=1.0, rear_hang=1.0,
                max_steer=math.radians(30), max_speed=3.0,
                dt=dt)

    veh_arc = VehicleArc(**base)
    veh_inc = VehicleIncremental(max_acc=1.0,
                                 max_steer_rate=math.radians(60),
                                 **base)

    veh_arc.reset_state(*start_pose)
    veh_inc.reset_state(*start_pose)

    traj_arc, traj_cont = [veh_arc.get_pose_center()], [veh_inc.get_pose_center()]

    for k, (s_idx, a_idx) in enumerate(actions):
        # ── ARC 车辆走一步 ──────────────────────────────
        veh_arc.step((s_idx, a_idx))
        x_a, y_a, yaw_a = veh_arc.get_pose_center()
        traj_arc.append((x_a, y_a, yaw_a))
        if verbose:
            print(f"[ARC ] step {k:03d}  action=({s_idx},{a_idx}) "
                  f"pose=({x_a:+7.2f},{y_a:+7.2f},{math.degrees(yaw_a):+6.1f}°)")

        # ── CONT 车辆用若干 micro-steps 拟合 ─────────────
        cmds = _arc_to_cont_cmds(veh_inc, s_idx, a_idx)
        for j, (d_steer, acc) in enumerate(cmds):
            veh_inc.step((d_steer, acc))
            x_c, y_c, yaw_c = veh_inc.get_pose_center()
            traj_cont.append((x_c, y_c, yaw_c))
            if verbose:
                print(f"  [CONT] micro {j:02d} "
                      f"a=({d_steer:+5.2f},{acc:+5.2f}) "
                      f"pose=({x_c:+7.2f},{y_c:+7.2f},{math.degrees(yaw_c):+6.1f}°)")

    return np.asarray(traj_arc), np.asarray(traj_cont)

# ------------------------------------------------------------
def _rmse(a: np.ndarray, b: np.ndarray):
    m = min(len(a), len(b))
    return float(np.sqrt(((a[:m,:2]-b[:m,:2])**2).sum(axis=1).mean()))

# -----------------------------------------------------------------------------
#  Plotting helper  —— 新版，支持朝向 / 车身轮廓
# -----------------------------------------------------------------------------
def plot_two_trajs(traj_arc: np.ndarray,
                   traj_cont: np.ndarray,
                   title: str = "",
                   veh_dims: dict | None = None,
                   skip: int = 5):
    """
    traj_* : N×3  (x, y, yaw)
    veh_dims: {"length": L, "width": W} —— 不传则只画箭头
    skip    : 每隔 skip 帧绘制一次姿态
    """
    rmse = _rmse(traj_arc, traj_cont)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(traj_arc[:, 0],  traj_arc[:, 1],  "-o", label="Discrete arc", lw=1)
    ax.plot(traj_cont[:, 0], traj_cont[:, 1], "-*", label="Continuous (inc.)", lw=2)

    # ---------- 画朝向箭头 / 车身 ----------
    have_shapely = True
    L = veh_dims.get("length") if veh_dims else None
    W = veh_dims.get("width")  if veh_dims else None
    
    for k in range(0, len(traj_cont), skip):
        x, y, yaw = traj_cont[k]
        # dx = 0.6 * (L or 2.5) * math.cos(yaw)
        # dy = 0.6 * (L or 2.5) * math.sin(yaw)
        # ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.25,
        #          fc="tab:red", ec="tab:red", alpha=0.8, length_includes_head=True)

        # 如果可用 shapely，则再画矩形车身轮廓
        if have_shapely:
            half_w = (W or 1.8) * 0.5
            l_f =  (L or 4.0) * 0.5      # 车头到中心
            l_r = -(L or 4.0) * 0.5      # 车尾到中心
            # 本地坐标系四个角
            corners = np.array([[l_r, -half_w],
                                [l_r,  half_w],
                                [l_f,  half_w],
                                [l_f, -half_w]], dtype=np.float32)
            c, s = math.cos(yaw), math.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            world = corners @ R.T + np.array([x, y])
            patch = MplPoly(world, closed=True, facecolor="tab:orange", alpha=0.25,
                            edgecolor="tab:orange", linewidth=0.7)
            ax.add_patch(patch)

    ax.set_title(f"{title}  |  RMSE={rmse:.3f} m")
    ax.set_aspect("equal", "box")
    ax.grid(True); ax.legend()
    plt.tight_layout(); plt.show(block=True)


# ------------------------------------------------------------
def _demo(n_steps: int):
    rng = random.Random(0)
    actions = [(rng.randrange(len(STEER_CHOICES)), 2)
            for _ in range(n_steps)]
    ta, tc = replay_action_sequence(actions, verbose=True, dt=0.2)
    plot_two_trajs(ta, tc,
                title="Random demo",
                veh_dims={"length": 4.0, "width": 2.0},   # 与 base 参数一致即可
                skip=5)                                    # 每 5 帧绘一次姿态

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=10,
                        help="随机生成 N 步离散动作并演示")
    args = parser.parse_args()
    _demo(args.random)
