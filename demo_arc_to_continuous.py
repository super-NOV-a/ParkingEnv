#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo – discrete arc‑length → incremental continuous actions (look‑ahead version)
================================================================================
**What's new?**  The micro‑planner that converts one discrete arc‑length action
`(steer_idx, arc_idx)` into **N_fixed = round(1 s / dt) = 5** continuous actions
now has *one‑step look‑ahead*. It receives the *next* discrete action as well and
slightly blends towards it in the last micro‑steps. The goal is to avoid the
sharp discontinuity that used to appear at each 1‑s boundary.

* Steering: a cosine profile brings the wheel to the current target in the first
  4 steps, the 5‑th step already starts moving towards the next target (20 % of
  the required delta).
* Longitudinal: still uses constant acceleration to match the current arc
  length, but if the **sign of the next arc** differs, the last step’s
  acceleration is smoothly reduced (50 %).

Everything else – plotting, verbose logging, CLI – stays the same.
"""
from __future__ import annotations

import math
import argparse
import random
from typing import List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from vehicles.vehicle_arc import VehicleArc, STEER_CHOICES, ARC_CHOICES
from vehicles.vehicle_incremental import VehicleIncremental

# -----------------------------------------------------------------------------
# Global parameters (single source of truth)
# -----------------------------------------------------------------------------
DEFAULT_DT = 0.2                     # [s] integrator step → 5 micro‑steps per 1 s
WHEEL_BASE = 3.0                    # [m]
MAX_STEER_DEG = 30                  # [deg]
MAX_STEER_RATE_DEG = 90             # [deg/s]
MAX_SPEED = 3.0                     # [m/s]
MAX_ACC = 4.0                       # [m/s²]

# -----------------------------------------------------------------------------
# Helper – RMSE in xy space
# -----------------------------------------------------------------------------

def _rmse(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    m = min(len(traj_a), len(traj_b))
    return float(np.sqrt(((traj_a[:m, :2] - traj_b[:m, :2]) ** 2).sum(axis=1).mean()))

# -----------------------------------------------------------------------------
# Micro‑planner with 1‑step look‑ahead
# -----------------------------------------------------------------------------

def _arc_to_cont_cmds(
        veh: VehicleIncremental,
        steer_idx: int,
        arc_idx: int,
        next_steer_idx: int | None = None,
        next_arc_idx: int | None = None,
) -> List[Tuple[float, float]]:
    """Convert the *current* discrete action to **exactly N_fixed continuous steps**.

    The last micro‑step already points a little (20 %) towards the *next* target
    if such information is provided. That anticipation greatly reduces jerk at
    the macro‑step boundary without violating the 1‑s latency budget.
    """
    dt = veh.dt
    N_fixed = max(1, int(round(1.0 / dt)))          # 5 for dt = 0.2 s

    # ----- steering targets ----------------------------------------------------
    steer_now = veh.state[4]
    steer_cur = STEER_CHOICES[steer_idx]
    delta_cur = steer_cur - steer_now               # what we must achieve *now*

    delta_next = 0.0
    if next_steer_idx is not None:
        steer_next = STEER_CHOICES[next_steer_idx]
        delta_next = steer_next - steer_cur         # where next action wants to go

    # cosine easing that sums to 1
    cos_prof = 0.5 * (1 - np.cos(np.linspace(0, math.pi, N_fixed, dtype=np.float32)))
    cos_prof /= cos_prof.sum()

    # Anticipation weight increases linearly and tops at 0.2 (= 20 %)
    anticipate = np.linspace(0.0, 0.2, N_fixed, dtype=np.float32)

    steer_steps = delta_cur * cos_prof + anticipate * delta_next / N_fixed
    d_steer_norm = steer_steps / (veh.max_steer_rate * dt)
    d_steer_norm = np.clip(d_steer_norm, -1.0, 1.0)

    # ----- longitudinal profile -----------------------------------------------
    s_target = ARC_CHOICES[arc_idx]
    v0 = veh.state[3]
    N = N_fixed
    a_const = 2 * (s_target - v0 * N * dt) / (dt * dt * N * N)
    a_const = np.clip(a_const, -veh.max_acc, veh.max_acc)
    acc_norm = float(a_const / veh.max_acc)
    acc_profile = np.full(N_fixed, acc_norm, dtype=np.float32)

    # If next arc changes sign, taper acceleration in the last step (50 %)
    if next_arc_idx is not None:
        s_next = ARC_CHOICES[next_arc_idx]
        if s_next * s_target < 0:            # direction flip ahead
            acc_profile[-1] *= 0.5

    return list(zip(d_steer_norm.tolist(), acc_profile.tolist()))

# -----------------------------------------------------------------------------
# Simulation wrapper – now passes look‑ahead information
# -----------------------------------------------------------------------------

def replay_action_sequence(
        actions: Sequence[Tuple[int, int]],
        start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt: float = DEFAULT_DT,
        *,
        verbose: bool = False,
):
    base_kwargs = dict(
        wheelbase=WHEEL_BASE,
        width=2.0,
        front_hang=1.0,
        rear_hang=1.0,
        max_steer=math.radians(MAX_STEER_DEG),
        max_speed=MAX_SPEED,
        dt=dt,
    )

    veh_arc = VehicleArc(**base_kwargs)
    veh_inc = VehicleIncremental(max_acc=MAX_ACC,
                                 max_steer_rate=math.radians(MAX_STEER_RATE_DEG),
                                 **base_kwargs)

    veh_arc.reset_state(*start_pose)
    veh_inc.reset_state(*start_pose)

    traj_arc = [veh_arc.get_pose_center()]
    traj_cont = [veh_inc.get_pose_center()]

    n_macro = len(actions)
    for k, (s_idx, a_idx) in enumerate(actions):
        s_idx_next, a_idx_next = actions[k + 1] if k + 1 < n_macro else (None, None)

        # --- macro step --------------------------------------------------------
        veh_arc.step((s_idx, a_idx))
        x_a, y_a, yaw_a = veh_arc.get_pose_center()
        traj_arc.append((x_a, y_a, yaw_a))
        if verbose:
            print(f"[ARC ] step {k:03d} act=({s_idx},{a_idx})"
                  f" pose=({x_a:+7.2f},{y_a:+7.2f},{math.degrees(yaw_a):+5.1f}°)")

        # --- micro steps -------------------------------------------------------
        cmds = _arc_to_cont_cmds(veh_inc, s_idx, a_idx, s_idx_next, a_idx_next)
        for j, (d_steer, acc) in enumerate(cmds):
            veh_inc.step((d_steer, acc))
            x_c, y_c, yaw_c = veh_inc.get_pose_center()
            traj_cont.append((x_c, y_c, yaw_c))
            if verbose:
                print(f"    [CONT] {j} d_steer={d_steer:+5.2f} acc={acc:+5.2f}"
                      f" pos=({x_c:+7.2f},{y_c:+7.2f},{math.degrees(yaw_c):+5.1f}°)")

    return np.asarray(traj_arc), np.asarray(traj_cont)

# -----------------------------------------------------------------------------
# Plotting helper – unchanged from previous version
# -----------------------------------------------------------------------------

def plot_two_trajs(
        traj_arc: np.ndarray,
        traj_cont: np.ndarray,
        *,
        veh_length: float = 4.0,
        veh_width: float = 2.0,
        skip: int = 5,
        title: str = "",
):
    rmse = _rmse(traj_arc, traj_cont)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(traj_arc[:, 0], traj_arc[:, 1], "-o", label="Discrete arc", lw=1)
    ax.plot(traj_cont[:, 0], traj_cont[:, 1], "-*", label="Continuous inc.", lw=2)

    have_shapely = False
    try:
        from shapely.geometry import Polygon  # type: ignore
        from matplotlib.patches import Polygon as MplPoly
        have_shapely = True
    except ModuleNotFoundError:
        pass

    for k in range(0, len(traj_cont), skip):
        x, y, yaw = traj_cont[k]
        dx = 0.6 * veh_length * math.cos(yaw)
        dy = 0.6 * veh_length * math.sin(yaw)
        ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.25,
                 fc="tab:red", ec="tab:red", alpha=0.8, length_includes_head=True)

        if have_shapely:
            half_w = veh_width * 0.5
            l_f = veh_length * 0.5
            l_r = -veh_length * 0.5
            corners = np.array([[l_r, -half_w],
                                 [l_r,  half_w],
                                 [l_f,  half_w],
                                 [l_f, -half_w]], dtype=np.float32)
            c, s = math.cos(yaw), math.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            world = corners @ R.T + np.array([x, y])
            from matplotlib.patches import Polygon as MplPoly
            ax.add_patch(MplPoly(world, closed=True, facecolor="tab:orange", alpha=0.25,
                                 edgecolor="tab:orange", linewidth=0.7))

    ax.set_aspect("equal", "box")
    ax.set_title(f"{title} | RMSE={rmse:.3f} m")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show(block=True)

# -----------------------------------------------------------------------------
# Demo driver (unchanged CLI, just uses new logic)
# -----------------------------------------------------------------------------

def _demo(n_steps: int, dt: float, verbose: bool):
    rng = random.Random(0)
    actions = [(rng.randrange(len(STEER_CHOICES)), rng.randrange(len(ARC_CHOICES)))
               for _ in range(n_steps)]
    ta, tc = replay_action_sequence(actions, dt=dt, verbose=verbose)
    plot_two_trajs(ta, tc, title=f"Random {n_steps} steps (dt={dt}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discrete arc → continuous demo (look‑ahead)")
    parser.add_argument("--random", type=int, default=40,
                        help="generate N random discrete steps")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT,
                        help="integration time step [s]")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="print per‑step debug info")
    args = parser.parse_args()

    _demo(args.random, args.dt, args.verbose)
