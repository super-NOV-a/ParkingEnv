#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test_arc_cont_eval.py
=====================
A **drop‑in replacement** for `Test.py` that, in addition to the usual PPO
policy evaluation, **replays every discrete episode with the continuous
vehicle model and plots the two centre‑line trajectories side‑by‑side** so
that you can visually compare precision.

Main additions
--------------
* **Action trace recording** – store every `(steer_idx, arc_idx)` tuple per
  episode.
* **arc→cont conversion** – use `arc_to_cont_replay.replay_sequence()` to
  obtain the continuous track.
* **Trajectory plot** – call `arc_to_cont_replay.plot_two_tracks()` at the
  end of each episode (optional `--no_plot`).

Vehicle params follow the latest spec (wheel‑base 3 m, L×W 5 × 2 m, front &
rear hang = 1 m, `dt=0.1 s`, `N_MICRO=5`).
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt

from parking_env_pkg import ParkingEnv
from custom_policy_model import RadarConvFusion, CustomMLP
from vehicles.vehicle_arc import VehicleArc

from demo_arc_to_continuous import replay_action_sequence, plot_two_trajs

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PPO model + compare arc vs continuous tracks")
    p.add_argument("--model", type=Path, default="C:\AI_Planner\Env2\\runs\ppo_arc_box_fc7\checkpoints\ppo_arc_4000000_steps.zip", help=".zip model file")
    p.add_argument("--norm_path", type=Path, help="VecNormalize stats")
    p.add_argument("--vecnorm", action="store_true", help="enable VecNormalize")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--timestep", type=float, default=0.1)
    p.add_argument("--no_plot", action="store_true", help="skip trajectory plots (headless)")
    p.add_argument("--save_fig", type=Path, help="folder to save figures instead of plt.show()")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def make_env_fn(**cfg):
    def _init():
        env = ParkingEnv(cfg)
        return Monitor(env)
    return _init


def safe_vehicle_pose(env) -> Tuple[float, float, float]:
    """Best‑effort fetch of the ego vehicle (x,y,heading) pose."""
    # try common attributes; fall back to zeros
    if hasattr(env, "vehicle") and hasattr(env.vehicle, "state"):
        x, y = env.vehicle.state[:2]
        heading = env.vehicle.state[2] if hasattr(env.vehicle, "state") else 0.0
        return float(x), float(y), float(heading)
    # return 0.0, 0.0, 0.0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    parking_cfg = dict(
        timestep=args.timestep,
        max_steps=args.max_steps,
        vehicle_type="arc",      # PPO policy operates in discrete‑arc space
        render_mode="human",
        scenario_mode="box",
        data_dir="./Train_data_energy/pygame_input_features_new_withinBEV_no_parallel_parking",
        lidar_max_range=30.0,
        world_size=40.0,
        difficulty_level=9,
        # scene generation params (gap / occupy prob)
        gap_base=4, gap_step=0.2, gap_min=2,
        occupy_prob_base=0, occupy_prob_step=0.05, occupy_prob_max=0.5,
        wall_thickness=0.1,
        model_ckpt=args.model,
        policy_class="CustomMLP",
    )

    vec_env = DummyVecEnv([make_env_fn(**parking_cfg)])
    if args.vecnorm:
        if not (args.norm_path and args.norm_path.exists()):
            raise FileNotFoundError("Missing vecnorm stats")
        vec_env = VecNormalize.load(args.norm_path, vec_env)
        vec_env.training, vec_env.norm_reward = False, False

    # choose extractor class
    policy_class_str = parking_cfg.get("policy_class", "CustomMLP")
    policy_class = globals().get(policy_class_str, CustomMLP)
    policy_kwargs = dict(features_extractor_class=policy_class,
                         features_extractor_kwargs=dict(features_dim=128))

    model: PPO = PPO.load(args.model, vec_env, policy_kwargs=policy_kwargs)

    n_steer, n_arc = VehicleArc.N_STEER, VehicleArc.N_ARC
    action_hist = np.zeros((n_steer, n_arc), dtype=np.int64)

    successes = 0
    for ep in range(args.episodes):
        reset_out = vec_env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out   # 1-或 2-tuple 兼容
        done = False
        ep_actions: List[Tuple[int, int]] = []

        start_pose = safe_vehicle_pose(vec_env.envs[0].env)  # unwrap Monitor→ParkingEnv

        while not done:
            action, _ = model.predict(obs, deterministic=True)  # [[s_idx, a_idx]]
            s_idx, a_idx = map(int, action[0])
            ep_actions.append((s_idx, a_idx))
            action_hist[s_idx, a_idx] += 1

            step_out = vec_env.step(action)
            if len(step_out) == 5:           # Gymnasium 原生 → 5 项
                obs, reward, dones, truncs, info = step_out
                done = bool(dones[0] or truncs[0])
            else:                            # DummyVecEnv 简化 → 4 项
                obs, reward, dones, info = step_out
                done = bool(dones[0])


        collision = info[0].get("collision", False)
        if not collision:
            successes += 1

        # ── trajectory replay & plot ───────────────────────────────────────
        traj_arc, traj_cont = replay_action_sequence(ep_actions, start_pose)
        if not args.no_plot:
            title = f"Episode {ep}: arc vs cont (RMSE shown)"
            fig = plot_two_trajs(traj_arc, traj_cont, title=title)
            if args.save_fig:
                args.save_fig.mkdir(parents=True, exist_ok=True)
                fig.savefig(args.save_fig / f"ep{ep:03d}.png", dpi=150)
                plt.close(fig)
            else:
                plt.show(block=False)

    success_rate = successes / args.episodes
    print(f"\n✅ 成功: {successes}/{args.episodes}   ✔️ 成功率: {success_rate:.2%}")

    # ── action distribution summary ───────────────────────────────────────
    total_steps = action_hist.sum()
    steer_counts, arc_counts = action_hist.sum(axis=1), action_hist.sum(axis=0)
    print("\n── 动作分布统计 ──")
    for i, c in enumerate(steer_counts):
        print(f"Steer {i:2d}: {c:6d}  ({c/total_steps:.2%})")
    for j, c in enumerate(arc_counts):
        print(f"Arc   {j:2d}: {c:6d}  ({c/total_steps:.2%})")
