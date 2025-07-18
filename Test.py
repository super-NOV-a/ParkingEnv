from __future__ import annotations

import argparse
from pathlib import Path
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from parking_env_pkg import ParkingEnv
from custom_policy_model import RadarConvFusion, CustomMLP
import numpy as np
from vehicles.vehicle_arc import VehicleArc     # 用于获取离散尺寸


def make_env_fn(**cfg):
    def _init():
        env = ParkingEnv({**cfg})
        return Monitor(env)
    return _init


def parse_args():
    p = argparse.ArgumentParser(description="Test PPO model on ParkingEnv")
    p.add_argument("--model", type=Path, required=True, help=".zip model file")
    p.add_argument("--norm_path", type=Path, default=None, help="VecNormalize stats")
    p.add_argument("--vecnorm", action="store_true", help="enable VecNormalize")
    p.add_argument("--render", choices=["none", "human", "rgb_array"], default="none")
    p.add_argument("--play", action="store_true", help="run & render one episode")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--vehicle_type", default="arc")
    p.add_argument("--timestep", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=500)
    return p.parse_args()

## -----------------------------------------------------------------------------------------------
## 纯评估模式
# python Test.py --model "runs/your_model/final.zip"

## 评估完再进行渲染查看效果
# python Test.py --model "runs/your_model/final.zip" --play
## -----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    parking_cfg = dict(
        timestep=args.timestep,
        max_steps=args.max_steps,
        vehicle_type="arc",
        render_mode=None,
        scenario_mode="random_box",     # file random  empty  box  random_box
        data_dir="./Train_data_energy/pygame_input_features_new_withinBEV_no_parallel_parking",
        lidar_max_range=30.0,
        world_size=40.0,
        difficulty_level=9,    # 环境难度，对于停车位容忍程度和障碍密度有影响

        gap_base = 4,
        gap_step = 0.2,  # 总共十个level
        gap_min = 2,
        occupy_prob_base = 0,
        occupy_prob_step = 0.05,
        occupy_prob_max = 0.5,

        wall_thickness=0.1,
        model_ckpt=args.model,
        policy_class="CustomMLP",     # 可选 "CustomMLP" 或 "RadarConvFusion"
    )

    # VecEnv wrapper
    vec_env = DummyVecEnv([make_env_fn(**parking_cfg)])

    if args.vecnorm:
        if not args.norm_path or not args.norm_path.exists():
            raise FileNotFoundError("Missing vecnorm stats.")
        vec_env = VecNormalize.load(args.norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Choose policy class
    policy_class_str = parking_cfg.get("policy_class", "CustomMLP")
    policy_class = globals().get(policy_class_str, CustomMLP)
    policy_kwargs = dict(
        features_extractor_class=policy_class,
        features_extractor_kwargs=dict(features_dim=128)
    )

    model: PPO = PPO.load(args.model, vec_env, policy_kwargs=policy_kwargs)

    # ───────────────────────── 自定义评估循环 ─────────────────────────
    n_steer, n_arc = VehicleArc.N_STEER, VehicleArc.N_ARC
    action_hist = np.zeros((n_steer, n_arc), dtype=np.int64)

    successes = 0
    for ep in range(args.episodes):
        reset_out = vec_env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out   # 1-或 2-tuple 兼容
        done = False

        while not done:
            act, _ = model.predict(obs, deterministic=True)   # shape = (1,2)
            s_idx, a_idx = map(int, act[0])
            action_hist[s_idx, a_idx] += 1

            step_out = vec_env.step(act)
            if len(step_out) == 5:           # Gymnasium 原生 → 5 项
                obs, reward, dones, truncs, infos = step_out
                done = bool(dones[0] or truncs[0])
            else:                            # DummyVecEnv 简化 → 4 项
                obs, reward, dones, infos = step_out
                done = bool(dones[0])

            info = infos[0]

        # “成功” → episode 终止且没有碰撞
        if not info.get("collision", False):
            successes += 1

    success_rate = successes / args.episodes
    print(f"\n✅ 成功: {successes}/{args.episodes}   ✔️ 成功率: {success_rate:.2%}")

    # ── 打印动作分布 ─────────────────────────────────────────────────────────
    total_steps = action_hist.sum()
    steer_counts = action_hist.sum(axis=1)
    arc_counts   = action_hist.sum(axis=0)

    print("\n── 动作分布统计 ──")
    print("Steer-idx  次数    占比")
    for i, c in enumerate(steer_counts):
        print(f"{i:9d}  {c:6d}  {c/total_steps:6.2%}")

    print("\nArc-idx    次数    占比")
    for j, c in enumerate(arc_counts):
        print(f"{j:9d}  {c:6d}  {c/total_steps:6.2%}")

    import matplotlib.pyplot as plt

    # … 成功率打印后 ↓↓↓
    fig1 = plt.figure()
    plt.bar(range(len(steer_counts)), steer_counts)
    plt.xlabel("Steer-idx");  plt.ylabel("Count")
    plt.title("Distribution of Steer Indices")
    plt.tight_layout()

    fig2 = plt.figure()
    plt.bar(range(len(arc_counts)), arc_counts)
    plt.xlabel("Arc-idx");    plt.ylabel("Count")
    plt.title("Distribution of Arc Indices")
    plt.tight_layout()
    plt.show() # 交互模式直接弹窗


    # Optional playback
    if args.play:
        env = ParkingEnv({**parking_cfg, "render_mode": "human"})

        while True:
            obs, _ = env.reset()
            done = truncated = coll= False
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = env.step(action)
                
                collision = info.get("collision", False)
                if done or truncated:
                    if done and not collision:
                        print("success")
                    else:
                        print("failure")
                    env.reset()
                env.render()
        # env.close()
