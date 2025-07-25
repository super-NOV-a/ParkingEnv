from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from parking_env_pkg import ParkingEnv
from vehicles.vehicle_arc import VehicleArc
from custom_policy_model import RadarConvFusion, CustomMLP


def make_env_fn(**cfg):
    def _init():
        return Monitor(ParkingEnv(cfg))
    return _init


def parse_args():
    p = argparse.ArgumentParser(description="Test trained PPO model on ParkingEnv")
    p.add_argument("--model", type=Path, required=True, help=".zip model file")
    p.add_argument("--norm_path", type=Path, help="Path to VecNormalize stats")
    p.add_argument("--vecnorm", action="store_true", help="Enable VecNormalize")
    p.add_argument("--play", action="store_true", help="Render one episode")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--vehicle_type", default="arc")
    p.add_argument("--timestep", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 环境配置（推荐与训练保持一致）
    parking_cfg = dict(
        vehicle_type=args.vehicle_type,
        timestep=args.timestep,
        max_steps=args.max_steps,
        render_mode=None,
        scenario_mode="file",  # "random_box" / "box" / "file"
        data_dir="./Train_data_energy/pygame_input_features_new_withinBEV_no_parallel_parking", #"./Train_data_energy/hard",
        lidar_max_range=30.0,
        world_size=30.0,
        difficulty_level=10,

        gap_base=4, gap_step=0.2, gap_min=2,
        occupy_prob_base=0, occupy_prob_step=0.05, occupy_prob_max=0.5,
        wall_thickness=0.1,
        model_ckpt=args.model,
        model_type="mlp",   # 可选 "RadarConvFusion"
    )

    vec_env = DummyVecEnv([make_env_fn(**parking_cfg)])

    # VecNormalize 加载
    if args.vecnorm:
        if not args.norm_path or not args.norm_path.exists():
            raise FileNotFoundError("Missing VecNormalize stats.")
        vec_env = VecNormalize.load(args.norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    from Train_easy import POLICY_REGISTRY
    model_type = parking_cfg["model_type"]
    algo_cls, _, _ = POLICY_REGISTRY[model_type]
    # ── load/resume helper --------------------------------------------------
    def load_model(path: Path):
        return algo_cls.load(path, env=vec_env, device="cpu")
    # model: PPO = PPO.load(args.model, env=vec_env, policy_kwargs=policy_kwargs)
    model = load_model(args.model)

    # 动作分布统计（适用于离散 arc）
    n_steer, n_arc = VehicleArc.N_STEER, VehicleArc.N_ARC
    action_hist = np.zeros((n_steer, n_arc), dtype=int)
    success_count = 0

    for ep in range(args.episodes):
        obs = vec_env.reset()
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            s_idx, a_idx = int(act[0][0]), int(act[0][1])
            action_hist[s_idx, a_idx] += 1

            step_out = vec_env.step(act)
            if len(step_out) == 5:
                obs, _, term, trunc, info = step_out
                done = bool(term[0] or trunc[0])
            else:
                obs, _, term, info = step_out
                done = term[0]

        if not info[0].get("collision", False):
            success_count += 1

    total = action_hist.sum()
    print("\n==== 动作分布统计 ====")
    for i in range(n_steer):
        print(f"Steer {i:2d}: {action_hist[i].sum():6d} 次")
    print(f"\n✅ 成功: {success_count}/{args.episodes}    ✔️ 成功率: {success_count/args.episodes:.2%}")

    # 可视化（可选）
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(action_hist, cmap='hot', interpolation='nearest')
    plt.title("Action Frequency Heatmap")
    plt.xlabel("Arc Index"); plt.ylabel("Steer Index")
    plt.colorbar()
    plt.show()

    # 交互渲染
    if args.play:
        env = ParkingEnv({**parking_cfg, "render_mode": "human"})
        while True:
            obs, _ = env.reset()
            done = False
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, _, done, trunc, info = env.step(act)
                env.render()
                if done or trunc:
                    print("Success" if not info.get("collision", False) else "Failure")
                    break
        env.close()
