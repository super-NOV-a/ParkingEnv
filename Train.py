"""Train.py – Universal PPO trainer for ParkingEnv with selectable vehicle type
============================================================================
This refactor makes **one script** capable of training **any** of the three
vehicle models supported by *parking_env_flex.ParkingEnvFlex*:

* `continuous`   – continuous steer × acceleration (`VehicleContinuous`)
* `disc_accel`   – 9‑action steer × target‑speed grid (`VehicleDiscAccel`)
* `arc`          – 15 × 4 discrete steer‑notch × arc‑length (`VehicleArc`)

Switch vehicle via CLI:
```bash
python Train.py --vehicle-type arc         # default (discrete‑arc)
python Train.py --vehicle-type continuous  # original continuous
python Train.py --vehicle-type disc_accel  # 9‑discrete accel
```
The rest (PPO hyper‑parameters, VecNormalize, checkpoints, evaluation) remains
unchanged.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from parking_env_pkg import ParkingEnv  # unified env with vehicle_type param
from custom_policy_model import RadarConvFusion, CustomMLP

###############################################################################
# Helpers
###############################################################################

def make_env_fn(vehicle_type: str, **parking_cfg: Dict):
    """Factory producing *Monitor*‑wrapped envs for vectorisation."""

    def _init() -> gym.Env:
        env = ParkingEnv({**parking_cfg, "vehicle_type": vehicle_type})
        return Monitor(env)

    return _init

###############################################################################
# CLI
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on flexible ParkingEnv")

    # –– environment choice ––
    p.add_argument(
        "--vehicle_type",
        choices=["continuous", "disc_accel", "arc"],
        default="arc",
        help="which vehicle dynamics to train",
    )

    # –– SB3 hyper‑parameters ––
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=2048)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)

    # –– logging / checkpoints ––
    p.add_argument("--logdir", type=Path, default=None, help="root log folder")
    p.add_argument("--save_freq", type=int, default=100_000)

    # –– parking‑env generic config ––
    p.add_argument("--timestep", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--render_mode", choices=["none", "rgb_array"], default="none")

    # –– misc ––
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", action="store_true", help="force CUDA (if available)")
    p.add_argument("--vecnorm", action="store_true", help="enable VecNormalize")
    p.add_argument("--resume", action="store_true", help="resume if logdir exists")

    # –– evaluation ––
    p.add_argument("--eval_freq", type=int, default=50_000)
    p.add_argument("--eval_episodes", type=int, default=20)

    return p.parse_args()

###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)

    # ── Parking‑env configuration (common) ────────────────────────────
    parking_cfg = dict(
        timestep=args.timestep,
        max_steps=args.max_steps,
        render_mode=None if args.render_mode == "none" else args.render_mode,
        scenario_mode="random",
        data_dir=".\pygame_input_features_new_withinBEV_no_parallel_parking",   # ← 指向你的 .json 文件夹
        max_speed=3.0,          # 在离散轨迹中用于表示映射到[-1,1]的轨迹长度
        lidar_max_range=15.0,

        min_obstacles=0,         # useless
        max_obstacles=10,
        row_count_range=(1, 4), # \useless 上述参数在genarate_random_legacy中使用了，目前是无用参数

        world_size=30.0,
        occupy_prob=0.,          # 初级课程
        gap=4.0,
        wall_thickness=0.1,

        # 训练模型管理项 ↓↓↓
        logdir="cnn", # 可以此处指定log_dir！

        # model_ckpt = "\\runs\ppo_arc_empty_fc\\best_model\\best_model.zip",
        model_ckpt=None,  # 如需加载模型路径填绝对路径或 pathlib.Path 对象 运行时使用： python Train.py --resume

        # 自定义模型类型，见--custom_policy_model.py
        policy_class="RadarConvFusion",  # 可选 "CustomMLP" 或 "RadarConvFusion"
        
    )
    # 训练时保存模型和log的位置可以在 parking_cfg 中指定 log_dir，
    # 也可以在命令行指定：python Train.py --logdir="runs/your_log_path"
    EVAL = False    # 先不进行评估，因为评估环境于训练环境的难度不一致，没有参考价值
    
    # ────────────────────── dynamic default logdir if not provided
    if args.logdir is None:
        logdir = parking_cfg.get("logdir", None)    
        if logdir:              # 若命令行没有给出，但是parking_cfg中配置了
            scenario_mode = parking_cfg.get("scenario_mode", None)    
            args.logdir = Path(f"runs/ppo_{args.vehicle_type}_{scenario_mode}_{logdir}")
        else:                   # 都没给出log_dir则按时间生成路径
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.logdir = Path(f"runs/ppo_{args.vehicle_type}_{timestamp}")
    
    tb_dir = args.logdir / "tb"
    ckpt_dir = args.logdir / "checkpoints"
    best_dir = args.logdir / "best_model"
    norm_path = args.logdir / "vec_normalize.pkl"
    model_dir = args.logdir / "models"

    for d in [tb_dir, ckpt_dir, best_dir, model_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Vectorised training env ───────────────────────────────────────
    env = make_vec_env(
        make_env_fn(args.vehicle_type, **parking_cfg),
        n_envs=args.n_envs,
        seed=args.seed,
    )

    # Optional VecNormalize -------------------------------------------
    if args.vecnorm:
        if args.resume and norm_path.exists():
            env = VecNormalize.load(norm_path, env)
            env.training = True
            print("[VecNormalize] Resumed stats from", norm_path)
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            print("[VecNormalize] Enabled")

    device = "cuda" if (args.cuda and os.environ.get("CUDA_VISIBLE_DEVICES", "")) else "auto"

    # ── Policy kwargs ─────────────────────────────────────────────────
    policy_class_str = parking_cfg.get("policy_class", "CustomMLP") # cfg中指定模型类型
    policy_class = globals()[policy_class_str]  # 或自定义映射表 if 安全性考虑

    policy_kwargs = dict(
        features_extractor_class=policy_class,
        features_extractor_kwargs=dict(features_dim=128)
    )

    # ── Model (resume or fresh) ────────────────────────────────────────────────────
    model_ckpt = parking_cfg.get("model_ckpt", None)
    if args.resume and model_ckpt:
        print("──────────────────────────────────────────────")
        print(f"[Resume] Loading model from {model_ckpt}")
        print("──────────────────────────────────────────────")
        model = PPO.load(model_ckpt, env=env, device=device, policy_kwargs=policy_kwargs)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=0.0,
            vf_coef=0.5,
            tensorboard_log=str(tb_dir),
            verbose=1,
            device=device,
            seed=args.seed,
        )

    # ── Callbacks ────────────────────────────────────────────────────
    ckpt_cb = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path=str(ckpt_dir),
        name_prefix=f"ppo_{args.vehicle_type}",
        save_replay_buffer=False,
        save_vecnormalize=args.vecnorm,
    )

    if EVAL:
        eval_env = make_vec_env(make_env_fn(args.vehicle_type, **parking_cfg), n_envs=1, seed=args.seed + 1)
        if args.vecnorm:
            eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
            if norm_path.exists():
                eval_env.load_running_average(norm_path)

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(args.logdir / "eval"),
            eval_freq=args.eval_freq // max(args.n_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
        )

    pbar_cb = ProgressBarCallback()

    # ── Train ────────────────────────────────────────────────────────
    if EVAL:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[ckpt_cb, eval_cb, pbar_cb],
            progress_bar=False,
        )
    else:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[ckpt_cb, pbar_cb],
            progress_bar=False,
        )

    # ── Save artifacts ───────────────────────────────────────────────
    model.save(model_dir / "final")
    if args.vecnorm:
        env.save(norm_path)
    env.close()
    print("saving final model to log:", args.logdir)

    if EVAL:
        # ── Final evaluation ─────────────────────────────────────────────
        eval_env.close()
        eval_env = make_vec_env(make_env_fn(args.vehicle_type, **parking_cfg), n_envs=1, seed=args.seed + 99)
        if args.vecnorm and norm_path.exists():
            eval_env = VecNormalize.load(norm_path, eval_env)
            eval_env.training = False

        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
        print(f"\n>>> Evaluation over {args.eval_episodes} episodes: {mean_r:.2f} ± {std_r:.2f}\n")
        eval_env.close()
