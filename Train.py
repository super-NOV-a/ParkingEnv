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
from datetime import datetime
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
def unique_logdir(base: Path, resume: bool) -> Path:
    """
    If resume=False and base already exists, append _1/_2/... until unused.
    Keeps base unchanged when resume=True.
    """
    if resume or not base.exists():
        return base
    idx = 1
    while True:
        cand = Path(f"{base}_{idx}")
        if not cand.exists():
            return cand
        idx += 1

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

    # –– resume / load ––
    p.add_argument("--resume", action="store_true", help="continue writing into logdir")
    p.add_argument("--load_model", type=Path, default=None,
                   help="path to a pre-trained .zip to initialise weights (does not imply --resume)")


    # –– evaluation ––
    p.add_argument("--eval_freq", type=int, default=50_000)
    p.add_argument("--eval_episodes", type=int, default=20)

    return p.parse_args()

###############################################################################
# Train.py – 日志目录 & 预训练权重 速查表
#
# 关键词             作用                         说明
# ---------------------------------------------------------------------------
# --logdir PATH      指定本次训练写入的目录       不含 runs/ 前缀也可
# --resume           继续写入同一个 logdir        自动寻找 checkpoints 下最新 .zip
# --load_model PATH  先加载指定 .zip 再训练        不会影响 logdir 的唯一化选择
#
# parking_cfg["log_dir"]   若 CLI 未指定 --logdir，则使用它
# parking_cfg["model_ckpt"]若 CLI 未指定 --load_model，则使用它
#
# 唯一化策略: 当 resume=False 且目标目录已存在时，在末尾自动追加 _1/_2/… 避免覆盖
# 目录格式: runs/ppo_<vehicle_type>_<logdir_stub>[_k]  (k 为自动编号)
#
# 优先级: CLI 明确参数 > parking_cfg 设置 > 默认 (时间戳)
#
# ─────────── 常用调用范例 ─────────────────────────────────────────────
#
# 1. 全新训练 – 自动生成时间戳目录
#    $ python Train.py --vehicle_type arc
#    → runs/ppo_arc_20250711_161233/
#
# 2. 全新训练 – 手动指定 logdir（若已存在自动 _1/_2）--------------------------     <-
#    $ python Train.py --vehicle_type arc --logdir random_cnn
#    → runs/ppo_arc_random_cnn/  (或 random_cnn_1 …)
#
# 3. 继续在同目录训练（恢复最新 checkpoint）
#    $ python Train.py --vehicle_type arc --logdir random_cnn --resume
#
# 4. 加载外部权重，但写入新目录------------------------------------------------     <-
#    $ python Train.py --vehicle_type arc \\
#                      --logdir random_cnn \\
#                      --load_model "runs/ppo_arc_empty_cnn/checkpoints/ppo_arc_1e6_steps.zip"
#
# 5. 仅靠配置文件驱动
#    parking_cfg = { "log_dir": "random_cnn", 
#                    "model_ckpt": "runs/ppo_arc_empty_cnn/..." }
#    $ python Train.py
#
# 6. 覆盖配置文件里的 model_ckpt
#    $ python Train.py --load_model my_pretrain.zip
#
# 7. 覆盖配置文件里的 log_dir 并接着写
#    $ python Train.py --logdir random_cnn --resume
#
# 8. 不想 resume，只想让脚本自己起 _1/_2
#    $ python Train.py --logdir random_cnn
#
# (所有示例中若省略 --logdir，则由 parking_cfg["log_dir"] 或时间戳替代)
# ────────────────────────────────────────────────────────────────────

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

        # 这里写好要导入的模型，然后在命令行继续训练：python Train.py --resume
        # model_ckpt = "\\runs\ppo_arc_empty_fc\\best_model\\best_model.zip","runs\ppo_arc_empty_cnn\checkpoints\ppo_arc_1000000_steps.zip"
        model_ckpt=".\\runs\ppo_arc_empty_cnn\checkpoints\ppo_arc_1000000_steps.zip",  # 如需加载模型路径填绝对路径或 pathlib.Path 对象

        # 自定义模型类型，见--custom_policy_model.py
        policy_class="RadarConvFusion",  # 可选 "CustomMLP" 或 "RadarConvFusion"
        
    )
    # 训练时保存模型和log的位置可以在 parking_cfg 中指定 log_dir，
    # 在命令行指定：python Train.py --logdir="runs/your_log_path"

    # ── auto-propagate model_ckpt from cfg if CLI didn’t specify ─────
    if args.load_model is None and parking_cfg.get("model_ckpt"):
        args.load_model = Path(parking_cfg["model_ckpt"])
        print(f"[Init]  Auto-loading weights from parking_cfg: {args.load_model}")

    EVAL = False    # 先不进行评估，因为评估环境于训练环境的难度不一致，没有参考价值
    
    # ────────────────────── dynamic default logdir if not provided
    if args.logdir is None:
        logdir_stub = parking_cfg.get("logdir") or datetime.now().strftime("%Y%m%d_%H%M%S")
        scene = parking_cfg.get("scenario_mode")
        args.logdir = Path(f"runs/ppo_{args.vehicle_type}_{scene}_{logdir_stub}")

    # ── ensure uniqueness unless --resume ───────────────────────────
    args.logdir = unique_logdir(args.logdir, args.resume)
    print(f"[Logdir] Writing to {args.logdir}")
    
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

    # logic priority: 1) --resume → reload latest ckpt  2) --load_model → warm-start
    #                 3) fresh init
    latest_ckpt = max((args.logdir / "checkpoints").glob("*.zip"),
                      key=lambda p: int(p.stem.split("_")[-2]) if "_" in p.stem else 0,
                      default=None)     # 路径下最新的

    if args.resume and latest_ckpt is not None:
        print(f"[Resume] Continue training from {latest_ckpt}")
        model = PPO.load(latest_ckpt, env=env, device=device, policy_kwargs=policy_kwargs)
    elif args.load_model is not None:
        print(f"[Init]  Load weights from {args.load_model}")
        model = PPO.load(args.load_model, env=env, device=device, policy_kwargs=policy_kwargs)
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
    step_tag = f"{model.num_timesteps:_}".replace(",", "")
    model.save(model_dir / f"final_{step_tag}")
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
