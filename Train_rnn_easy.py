###############################################################################
# Train_easy.py — now supports 4 model types via parking_cfg["model_type"]
#   model_type ∈ { mlp, mlp_rnn, conv, conv_rnn }
# ---------------------------------------------------------------------------
# * mlp        : CustomMLP + PPO
# * mlp_rnn    : CustomMLP + LSTM (RecurrentPPO)
# * conv       : RadarConvFusion + PPO
# * conv_rnn   : RadarConvExtractor + LSTM (RecurrentPPO)
###############################################################################
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    # EvalCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.logger import configure

from parking_env_pkg import ParkingEnv
from custom_policy_model import POLICY_REGISTRY, make_model  # ← unified helper

###############################################################################
# Callbacks (unchanged)
###############################################################################
class SuccessRateCallback(BaseCallback):
    def __init__(self, verbose: int = 0, max_envs_logged: int = 4):
        super().__init__(verbose)
        self.max_envs_logged = max_envs_logged

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        total_successes = total_trials = 0
        levels: List[int] = []
        for idx, env in enumerate(self.training_env.envs):  # type: ignore[attr-defined]
            raw = env
            while hasattr(raw, "env"):
                raw = raw.env
            if hasattr(raw, "_success_history"):
                hist = raw._success_history
                total_successes += sum(hist)
                total_trials += len(hist)
            if hasattr(raw, "level") and idx < self.max_envs_logged:
                levels.append(raw.level)
        self.logger.record("rollout/success_rate", total_successes / total_trials if total_trials else 0)
        self.logger.record("rollout/mean_level", sum(levels) / len(levels) if levels else 0)

###############################################################################
# Utilities
###############################################################################

def unique_logdir(base: Path, resume: bool) -> Path:
    if resume or not base.exists():
        return base
    idx = 1
    while True:
        cand = Path(f"{base}_{idx}")
        if not cand.exists():
            return cand
        idx += 1


def make_env_fn(**parking_cfg: Dict):
    def _init() -> gym.Env:  # type: ignore[valid-type]
        return Monitor(ParkingEnv({**parking_cfg}))
    return _init

###############################################################################
# CLI (unchanged except default n_steps=128)
###############################################################################

def parse_args():
    p = argparse.ArgumentParser("Train PPO/RecurrentPPO on ParkingEnv")
    # SB3 params
    p.add_argument("--total_timesteps", type=int, default=20_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=128)  # shorter default good for RNN
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)

    # logging
    p.add_argument("--logdir", type=Path, default=None)
    p.add_argument("--save_freq", type=int, default=100_000)

    # env generic
    p.add_argument("--timestep", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--render_mode", choices=["none", "rgb_array"], default="none")

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--vecnorm", action="store_true")

    # resume/load
    p.add_argument("--resume", action="store_true")
    p.add_argument("--load_model", type=Path)

    # evaluation
    p.add_argument("--eval_freq", type=int, default=50_000)
    p.add_argument("--eval_episodes", type=int, default=20)

    return p.parse_args()

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)

    # ── parking config (specify model_type here) ───────────────────
    parking_cfg = dict(
        model_type="conv_rnn",              # ← change to mlp / mlp_rnn / conv / conv_rnn
        # 这里写好要导入的模型，然后在命令行继续训练：python Train_easy.py 即可继续训练，不需要.zip
        # model_ckpt= ".\\runs\\ppo_arc_parking_fc01\\checkpoints\\ppo_arc_10000000_steps.zip",  # None
        model_ckpt= None,
        vehicle_type="arc",                 # incremental  arc
        timestep=args.timestep,
        max_steps=args.max_steps,
        render_mode=None if args.render_mode == "none" else args.render_mode,  # "human",
        scenario_mode="parking",
        logdir="fc_easy", # 可以此处指定log_dir！ 继续训练时会保存在此处
        max_speed=3.0,          # 在离散轨迹中用于表示映射到[-1,1]的轨迹长度
        lidar_max_range=30.0,
        world_size=30.0,

        # 配置课程，scenario_manager.py中的__post_init__方法提供了默认的课程，但是训练起来较难成长
        # 不同等级之间难度差别大
        difficulty_level=0,
        gap_base = 4,       # 在random中使用的内容
        gap_step = 0.2,     # 总共十个level  gap与occupy_prob根据level在min/max之间调节
        gap_min = 2,
        occupy_prob_base = 0,   # parking中是车位附近的障碍车位概率
        occupy_prob_step = 0.05,
        occupy_prob_max = 0.5,
        wall_thickness=0.1,

        # json file场景的相关参数
        energy=False,    # True, False
        random_file_init = False, # 导入file时ego是否随机初始位置
        data_dir=".\Train_data_energy\pygame_input_features_new_withinBEV_no_parallel_parking",   # ← 指向你的 .json 文件夹
        energy_data_dir=".\Train_data_energy\Energy_train",   # ← 指向你的 .json 文件夹
    )

    args.vehicle_type = parking_cfg.get("vehicle_type")
    model_type = parking_cfg["model_type"]
    algo_cls, _, _ = POLICY_REGISTRY[model_type]
    if args.load_model is None and parking_cfg.get("model_ckpt"):
        args.load_model = Path(parking_cfg["model_ckpt"])
        print(f"[Init]  Auto-loading weights from parking_cfg: {args.load_model}")

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

    # ── env & VecNormalize ─────────────────────────────────────────
    env = make_vec_env(
        make_env_fn(**parking_cfg), n_envs=args.n_envs, seed=args.seed
        )
    if args.vecnorm:
        if args.resume and norm_path.exists():
            env = VecNormalize.load(norm_path, env)
            env.training = True
            print("[VecNormalize] Resumed stats from", norm_path)
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            print("[VecNormalize] Enabled")
    device = "cuda" if (args.cuda and os.environ.get("CUDA_VISIBLE_DEVICES", "")) else "auto"

    # ── load/resume helper --------------------------------------------------
    def smart_load(path: Path, env, device="auto"):
        data, _ = BaseAlgorithm._load(path, "", device=device, print_system_info=False, env=None)
        algo_name = data["algo_name"]  # e.g. 'RecurrentPPO' or 'PPO'
        algo_cls = RecurrentPPO if algo_name == "RecurrentPPO" else PPO
        return algo_cls.load(path, env=env, device=device)

    # then replace load_model helper
    def load_model(path: Path):
        return smart_load(path, env=env, device=device)

    latest_ckpt = max(ckpt_dir.glob("*.zip"), key=os.path.getmtime, default=None) if ckpt_dir.exists() else None

    # 根据是否继续训练，导入模型或新建模型
    if args.resume and latest_ckpt is not None:
        print("[Resume]", latest_ckpt)
        model = load_model(latest_ckpt)
        model.set_logger(configure(str(tb_dir), ["stdout", "tensorboard"]))
    elif args.load_model:
        print("[Init] load", args.load_model)
        model = load_model(args.load_model)
        model.set_logger(configure(str(tb_dir), ["stdout", "tensorboard"]))
    else:
        model = make_model(
            model_type,
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            tensorboard_log=str(tb_dir),
            verbose=1,
            device=device,
            seed=args.seed,
        )
        model.set_logger(configure(str(tb_dir), ["stdout", "tensorboard"]))

    # ── callbacks -----------------------------------------------------------
    ckpt_cb = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path=str(ckpt_dir),
        name_prefix=f"{algo_cls.__name__.lower()}_{model_type}",
        save_vecnormalize=args.vecnorm,
    )
    pbar_cb = ProgressBarCallback(); success_cb = SuccessRateCallback()

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[ckpt_cb, success_cb, pbar_cb],
        progress_bar=False,
    )

    # final save
    final_path = args.logdir / f"final_{model.num_timesteps:_}.zip".replace(",", "")
    model.save(final_path)
    if args.vecnorm:
        env.save(args.logdir / "vec_normalize.pkl")
    env.close()
    print("[Done] saved", final_path)
