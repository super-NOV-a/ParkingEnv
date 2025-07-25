# test_rnn.py – evaluation / playback helper for LSTM‑based RecurrentPPO models
# ----------------------------------------------------------------------------------
# Usage examples
#   ▶ 纯评估（打印成功率 + 动作统计）
#       $ python test_rnn.py --model runs/ppo_arc_rnn_fc00/final_2_000_000_steps.zip --episodes 50
#
#   ▶ 带 VecNormalize 的模型
#       $ python test_rnn.py --model .../final.zip --vecnorm --norm_path .../vec_normalize.pkl
#
#   ▶ 单次可视化回放
#       $ python test_rnn.py --model .../final.zip --play --render human
# ----------------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ⚠️  需要确保 Python 能 import 下方自定义类，否则加载失败
from parking_env_pkg import ParkingEnv
from custom_policy_model import (
    RadarConvExtractor,
    RadarConvFusion,
    CustomMLP,
    RadarLstmPolicy,  # 注册自定义策略类，便于 .zip 反序列化
)

# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate an LSTM RecurrentPPO model on ParkingEnv")
    p.add_argument("--model", type=Path, required=True, help="Path to .zip model file")
    p.add_argument("--norm_path", type=Path, default=None, help="VecNormalize stats (.pkl)")
    p.add_argument("--vecnorm", action="store_true", help="Enable VecNormalize during eval")
    p.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    p.add_argument("--render", choices=["none", "human", "rgb_array"], default="none")
    p.add_argument("--play", action="store_true", help="Render one interactive episode after eval")
    p.add_argument("--device", default="cpu", help="torch device override (cpu/cuda:auto)")
    return p.parse_args()

# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------

def make_env_fn(**cfg):
    """Factory that wraps ParkingEnv in Monitor for VecEnv compatibility."""

    def _init() -> gym.Env:  # type: ignore[valid-type]
        env = ParkingEnv(cfg)
        return Monitor(env)

    return _init

# ----------------------------------------------------------------------------------
# Main evaluation loop
# ----------------------------------------------------------------------------------

def evaluate(model: RecurrentPPO, env: DummyVecEnv, episodes: int = 10) -> None:
    n_envs = env.num_envs
    successes = 0

    # place‑holders for RecurrentPPO predict() arguments
    lstm_states: Tuple[torch.Tensor, ...] | None = None
    episode_starts = np.ones((n_envs,), dtype=bool)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = np.array([False])
        # reset flags for new episode
        lstm_states = None
        episode_starts.fill(True)

        while not done[0]:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            # after first call no envs are at first step
            episode_starts.fill(False)

            obs, _rewards, dones, truncs, infos = env.step(action)
            done = np.logical_or(dones, truncs)
        info = infos[0]
        if not info.get("collision", False):
            successes += 1

    print(f"Evaluation finished → Success {successes}/{episodes}  |  Rate {successes/episodes:.2%}")

# ----------------------------------------------------------------------------------
# Interactive playback (optional)
# ----------------------------------------------------------------------------------

def play_one(model: RecurrentPPO, cfg: dict, render_mode: str) -> None:
    env = ParkingEnv({**cfg, "render_mode": render_mode})
    obs, _ = env.reset()
    lstm_state = None
    episode_start = np.array([True])  # single env

    done = truncated = False
    while not (done or truncated):
        action, lstm_state = model.predict(obs, state=lstm_state, episode_start=episode_start, deterministic=True)
        episode_start[0] = False
        obs, _r, done, truncated, info = env.step(action)
        env.render()
    env.close()

# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # --- ParkingEnv base configuration -------------------------------------------------
    parking_cfg = dict(
        vehicle_type="arc",
        timestep=0.1,
        max_steps=500,
        render_mode=None,
        scenario_mode="parking",
        lidar_max_range=30.0,
        world_size=30.0,
        difficulty_level=10,
    )

    # --- Build VecEnv (1 env for evaluation) -------------------------------------------
    vec_env = DummyVecEnv([make_env_fn(**parking_cfg)])

    # --- Optional VecNormalize ---------------------------------------------------------
    if args.vecnorm:
        if not args.norm_path or not args.norm_path.exists():
            raise FileNotFoundError("VecNormalize enabled but --norm_path is missing.")
        vec_env = VecNormalize.load(args.norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print("[VecNormalize] stats loaded from", args.norm_path)

    # --- Load model --------------------------------------------------------------------
    model: RecurrentPPO = RecurrentPPO.load(args.model, env=vec_env, device=args.device)
    print("[Loaded]", args.model)

    # --- Evaluate ----------------------------------------------------------------------
    evaluate(model, vec_env, episodes=args.episodes)

    # --- Optional playback -------------------------------------------------------------
    if args.play:
        play_one(model, parking_cfg, render_mode=args.render)
