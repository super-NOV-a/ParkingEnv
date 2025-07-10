"""test.py – Evaluate or visualise a trained PPO model on *ParkingEnvArc*
===========================================================================
Quick examples
--------------
1. **Headless evaluation (mean reward) – 20 episodes**
   ```bash
   python test.py --model runs/arc_exp01/models/final.zip --vecnorm \
                  --norm-path runs/arc_exp01/vec_normalize.pkl
   ```
2. **Watch one deterministic episode with on‑screen rendering**
   ```bash
   python test.py --model runs/arc_exp01/models/final.zip \
                  --render human --play
   ```
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from parking_env_pkg import ParkingEnv

###############################################################################
# Helpers
###############################################################################

def make_env_fn(**parking_cfg: Dict):
    """Return a Monitor‑wrapped env (factory for DummyVecEnv)."""

    def _init() -> gym.Env:
        return Monitor(ParkingEnv(parking_cfg))

    return _init

###############################################################################
# CLI
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Test PPO model on ParkingEnv")

    p.add_argument("--model", type=Path, # required=True, 
                    help=".zip model file", 
                    default="C:\AI_Planner\RL\\runs\ppo_arc_empty\\checkpoints\ppo_arc_100000_steps.zip")
                    # default="C:\AI_Planner\RL\\runs\ppo_arc_cnn1\checkpoints\ppo_arc_1000000_steps.zip")  # 给定模型路径

    p.add_argument("--norm_path", type=Path, default=None,
                   help="VecNormalize statistics pickle (if used during training)")
    p.add_argument("--vecnorm", action="store_true", help="enable VecNormalize wrapper")

    p.add_argument("--episodes", type=int, default=10, help="episodes for evaluation")
    p.add_argument("--render", type=str, choices=["none", "human", "rgb_array"], default="human", help="是否渲染")
    p.add_argument("--play", action="store_true", help="run & render one episode after eval")

    # Must be same as training to ensure compatibility
    p.add_argument("--timestep", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=500)

    return p.parse_args()

###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(args.model)

    # Parking‑env base config (match training)
    parking_cfg = dict(
        timestep=args.timestep,
        max_steps=args.max_steps,
        render_mode=None,
        scenario_mode="random",
        lidar_max_range=15.0,

        world_size=40.0,
        occupy_prob=0.3,          # 初级课程
        gap=4.0,
        wall_thickness=0.15,
        vehicle_type="arc",
    )

    # Build VecEnv (single env) ------------------------------------------------
    vec_env = DummyVecEnv([make_env_fn(**parking_cfg)])

    # Attach VecNormalize if requested -----------------------------------------
    if args.vecnorm:
        if not args.norm_path:
            raise ValueError("--vecnorm requires --norm-path pointing to vec_normalize.pkl")
        if not args.norm_path.exists():
            raise FileNotFoundError(args.norm_path)
        vec_env = VecNormalize.load(args.norm_path, vec_env)
        vec_env.training = False  # turn off updates
        vec_env.norm_reward = False

    # Load model
    model: PPO = PPO.load(args.model, vec_env)

    # Evaluation ---------------------------------------------------------------
    mean_r, std_r = evaluate_policy(
        model, vec_env, n_eval_episodes=args.episodes, deterministic=True
    )
    print(f"Evaluated {args.episodes} episodes → mean reward {mean_r:.2f} ± {std_r:.2f}")

    # Optional playback --------------------------------------------------------
    if args.play and args.render != "none":
        env = ParkingEnv({**parking_cfg, "render_mode": args.render})
        if args.vecnorm:
            env = VecNormalize.load(args.norm_path, DummyVecEnv([lambda: env]))
            env.training = False
            env.norm_reward = False
        
        while True:
            obs, _ = env.reset()
            done = truncated = coll= False
            while not (done or truncated):
                # action, _ = model.predict(obs, deterministic=True)
                action = env.action_space.sample()
                obs, _, done, truncated, info = env.step(action)
                
                collision = info.get("collision", False)
                if done or truncated:
                    if done and not collision:
                        print("success")
                    else:
                        print("failure")
                    env.reset()
                env.render()
        env.close()

    vec_env.close()
