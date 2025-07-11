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
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--vehicle_type", default="arc")
    p.add_argument("--timestep", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=500)
    return p.parse_args()

## -----------------------------------------------------------------------------------------------
## 纯评估模式
# python Test.py --model "runs/your_model/final.zip"

## 评估完再播放一个轨迹
# python Test.py --model "runs/your_model/final.zip" --render human --play
## -----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    parking_cfg = dict(
        timestep=args.timestep,
        max_steps=args.max_steps,
        render_mode=None,
        scenario_mode="empty",
        data_dir="./pygame_input_features_new_withinBEV_no_parallel_parking",
        lidar_max_range=15.0,
        difficulty_level=5,
        world_size=40.0,
        occupy_prob=0.3,
        gap=4.0,
        wall_thickness=0.15,
        model_ckpt=args.model,
        policy_class="RadarConvFusion",
        vehicle_type="arc",
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

    # Evaluation
    mean_r, std_r = evaluate_policy(
        model, vec_env, n_eval_episodes=args.episodes, deterministic=True
    )
    print(f"Evaluation: {mean_r:.2f} ± {std_r:.2f} over {args.episodes} episodes")

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
