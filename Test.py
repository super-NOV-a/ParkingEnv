import os
import time
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from parking_env import ParkingEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="./ppo_parking_logs/ppo_parking_final.zip",
                        help="路径到训练好的 .zip 模型")
    parser.add_argument("--episodes", type=int, default=5,
                        help="测试回合数")
    parser.add_argument("--control_mode", choices=["continuous", "discrete"],
                        default="discrete", help="动作空间类型")
    parser.add_argument("--deterministic", action="store_true",
                        help="使用确定性策略")
    parser.add_argument("--render", action="store_true",
                        help="是否渲染环境")
    parser.add_argument("--manual", action="store_true",
                        help="手动控制，忽略模型")
    return parser.parse_args()


def make_env(control_mode: str, render: bool, manual: bool):
    config = {
        'data_dir': 'C:\\AI_Planner\\RL\\pygame_input_features_new_withinBEV_no_parallel_parking',
        'lidar_max_range': 15.0,
        'timestep': 0.1,
        'max_steps': 500,
        'render_mode': 'human' if render else None,
        'scenario_mode': 'random',
        'world_size': 30.0,
        'min_obstacles': 0,
        'max_obstacles': 0,
        'manual': manual,
        'control_mode': control_mode,
    }
    return ParkingEnv(config)


def main():
    args = parse_args()

    # 如果选择手动模式，不加载模型
    model = None
    if not args.manual:
        if not os.path.isfile(args.model_path):
            raise FileNotFoundError(f"模型未找到: {args.model_path}")
        model = PPO.load(args.model_path)
        print(f"✅ 已加载模型: {args.model_path}")

    env = make_env(args.control_mode, args.render, args.manual)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done, truncated = False, False
        ep_reward = 0.0
        step = 0
        while not (done or truncated):
            if args.manual:
                # 在 env 内部处理键盘，外部只需发送占位动作
                action = np.array([0.0, 0.0]) if args.control_mode == 'continuous' else 1
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            step += 1
            if args.render and not args.manual:
                # 若渲染但非手动，适当放慢速度
                time.sleep(env.dt)
        print(f"Episode {ep}: steps={step}, reward={ep_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
