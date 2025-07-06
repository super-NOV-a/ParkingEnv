import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# 假设你的ParkingEnv类定义在 parking_env.py 中
from Random_JsonEnv import ParkingEnv

# 你直接复制代码中ParkingEnv类到此文件或保证能导入


def make_env():
    config = {
        'data_dir': 'C:\\AI_Planner\\RL\\pygame_input_features_new_withinBEV_no_parallel_parking',
        'max_range': 15.0,
        'timestep': 0.1,
        'max_steps': 500,
        'render_mode': None,    # 训练时关闭渲染
        'scenario_mode': 'random',
        'world_size': 30.0,
        'min_obstacles': 0,
        'max_obstacles': 1,
        "manual": False,
    }
    env = ParkingEnv(config)
    return env


def main():
    env = make_env()

    # 创建保存模型和日志目录
    log_dir = "./ppo_parking_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Checkpoint callback，每训练一定步数保存一次模型
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="ppo_parking")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        max_grad_norm=0.5,
        seed=123,
    )

    # 训练10万步（根据需求调整）
    total_timesteps = 100000

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # 保存最终模型
    model.save(os.path.join(log_dir, "ppo_parking_final"))

    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
