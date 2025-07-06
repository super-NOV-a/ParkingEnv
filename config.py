import os
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # 实验设置
    exp_name: str = "parking_ppo"
    seed: int = 42
    device: str = "cuda"  # "cpu" or "cuda"

    # 环境参数
    data_dir: str = "scenarios"
    max_range: float = 15.0
    timestep: float = 0.1
    max_steps: int = 500
    scenario_mode: str = "random"  # "random" or "file"
    world_size: float = 30.0
    min_obstacles: int = 0
    max_obstacles: int = 1
    collision_threshold: float = 0.5

    # 训练参数
    n_envs: int = 8  # 并行环境数量
    total_timesteps: int = 2_000_000
    lr: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # 回调参数
    save_freq: int = 50_000  # 保存模型的步数间隔
    eval_freq: int = 25_000  # 评估模型的步数间隔

    # 路径设置
    log_dir: str = os.path.join("logs", exp_name)
    save_dir: str = os.path.join("models", exp_name)