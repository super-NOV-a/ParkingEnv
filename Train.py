import multiprocessing
from parallel_wrap import make_parallel_envs
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from custom_callbacks import CustomLoggingCallback

env_name = "lidar_car"  # "point" "lidar" "car" "vel_point"


def train():
    # 创建并行环境 (8个并行)
    env = make_parallel_envs(num_envs=6, env_name=env_name)

    # 创建PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./" + env_name + "_env_tensorboard/",
        device="auto",
        # 调整超参数以获得更好的性能
        batch_size=64,
        n_steps=1024,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=2e-3,
        clip_range=0.2,
        n_epochs=10,
        vf_coef=0.5
    )

    # 每x步保存一次模型
    checkpoint_callback = CheckpointCallback(
        save_freq=30000,
        save_path="./"+env_name+"_env_models/",
        name_prefix="rl_"+env_name+"_model"
    )

    # 创建自定义日志回调
    logging_callback = CustomLoggingCallback()

    # 组合回调
    callback = CallbackList([checkpoint_callback, logging_callback])

    # 训练模型
    model.learn(
        total_timesteps=500_000,
        callback=callback,
        tb_log_name="ppo_"+env_name+"_env",
        log_interval=1  # 确保每次迭代都记录
    )

    # 保存最终模型
    model.save(env_name + "_env_models/ppo_" + env_name + "_env_")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    train()
