import numpy as np
import pygame
from stable_baselines3 import PPO

env_name = "car"  # "point" "lidar" "car" "vel_point"
if env_name == "point":
    from PointEnv import PointEnv
    env = PointEnv(render_mode="human")

elif env_name == "vel_point":
    from vel_PointEnv import PointEnv
    env = PointEnv(render_mode="human")

elif env_name == "car":
    from CarEnv import CarEnv
    env = CarEnv(render_mode="human")

else:
    raise AssertionError("--parallel_wrap.py中没找到该环境")

# 加载训练好的模型
# MODEL_PATH = "vel_point_env_models/rl_vel_point_model_240000_steps"  # 根据训练代码中保存的模型名称
MODEL_PATH = "final_model/ppo_" + env_name + "_env_final.zip"  # 根据训练代码中保存的模型名称
model = PPO.load(MODEL_PATH)

if __name__ == "__main__":

    obs, _ = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 使用PPO模型预测动作（关键修改）
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        # 定期重置环境
        if terminated or truncated:
            print(f"Episode finished! Total reward: {env.total_reward}")
            obs, _ = env.reset()

    env.close()
