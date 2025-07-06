import numpy as np
import pygame
from stable_baselines3 import PPO

env_name = "vel_point"  # "point" "lidar" "car" "vel_point" "lidar_car"
if env_name == "point":
    from PointEnv import PointEnv
    env = PointEnv(render_mode="human")

elif env_name == "vel_point":
    from vel_PointEnv import PointEnv
    env = PointEnv(render_mode="human")

elif env_name == "car":
    from CarEnv import CarEnv
    env = CarEnv(render_mode="human")

elif env_name == "jit_car":
    from jit_CarEnv import CarEnv
    env = CarEnv(render_mode="human")

elif env_name == "lidar_car":
    from parkingEnv import CarEnv
    env = CarEnv(render_mode="human")

else:
    raise AssertionError("--parallel_wrap.py中没找到该环境")

# 加载训练好的模型
# MODEL_PATH = "car_env_models/rl_car_model_360000_steps"  # 根据训练代码中保存的模型名称
# MODEL_PATH = env_name + "_env_models/ppo_" + env_name + "_env_4.zip"  # 根据训练代码中保存的模型名称
MODEL_PATH = "E:\python\PointParkingEnv\\final_model\ppo_vel_point_env_18.zip"
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
