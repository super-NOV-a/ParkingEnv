'''manual_continuous_env.py – 手动控制 VehicleContinuous 模型进行泊车实验
====================================================================
该脚本允许使用键盘控制基于连续动作（速度 + 转向）的车辆模型 `VehicleContinuous`。

控制方式：
    ← / →：转向（每次 ±0.05，范围 −1 到 1）
    ↑ / ↓：加速 / 减速（每次 ±0.1，范围 −1 到 1）
    C：转向归中
    S：速度归零
    SPACE：重复上一次动作（如保持油门）
    R：重置当前回合
    ESC：退出程序

运行方式：
    python manual_continuous_env.py            # 启动带界面窗口控制
    python manual_continuous_env.py --headless # 无界面测试逻辑运行
'''


import argparse
import time
import numpy as np
import pygame
from parking_env_pkg import ParkingEnv


def make_env(render: bool):
    cfg = dict(
        timestep=0.1,
        max_steps=500,
        render_mode="human" if render else "none",
        vehicle_type="continuous",
        scenario_mode="random",
        data_dir="./Train_data_energy/pygame_input_features_new_withinBEV_no_parallel_parking",
        manual=True,
        lidar_max_range=30.0,
        world_size=40.0,
        difficulty_level=10,     # 修改成指定难度就可，不需要给定障碍等内容, 在parking_core中，指定了不同难度成功条件

        # 配置课程，scenario_manager.py中的__post_init__方法提供了默认的课程，但是训练起来较难成长
        # 不同等级之间难度差别大
        gap_base = 4,
        gap_step = 0.2,  # 总共十个level
        gap_min = 2,
        occupy_prob_base = 0.5,
        occupy_prob_step = 0.05,
        occupy_prob_max = 0.9,
        wall_thickness=0.1,
    )
    return ParkingEnv(cfg)


def run(env: ParkingEnv, render: bool):
    steer_cmd = 0.0
    accel_cmd = 0.0
    last_action = np.array([steer_cmd, accel_cmd], dtype=np.float32)

    if render:
        pygame.init()
        pygame.display.set_caption("Parking – Continuous manual control")
        from parking_env_pkg.render import PygameRenderer
        renderer = PygameRenderer(screen_size=(800, 800))
        screen = pygame.display.set_mode(renderer.screen_size)
        clock = pygame.time.Clock()
    else:
        screen = None

    running = True
    while running:
        obs, _ = env.reset()
        terminated = truncated = False
        ep_reward = 0.0
        start_t = time.time()

        while not (terminated or truncated):
            for event in pygame.event.get() if screen else []:
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                    env.close()
                    if screen:
                        pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    terminated = True

            keys = pygame.key.get_pressed() if screen else []

            if screen and keys:
                if keys[pygame.K_SPACE]:
                    action = last_action
                else:
                    if keys[pygame.K_LEFT]:
                        steer_cmd = min(1.0, steer_cmd + 0.05)
                    if keys[pygame.K_RIGHT]:
                        steer_cmd = max(-1.0, steer_cmd - 0.05)
                    if keys[pygame.K_c]:
                        steer_cmd = 0.0

                    if keys[pygame.K_UP]:
                        accel_cmd = min(1.0, accel_cmd + 0.1)
                    if keys[pygame.K_DOWN]:
                        accel_cmd = max(-1.0, accel_cmd - 0.1)
                    if keys[pygame.K_s]:
                        accel_cmd = 0.0

                    action = np.array([steer_cmd, accel_cmd], dtype=np.float32)
                    last_action = action.copy()
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            if terminated or truncated:
                steer_cmd = 0.0
                accel_cmd = 0.0
                last_action = np.array([steer_cmd, accel_cmd], dtype=np.float32)
                env.reset()

            if screen:
                pygame.display.set_caption(f"Steer {steer_cmd:+.2f} | Accel {accel_cmd:+.2f}")
                env.render()
                clock.tick(30)

        dur = time.time() - start_t
        print(f"Episode finished in {dur:.1f}s; Reward = {ep_reward:.2f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env = make_env(render=not args.headless)
    try:
        run(env, render=not args.headless)
    finally:
        env.close()
        if not args.headless:
            pygame.quit()
