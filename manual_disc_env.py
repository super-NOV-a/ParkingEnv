'''manual_disc_env.py – 手动控制 VehicleDiscAccel 模型进行泊车实验
====================================================================
该脚本允许使用键盘控制基于离散动作网格（3x3 的速度 × 转向组合）的 `VehicleDiscAccel` 模型。

动作网格：
    STEER_GRID = (-1, 0, +1) × 最大转角
    SPEED_GRID = (+1, 0, −1) × 最大速度

控制方式：
    ← / →：改变转向索引（向左 / 向右）
    ↑ / ↓：改变速度索引（前进 / 后退）
    C：转向归中
    X：速度归零（滑行）
    SPACE：重复上一次动作
    R：重置当前回合
    ESC：退出程序

运行方式：
    python manual_disc_env.py            # 启动带界面窗口控制
    python manual_disc_env.py --headless # 无界面测试逻辑运行
'''

import argparse
import time
import numpy as np
import pygame
from parking_env_pkg import ParkingEnv
# from vehicles import STEER_GRID, SPEED_GRID
STEER_GRID = (-1.0, 0.0, 1.0)
SPEED_GRID = (1.0, 0.0, -1.0)

def make_env(render: bool):
    cfg = dict(
        timestep=0.1,
        max_steps=500,
        lidar_max_range=15.0,
        render_mode="human" if render else "none",
        vehicle_type="disc_accel",
        scenario_mode="random",
        data_dir="./pygame_input_features_new_withinBEV_no_parallel_parking",
        manual=True,
        world_size=30.0,
        occupy_prob=0.5,
        gap=4.0,
        wall_thickness=0.1,
    )
    return ParkingEnv(cfg)


def run(env: ParkingEnv, render: bool):
    steer_idx = 1
    speed_idx = 1
    action_id = speed_idx * 3 + steer_idx
    last_action = action_id

    if render:
        pygame.init()
        pygame.display.set_caption("Parking – DiscAccel manual control")
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
                    action_id = last_action
                else:
                    if keys[pygame.K_LEFT]:
                        steer_idx = max(0, steer_idx - 1)
                    if keys[pygame.K_RIGHT]:
                        steer_idx = min(2, steer_idx + 1)
                    if keys[pygame.K_c]:
                        steer_idx = 1

                    if keys[pygame.K_UP]:
                        speed_idx = max(0, speed_idx - 1)
                    if keys[pygame.K_DOWN]:
                        speed_idx = min(2, speed_idx + 1)
                    if keys[pygame.K_x]:
                        speed_idx = 1

                    action_id = speed_idx * 3 + steer_idx
                    last_action = action_id
            else:
                action_id = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action_id)
            ep_reward += reward

            if terminated or truncated:
                steer_idx = 1
                speed_idx = 1
                action_id = speed_idx * 3 + steer_idx
                last_action = action_id
                env.reset()

            if screen:
                title = (
                    f"Steer idx {steer_idx} ({STEER_GRID[steer_idx]:+0.0f})  |  "
                    f"Speed idx {speed_idx} ({SPEED_GRID[speed_idx]:+0.0f})  |  AID {action_id}"
                )
                pygame.display.set_caption(title)
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
