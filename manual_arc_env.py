'''manual_arc_env.py – 手动控制 VehicleArc 模型进行泊车实验
==============================================================
此脚本允许使用键盘控制基于离散弧长和转向角的车辆模型 `VehicleArc`。

控制方式：
    ↑ / ↓：增大 / 减小弧长索引（控制前进后退距离）
    ← / →：左转 / 右转索引（控制方向）
    C：转向归中
    SPACE：重复上一步动作
    R：重置当前回合
    ESC：退出程序

运行方式：
    python manual_arc_env.py           # 启动带界面窗口控制
    python manual_arc_env.py --headless  # 无界面测试逻辑运行
'''

import argparse
import time
import numpy as np
import pygame
from parking_env_pkg import ParkingEnv
from vehicles.vehicle_arc import VehicleArc


def make_env(render: bool):
    cfg = dict(
        timestep=0.1,
        max_steps=500,
        lidar_max_range=15.0,
        render_mode="human" if render else "none",
        vehicle_type="arc",
        scenario_mode="random",
        data_dir="./pygame_input_features_new_withinBEV_no_parallel_parking",
        manual=True,
        world_size=30.0,
        occupy_prob=0.5,
        gap=4.0,
        wall_thickness=0.1,
    )
    return ParkingEnv(cfg)


STEER_DEG = list(range(-28, 29, 4))
STEER_CHOICES = np.deg2rad(STEER_DEG).astype(np.float32)
ARC_CHOICES = np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32)


def run(env: ParkingEnv, render: bool):
    steer_idx = VehicleArc.N_STEER // 2
    arc_idx = 2
    last_action = np.array([steer_idx, arc_idx], dtype=np.int32)

    if render:
        pygame.init()
        pygame.display.set_caption("Parking – VehicleArc manual control")
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
                    if keys[pygame.K_UP]:
                        arc_idx = min(VehicleArc.N_ARC - 1, arc_idx + 1)
                    if keys[pygame.K_DOWN]:
                        arc_idx = max(0, arc_idx - 1)
                    if keys[pygame.K_LEFT]:
                        steer_idx = min(VehicleArc.N_STEER - 1, steer_idx + 1)
                    if keys[pygame.K_RIGHT]:
                        steer_idx = max(0, steer_idx - 1)
                    if keys[pygame.K_c]:
                        steer_idx = VehicleArc.N_STEER // 2

                    action = np.array([steer_idx, arc_idx], dtype=np.int32)
                    last_action = action.copy()
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            if terminated or truncated:
                steer_idx = VehicleArc.N_STEER // 2
                arc_idx = VehicleArc.N_ARC // 2
                last_action = np.array([steer_idx, arc_idx], dtype=np.int32)
                env.reset()

            if screen:
                title = (
                    f"Arc({arc_idx}) {ARC_CHOICES[arc_idx]:+0.2f} m  |  "
                    f"Steer({steer_idx}) {STEER_DEG[steer_idx]:+d}°"
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
