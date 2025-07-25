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
from vehicles.vehicle_arc import VehicleArc, ARC_CHOICES, STEER_CHOICES, STEER_DEG


def make_env(render: bool):
    cfg = dict(
        timestep=0.1,
        max_steps=500,
        render_mode="human" if render else "none",
        vehicle_type="arc",
        scenario_mode="parking",   # file random box empty  random_box  parking
        data_dir="./Train_data_energy/pygame_input_features_new_withinBEV_no_parallel_parking",
        manual=True,
        lidar_max_range=30.0,
        world_size=30.0,
        difficulty_level=0,     # 修改成指定难度就可，不需要给定障碍等内容, 在parking_core中，指定了不同难度成功条件

        # 配置课程，scenario_manager.py中的__post_init__方法提供了默认的课程，但是训练起来较难成长
        # 不同等级之间难度差别大
        gap_base = 2,
        gap_step = 0.17,  # 总共十个level
        gap_min = 0.3,
        occupy_prob_base = 0.5,
        occupy_prob_step = 0.05,
        occupy_prob_max = 1,
        wall_thickness=0.1,
    )
    return ParkingEnv(cfg)


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
            print(f"Step Reward = {reward:.4f}")
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
        print(f"Episode finished in {dur:.1f}s; Reward = {ep_reward:.4f}")


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
