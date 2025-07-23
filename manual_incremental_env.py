"""manual_incremental_env.py – 手动控制 VehicleIncremental 模型进行泊车实验
===================================================================
本脚本提供 **增量式车辆模型**（``VehicleIncremental``）的键盘手动控制，
整体结构与 *manual_arc_env.py* 保持一致，方便快速对比和测试。

控制方式（默认方向：车辆朝 **+x** 方向时为"正向"）::

    ↑ / ↓  : 油门 / 刹车（加速度归一化 acc_norm = +1 / −1）
    ← / →  : 左 / 右打方向（方向盘增量 d_steer_norm = +1 / −1）
    C      : 方向盘归中（d_steer_norm = 0）
    SPACE  : 重复上一步动作
    R      : 重置当前回合
    ESC    : 退出程序

运行示例::

    # 带可视化窗口
    python manual_incremental_env.py

    # 仅逻辑验证（无渲染）
    python manual_incremental_env.py --headless
"""

import argparse
import time
import math
from pathlib import Path

import numpy as np
import pygame
import gymnasium as gym

from parking_env_pkg import ParkingEnv
from vehicles.vehicle_incremental import VehicleIncremental

# ---------------------------------------------------------------------------
# 1. 构造环境 + 替换为增量式车辆
# ---------------------------------------------------------------------------

def make_env(render: bool) -> ParkingEnv:
    """创建 ParkingEnv 并将内部车辆替换为 *VehicleIncremental*。"""

    cfg = dict(
        timestep=0.1,
        max_steps=500,
        render_mode="human" if render else "none",
        vehicle_type="continuous",  # 初始占位，后续会替换
        scenario_mode="box",
        lidar_max_range=30.0,
        world_size=40.0,
        difficulty_level=10,
        wall_thickness=0.1,
    )

    env = ParkingEnv(cfg)

    # ---- 用增量车辆模型替换原 vehicle ------------------------------------
    inc_vehicle = VehicleIncremental(
        wheelbase=env.wheelbase,
        width=env.car_width,
        front_hang=env.front_hang,
        rear_hang=env.rear_hang,
        max_steer=env.max_steer,
        max_speed=env.max_speed,
        dt=env.dt,
        max_acc=4.0,
        max_steer_rate=math.radians(90),
    )
    env.vehicle = inc_vehicle

    # 覆盖动作空间：Box([-1, -1], [1, 1]) → (d_steer_norm, acc_norm)
    env.action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

    # 关闭基于离散索引的 LUT（观测里 prev_action 置零即可）
    env.N_STEER = env.N_ARC = 1

    return env

# ---------------------------------------------------------------------------
# 2. 主循环
# ---------------------------------------------------------------------------

def run(env: ParkingEnv, render: bool):
    d_steer_norm = 0.0  # 当前帧方向盘增量
    acc_norm = 0.0      # 当前帧加速度指令
    last_action = np.array([0.0, 0.0], dtype=np.float32)

    if render:
        pygame.init()
        pygame.display.set_caption("Parking – VehicleIncremental manual control")
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((800, 800))
    else:
        screen = None

    running = True
    while running:
        obs, _ = env.reset()
        terminated = truncated = False
        ep_reward = 0.0
        start_t = time.time()

        while not (terminated or truncated):
            # 1) 处理退出 / 重置事件 --------------------------------------
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
                    terminated = True  # 提前结束 → reset

            # 2) 读取键盘状态 ---------------------------------------------
            keys = pygame.key.get_pressed() if screen else []
            if screen and keys:
                # --- 重复动作 -----------------------------------------
                if keys[pygame.K_SPACE]:
                    action = last_action.copy()

                else:
                    # --- 加速度指令 -----------------------------------
                    if keys[pygame.K_UP]:
                        acc_norm = 1.0
                    elif keys[pygame.K_DOWN]:
                        acc_norm = -1.0
                    else:
                        acc_norm = 0.0

                    # --- 方向盘增量 -----------------------------------
                    if keys[pygame.K_LEFT]:
                        d_steer_norm = 1.0
                    elif keys[pygame.K_RIGHT]:
                        d_steer_norm = -1.0
                    elif keys[pygame.K_c]:
                        d_steer_norm = 0.0
                    else:
                        d_steer_norm = 0.0

                    action = np.array([d_steer_norm, acc_norm], dtype=np.float32)
                    last_action = action.copy()
            else:
                # 头less 时随机探索
                action = env.action_space.sample()

            # 3) 交互一步 ----------------------------------------------
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            # 4) 渲染 & HUD -------------------------------------------
            if screen:
                title = (
                    f"d_steer_norm: {action[0]:+0.2f}  |  acc_norm: {action[1]:+0.2f}"
                )
                pygame.display.set_caption(title)
                env.render()
                clock.tick(30)

            # 5) Episode 结束处理 --------------------------------------
            if terminated or truncated:
                # 重置内部动作（防止直接继承上一集方向）
                d_steer_norm = 0.0
                acc_norm = 0.0
                last_action[:] = 0.0

        dur = time.time() - start_t
        print(f"Episode finished in {dur:.1f}s; Reward = {ep_reward:.2f}")

# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="Run without rendering")
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
