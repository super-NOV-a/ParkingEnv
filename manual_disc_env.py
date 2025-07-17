"""
manual_disc_env.py ─ 手动键盘控制 VehicleDiscAccel
--------------------------------------------------
← / →  : 调整转向索引  (15 档，-30°…+30°)
C      : 转向回中 (idx 7, 0°)

↑ / ↓  : 调整加速度  (-1 / 0 / +1 m/s²)
X      : 加速度归零 (idx 1, 0 m/s²)

SPACE  : 重复上一帧动作
R      : 提前重置当前回合
ESC    : 退出
"""

import argparse, time, math, pygame, numpy as np
from parking_env_pkg import ParkingEnv
from vehicles.vehicle_disc_accel import (
    VehicleDiscAccel, STEER_CHOICES, ACC_CHOICES,
)

N_STEER, N_ACC = VehicleDiscAccel.N_STEER, VehicleDiscAccel.N_ACC

# ----------------------------------------------------------------- 环境构造
def make_env(render=True):
    cfg = dict(
        timestep=0.1, max_steps=500,
        render_mode="human" if render else "none",
        vehicle_type="disc_accel",
        scenario_mode="random",
        world_size=30.0, margin=1.0,
    )
    return ParkingEnv(cfg)

# ----------------------------------------------------------------- 主循环
def run(env: ParkingEnv, render=True):
    # 起始：转向 idx=7 (0°)  加速度 idx=1 (0 m/s²)
    steer_idx, acc_idx = 7, 1
    last_action = steer_idx * N_ACC + acc_idx

    if render:
        pygame.init()
        pygame.display.set_caption("Parking – DiscAccel manual control")
        screen = pygame.display.set_mode((800, 800))
        clock  = pygame.time.Clock()
    else:
        screen = None

    running = True
    while running:
        obs, _ = env.reset()
        terminated = truncated = False
        ep_reward = 0.0
        start_t   = time.time()

        while not (terminated or truncated):
            # ---------- 事件处理 ----------
            for event in pygame.event.get() if screen else []:
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                    env.close()
                    if screen: pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    terminated = True   # 提前结束回合

            keys = pygame.key.get_pressed() if screen else []

            if screen and keys:
                # SPACE ⇒ 重复上一动作
                if keys[pygame.K_SPACE]:
                    action_id = last_action
                else:
                    # ← / → 调整转向 idx
                    if keys[pygame.K_LEFT]:
                        steer_idx = min(N_STEER - 1, steer_idx + 1)
                    if keys[pygame.K_RIGHT]:
                        steer_idx = max(0, steer_idx - 1)
                    if keys[pygame.K_c]:
                        steer_idx = 7  # 中位 0°

                    # ↑ / ↓ 调整加速度 idx
                    if keys[pygame.K_UP]:
                        acc_idx = 2    # +1 m/s²
                    if keys[pygame.K_DOWN]:
                        acc_idx = 0    # -1 m/s²
                    if keys[pygame.K_x]:
                        acc_idx = 1    #  0 m/s²

                    action_id  = steer_idx * N_ACC + acc_idx
                    last_action = action_id
            else:
                action_id = env.action_space.sample()

            # ---------- 环境步进 ----------
            obs, reward, terminated, truncated, _ = env.step(action_id)
            ep_reward += reward

            # ---------- 渲染 ----------
            if screen:
                steer_deg = math.degrees(STEER_CHOICES[steer_idx])
                acc_val   = ACC_CHOICES[acc_idx]
                title = f"Steer idx {steer_idx:2d} ({steer_deg:+.0f}°) | " \
                        f"Acc idx {acc_idx} ({acc_val:+.0f} m/s²) | " \
                        f"AID {action_id}"
                pygame.display.set_caption(title)
                env.render()
                clock.tick(30)

            # 回合结束后自动复位索引
            if terminated or truncated:
                steer_idx, acc_idx = 7, 1
                last_action        = steer_idx * N_ACC + acc_idx

        print(f"Episode finished in {time.time()-start_t:.1f}s; "
              f"Reward = {ep_reward:.2f}")

# ----------------------------------------------------------------- CLI
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
