"""manual_arc_env.py – Keyboard driver for *VehicleArc* (15×4 discrete)
=======================================================================
Control the **discrete‑arc** parking environment using your keyboard.
Only the *VehicleArc* model is handled, so we fix `vehicle_type='arc'` when
creating the environment.

Keybindings
-----------
* **↑** : *Increase* arc‑index  (→ longer forward arc)
* **↓** : *Decrease* arc‑index  (→ shorter / backward arc)
* **←** : steer **more left**  (towards −28°)
* **→** : steer **more right** (towards +28°)
* **C** : centre steering (index 7 ⇒ 0°)
* **Space** : repeat last action (like holding brake)
* **R** : reset episode
* **Esc / close window** : quit

Hot‑reloading: the window title shows current *(steer_idx, arc_idx)* and the
corresponding physical values for quick reference.

Run
---
```bash
python manual_arc_env.py            # with on‑screen rendering
python manual_arc_env.py --headless # logic only, no window
```
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pygame

from parking_env_pkg import ParkingEnv
from vehicles.vehicle_continuous import VehicleContinuous
from vehicles.vehicle_disc_accel import VehicleDiscAccel
from vehicles.vehicle_arc import VehicleArc


###############################################################################
# Environment helper
###############################################################################

def make_env(render: bool):
    cfg = dict(
        timestep=0.1,
        max_steps=500,
        render_mode="human" if render else "none",
        vehicle_type="arc",
        scenario_mode="empty",     # file empty random
        manual=True,

        lidar_max_range=15.0,

        world_size=20.0,
        occupy_prob=0.5,          # 初级课程
        gap=4.0,
        wall_thickness=0.1,
    )
    env = ParkingEnv(cfg)
    return env

# 3. Discrete arc‑length grid (15×4) -----------------------------------------
# ---------------------------------------------------------------------------
STEER_DEG = list(range(-28, 29, 4))  # ‑28 … +28, step 4 → 15
STEER_CHOICES = np.deg2rad(STEER_DEG).astype(np.float32)
ARC_CHOICES = np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32)
N_STEER, N_ARC = len(STEER_CHOICES), len(ARC_CHOICES)
###############################################################################
# Main interaction loop
###############################################################################

def run(env: ParkingEnv, render: bool):
    # current discrete indices
    steer_idx = VehicleArc.N_STEER // 2  # 7 (0 deg)
    arc_idx = 2                           # +0.25 m
    last_action = np.array([steer_idx, arc_idx], dtype=np.int32)

    if render:
        pygame.init()
        pygame.display.set_caption("Parking – VehicleArc manual control")
        from parking_env_pkg.render import PygameRenderer
        renderer = PygameRenderer(screen_size=(800, 800))  # 或其它尺寸
        screen = pygame.display.set_mode(renderer.screen_size)
        clock = pygame.time.Clock()
    else:
        screen = None

    running = True
    while running:
        obs, _ = env.reset()
        obs_list = []  # 👈 用于收集所有 obs
        terminated = truncated = False
        ep_reward = 0.0
        start_t = time.time()

        while not (terminated or truncated):
            # handle quit/reset events
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
                    terminated = True  # early episode reset

            # poll keys
            keys = pygame.key.get_pressed() if screen else []

            if screen and keys:
                if keys[pygame.K_SPACE]:
                    action = last_action  # repeat
                else:
                    # arc length control
                    if keys[pygame.K_UP]:
                        arc_idx = min(VehicleArc.N_ARC - 1, arc_idx + 1)
                    if keys[pygame.K_DOWN]:
                        arc_idx = max(0, arc_idx - 1)

                    # steering control
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
            obs_list.append(obs)

            if terminated or truncated:
                print(env.vehicle.switch_count)
                #  👇加上这三行确保清除历史输入影响
                steer_idx = VehicleArc.N_STEER // 2
                arc_idx = VehicleArc.N_ARC // 2
                last_action = np.array([steer_idx, arc_idx], dtype=np.int32)
                env.reset()

            if screen:
                # HUD text
                title = (
                    f"Arc({arc_idx}) {ARC_CHOICES[arc_idx]:+0.2f} m  |  "
                    f"Steer({steer_idx}) {STEER_DEG[steer_idx]:+d}°"
                )
                pygame.display.set_caption(title)
                env.render()
                clock.tick(30)

        dur = time.time() - start_t
        print(f"Episode finished in {dur:.1f}s; Reward = {ep_reward:.2f}")
        obs_array = np.array(obs_list)
        mean = np.mean(obs_array, axis=0)
        std = np.std(obs_array, axis=0)

        # print("\n🔍 观测统计分析:")
        # for i, (m, s) in enumerate(zip(mean, std)):
        #     print(f"obs[{i}]: mean = {m:.4f}, std = {s:.4f}")


###############################################################################
# CLI
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Manual controller for VehicleArc")
    p.add_argument("--headless", action="store_true", help="no rendering window")
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
