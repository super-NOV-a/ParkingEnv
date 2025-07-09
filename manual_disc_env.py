"""manual_disc_accel_env.py – Keyboard controller for VehicleDiscAccel (9 actions)
=================================================================================
Drive the parking environment with the 3×3 *steer × target‑speed* discrete
model.

Grids
-----
* **STEER_GRID** = (−1, 0, +1) × max_steer → indices 0 (left),1 (center),2 (right)
* **SPEED_GRID** = (+1, 0, −1) × max_speed → indices 0 (forward),1 (coast),2 (reverse)
Action ID = `speed_idx * 3 + steer_idx`.

Keybindings
-----------
* **← / →** : steer index −/+
* **↑ / ↓** : speed index −/+  (↑ → forward, ↓ → reverse)
* **Space** : repeat previous action
* **C**     : steer center (idx 1)
* **X**     : speed coast  (idx 1)
* **R**     : reset episode
* **Esc / quit window** : exit

Run
---
```bash
python manual_disc_accel_env.py          # with rendering
python manual_disc_accel_env.py --headless  # logic only
```
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pygame

from parking_env_pkg import ParkingEnv
from vehicles import VehicleDiscAccel, STEER_GRID, SPEED_GRID

###############################################################################
# Env factory
###############################################################################

def make_env(render: bool):
    cfg = dict(
        timestep=0.1,
        max_steps=500,
        render_mode="human" if render else "none",
        scenario_mode="random",
        world_size=30.0,
        min_obstacles=0,
        max_obstacles=10,
        max_speed=3.0,
        vehicle_type="disc_accel",
        manual=True,
    )
    return ParkingEnv(cfg)

###############################################################################
# Main loop
###############################################################################

def run(env: ParkingEnv, render: bool):
    steer_idx = 1  # center
    speed_idx = 1  # coast
    last_action = speed_idx * 3 + steer_idx

    if render:
        pygame.init()
        pygame.display.set_caption("Parking – DiscAccel manual control")
        from parking_env_pkg.render import PygameRenderer
        renderer = PygameRenderer(screen_size=(800, 800))  # 或其它尺寸
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

        if screen:
            env.render()

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
                    # steering
                    if keys[pygame.K_LEFT]:
                        steer_idx = max(0, steer_idx - 1)
                    if keys[pygame.K_RIGHT]:
                        steer_idx = min(2, steer_idx + 1)
                    if keys[pygame.K_c]:
                        steer_idx = 1

                    # speed
                    if keys[pygame.K_UP]:
                        speed_idx = max(0, speed_idx - 1)  # toward forward
                    if keys[pygame.K_DOWN]:
                        speed_idx = min(2, speed_idx + 1)  # toward reverse
                    if keys[pygame.K_x]:
                        speed_idx = 1  # coast

                    action_id = speed_idx * 3 + steer_idx
                    last_action = action_id
            else:
                action_id = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action_id)
            ep_reward += reward
            if terminated or truncated:
                #  👇加上这四行确保清除历史输入影响
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

###############################################################################
# CLI
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Manual controller (disc_accel model)")
    p.add_argument("--headless", action="store_true", help="run without rendering")
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
