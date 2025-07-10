"""manual_continuous_env.py – Keyboard controller for VehicleContinuous
===========================================================================
Real‑time manual driving in the parking environment with **continuous** steer &
acceleration commands (range −1 … 1).

Keybindings
-----------
* **← / →**  : steering −/+ (±0.05 per tick)
* **C**      : steering = 0 (centre)
* **↑ / ↓**  : accel_cmd +/− (±0.1 per tick)
* **S**      : accel_cmd = 0 (coast)
* **Space**  : repeat last command (hold)
* **R**      : reset episode
* **Esc / quit window** : exit

Run
---
```bash
python manual_continuous_env.py          # with rendering
python manual_continuous_env.py --headless  # no window
```
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pygame

from parking_env_pkg import ParkingEnv

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
        vehicle_type="continuous",
        manual=True,
    )
    return ParkingEnv(cfg)

###############################################################################
# Main loop
###############################################################################

def run(env: ParkingEnv, render: bool):
    steer_cmd = 0.0  # in [-1, 1]
    accel_cmd = 0.0  # in [-1, 1]
    last_action = np.array([steer_cmd, accel_cmd], dtype=np.float32)

    if render:
        pygame.init()
        pygame.display.set_caption("Parking – Continuous manual control")
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
                    action = last_action
                else:
                    # steering
                    if keys[pygame.K_LEFT]:
                        steer_cmd = min(1.0, steer_cmd + 0.05)
                    if keys[pygame.K_RIGHT]:
                        steer_cmd = max(-1.0, steer_cmd - 0.05)
                    if keys[pygame.K_c]:
                        steer_cmd = 0.0

                    # acceleration
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
                #  👇加上这三行确保清除历史输入影响
                steer_cmd = 0.0
                accel_cmd = 0.0
                last_action = np.array([steer_cmd, accel_cmd], dtype=np.int32)
                env.reset()

            if screen:
                title = f"Steer {steer_cmd:+.2f}  |  Accel {accel_cmd:+.2f}"
                pygame.display.set_caption(title)
                env.render()
                clock.tick(30)

        dur = time.time() - start_t
        print(f"Episode finished in {dur:.1f}s; Reward = {ep_reward:.2f}")

###############################################################################
# CLI
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Manual controller (continuous model)")
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
