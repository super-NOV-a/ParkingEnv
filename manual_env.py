"""Manual test harness for unified ParkingEnv
===========================================
Control the parking task with **keyboard** in either *continuous* or *discrete*
mode.

Keybindings (this script)
-------------------------
* **Continuous** (`--mode continuous`, default):
    * ← / →  : steer left / right (gradual)
    * ↑ / ↓  : throttle forward / reverse (analogue)
    * <space>: brake (throttle = 0)
    * R      : restart episode
    * Esc    : quit program

* **Discrete** (`--mode discrete`):
    * ↑      : set speed = Forward
    * ↓      : set speed = Backward
    * ← / →  : set steering = Left / Right
    * <space>: Stop **and** return steering to straight (park)
    
  You may press *two* keys at the same time – e.g. **↑ + ←** ⇒ *Forward‑Left*.
  Internally these states are encoded to the 9‑action table expected by
  :class:`parking_env.ParkingEnv`:

  | speed  | steer | action index |
  |--------|-------|--------------|
  | Fwd    | Left  | 0 |
  | Fwd    | Str.  | 1 |
  | Fwd    | Right | 2 |
  | Back   | Left  | 3 |
  | Back   | Str.  | 4 |
  | Back   | Right | 5 |
  | Stop   | Left  | 6 |
  | Stop   | Str.  | 7 |
  | Stop   | Right | 8 |

Usage
-----
```bash
python manual_test.py --mode continuous   # or discrete
```

After each episode the script prints duration and cumulative reward.
"""
from __future__ import annotations

import argparse
import time
from typing import Tuple

import numpy as np
import pygame

from parking_env import ParkingEnv

###############################################################################
# Environment factory
###############################################################################

def make_env(control_mode: str, render_mode: str) -> Tuple[ParkingEnv, dict]:
    cfg = {
        "control_mode": control_mode,   # continuous | discrete
        "data_dir": "C:\AI_Planner\RL\pygame_input_features_new_withinBEV_no_parallel_parking",
        "scenario_mode": "file",     # or 'file'
        "world_size": 30.0,
        "min_obstacles": 0,
        "max_obstacles": 5,
        "max_steps": 500,
        "timestep": 0.1,
        "lidar_max_range": 15.0,
        "render_mode": render_mode,
        "manual": True,
    }
    return ParkingEnv(cfg), cfg

###############################################################################
# Continuous‑control loop
###############################################################################

def run_continuous(env: ParkingEnv, cfg: dict) -> None:
    action = np.zeros(2, dtype=np.float32)  # [steer, throttle]
    steer_step = 0.1
    throttle_step = 0.1

    # Pygame setup
    if cfg["render_mode"] == "human":
        pygame.init()
        pygame.display.set_caption("ParkingEnv – continuous control")
        screen = pygame.display.set_mode(env.screen_size)
        clock = pygame.time.Clock()
    else:
        screen = None

    while True:
        obs, _ = env.reset()
        term = trunc = False
        total_reward = 0.0
        start_t = time.time()

        if screen:
            env.render()

        while not (term or trunc):
            if cfg["manual"] and screen:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        env.close(); pygame.quit(); return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        term = True  # early reset
                keys = pygame.key.get_pressed()

                # steering
                if keys[pygame.K_LEFT]:
                    action[0] = max(-1.0, action[0] + steer_step)
                elif keys[pygame.K_RIGHT]:
                    action[0] = min(1.0, action[0] - steer_step)
                else:
                    action[0] *= 0.9  # auto‑centering

                # throttle
                if keys[pygame.K_UP]:
                    action[1] = min(1.0, action[1] + 2 * throttle_step)
                elif keys[pygame.K_DOWN]:
                    action[1] = max(-1.0, action[1] - throttle_step)
                elif keys[pygame.K_SPACE]:
                    action[1] = 0.0  # brake
                else:
                    action[1] *= 0.9  # friction
            else:
                action = env.action_space.sample()

            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward

            if screen:
                # overlay steering/throttle text
                font = pygame.font.SysFont(None, 24)
                screen.blit(font.render(f"Steering: {action[0]:.2f}", True, (0, 0, 255)), (10, 240))
                screen.blit(font.render(f"Throttle: {action[1]:.2f}", True, (0, 0, 255)), (10, 270))
                pygame.display.flip()
                clock.tick(30)

        dur = time.time() - start_t
        print(f"Episode finished in {dur:.1f}s – reward {total_reward:.2f}")

###############################################################################
# Discrete‑control loop – new key scheme
###############################################################################

def encode_action(speed_state: int, steer_state: int) -> int:
    """Map *(speed, steer)* ∈ {‑1,0,1}² to action index 0‑8.
    speed_state:  1=Fwd, ‑1=Back, 0=Stop
    steer_state: ‑1=Left, 1=Right, 0=Straight
    Table row‑major order matches env expectation. """
    speed_row = {1: 0, -1: 1, 0: 2}[speed_state]
    steer_col = {-1: 0, 0: 1, 1: 2}[steer_state]
    return speed_row * 3 + steer_col


def run_discrete(env: ParkingEnv, cfg: dict) -> None:
    speed_state = 0   # 1=F, -1=B, 0=S
    steer_state = 0   # -1=L, 0=S, 1=R

    if cfg["render_mode"] == "human":
        pygame.init()
        pygame.display.set_caption("ParkingEnv – discrete control")
        screen = pygame.display.set_mode(env.screen_size)
        clock = pygame.time.Clock()
    else:
        screen = None

    running = True
    while running:
        obs, _ = env.reset()
        done = False
        start_t = time.time()
        total_reward = 0.0
        if screen:
            env.render()
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False; env.close(); pygame.quit(); return

            # Poll keys – allows pressing two keys simultaneously
            keys = pygame.key.get_pressed()

            # Speed control
            if keys[pygame.K_SPACE]:
                speed_state = 0
                steer_state = 0  # wheel to centre
            else:
                if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
                    speed_state = 1
                elif keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
                    speed_state = -1

                # Steering control
                # ─── 修改后（左键=左转，右键=右转） ─────────────────────────
                if keys[pygame.K_LEFT]:
                    steer_state = -1         # ← 左键 → 左转
                elif keys[pygame.K_RIGHT]:
                    steer_state = 1        # → 右键 → 右转


            # ── NEW: 数字键 (1-9) 直接指定离散动作 ──────────────────────
            digit_action = None
            for n in range(1, 10):                           # 键 1-9
                if keys[getattr(pygame, f'K_{n}')]:
                    digit_action = n - 1                     # 0-8 索引
                    break                                   # 同时按多个时取最小编号

            # 若按了数字键，则覆盖方向键组合
            if digit_action is not None:
                action = digit_action
            else:
                action = encode_action(speed_state, steer_state)
                speed_state, steer_state = 0, 0

            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            done = term or trunc

            if done:
                dur = time.time() - start_t
                print(f"Episode finished in {dur:.1f}s – reward {total_reward:.2f}")

            if screen:
                # Heads‑up display of current discrete state
                font = pygame.font.SysFont(None, 24)
                lbl = f"Speed: {speed_state:+d}  Steer: {steer_state:+d}  Action: {action}"
                screen.blit(font.render(lbl, True, (0, 0, 255)), (10, 240))
                pygame.display.flip(); clock.tick(30)

###############################################################################
# CLI
###############################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manual tester for ParkingEnv")
    p.add_argument("--mode", choices=["continuous", "discrete"], default="continuous",
                   help="测试时选择控制方式 连续还是离散")
    p.add_argument("--render", choices=["human", "none"], default="human",
                   help="rendering mode")
    return p.parse_args()

###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    args = parse_args()
    env, cfg = make_env(args.mode, args.render)
    try:
        if args.mode == "continuous":
            run_continuous(env, cfg)
        else:
            run_discrete(env, cfg)
    finally:
        env.close()
        if cfg["render_mode"] == "human":
            pygame.quit()
