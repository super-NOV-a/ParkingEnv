from typing import Optional
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import multiprocessing


def make_env(rank: int, render_mode=None, env_name="point"):
    def _init():
        if env_name == "point":
            from PointEnv import PointEnv
            env = PointEnv(render_mode=render_mode if rank == 0 else None)

        elif env_name == "vel_point":
            from vel_PointEnv import PointEnv
            env = PointEnv(render_mode=render_mode if rank == 0 else None)

        elif env_name == "car":
            from CarEnv import CarEnv
            env = CarEnv(render_mode=render_mode if rank == 0 else None)

        elif env_name == "jit_car":
            from jit_CarEnv import CarEnv
            env = CarEnv(render_mode=render_mode if rank == 0 else None)

        elif env_name == "lidar_car":
            from parkingEnv import CarEnv
            env = CarEnv(render_mode=render_mode if rank == 0 else None)

        else:
            raise AssertionError("--parallel_wrap.py中没找到该环境")

        return env
    return _init


def make_parallel_envs(num_envs: int = 4, render_mode: Optional[str] = None, env_name="point"):
    # 解决 Windows 多进程问题
    if __name__ == "__main__" or __name__ == "parallel_wrap":
        # 单环境创建函数
        def make_env_fn(rank: int):
            return make_env(rank, render_mode, env_name)

        # 选择并行化方式
        if num_envs > 1:
            return SubprocVecEnv([make_env_fn(i) for i in range(num_envs)])
        return DummyVecEnv([make_env_fn(0)])