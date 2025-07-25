import math
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams ["axes.unicode_minus"]=False
Point = Tuple[float, float]
Action = Tuple[float, float]  # (delta, arc_length)

def sample_arc(x, y, yaw, delta, s, wheelbase, ds=0.05) -> Tuple[List[Point], float]:
    """根据固定 steering delta 和 arc_length s 采样轨迹点"""
    pts = []
    steps = max(2, int(abs(s) / ds))
    sign = 1 if s >= 0 else -1
    kappa = math.tan(delta) / wheelbase
    for i in range(1, steps + 1):
        ds_i = sign * ds * i
        if abs(kappa) < 1e-6:
            xi = x + ds_i * math.cos(yaw)
            yi = y + ds_i * math.sin(yaw)
            yaw_i = yaw
        else:
            R = 1 / kappa
            d_yaw = ds_i * kappa
            xi = x + R * (math.sin(yaw + d_yaw) - math.sin(yaw))
            yi = y - R * (math.cos(yaw + d_yaw) - math.cos(yaw))
            yaw_i = yaw + d_yaw
        pts.append((xi, yi))
    return pts, yaw_i

class SegmentTracker:
    """每段轨迹用恒定速度 + 固定 steering，完整跟踪器"""
    def __init__(self, wheelbase: float, dt: float = 0.1, segment_time: float = 1.0):
        self.wheelbase = wheelbase
        self.dt = dt
        self.segment_time = segment_time  # 每段轨迹执行时间
        self.actions: List[Action] = []
        self.pose = (0.0, 0.0, 0.0)
        self.traj = []

    def reset(self, pose: Tuple[float, float, float]):
        self.pose = pose
        self.traj = [pose]

    def set_plan(self, actions: List[Action]):
        self.actions = actions

    def run(self) -> List[Tuple[float, float, float]]:
        """顺序执行所有动作段，返回完整轨迹"""
        x, y, yaw = self.pose
        for delta, s in self.actions:
            v = s / self.segment_time  # 弧长 / 时间
            steps = int(self.segment_time / self.dt)
            for _ in range(steps):
                x += v * math.cos(yaw) * self.dt
                y += v * math.sin(yaw) * self.dt
                yaw += (v / self.wheelbase) * math.tan(delta) * self.dt
                yaw = math.atan2(math.sin(yaw), math.cos(yaw))
                self.traj.append((x, y, yaw))
        return self.traj


def generate_random_actions(num_actions: int = 10) -> List[Action]:
    steer_choices = np.deg2rad([-30, -20, -10, 0, 10, 20, 30])
    arc_choices = np.array([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
    actions = []
    for _ in range(num_actions):
        delta = np.random.choice(steer_choices)
        arc_length = np.random.choice(arc_choices)
        actions.append((delta, arc_length))
    return actions

def reconstruct_reference_path(start_pose: Tuple[float, float, float], actions: List[Action], wheelbase: float) -> List[Point]:
    x, y, yaw = start_pose
    ref_path = []
    for delta, s in actions:
        pts, yaw = sample_arc(x, y, yaw, delta, s, wheelbase)
        ref_path.extend(pts)
        x, y = pts[-1]
    return ref_path

def simulate_random_tracking():
    wheelbase = 2.5
    start_pose = (0.0, 0.0, 0.0)
    actions = generate_random_actions(10)

    # 构造参考路径
    ref_path = reconstruct_reference_path(start_pose, actions, wheelbase)
    ref_x, ref_y = zip(*ref_path)

    # 执行跟踪
    tracker = SegmentTracker(wheelbase=wheelbase, dt=0.1, segment_time=1.0)
    tracker.reset(start_pose)
    tracker.set_plan(actions)
    traj = tracker.run()

    x, y, yaw = zip(*traj)

    # 提取每步动作（delta, velocity）
    deltas = []
    speeds = []
    for i in range(1, len(traj)):
        x0, y0, yaw0 = traj[i - 1]
        x1, y1, yaw1 = traj[i]
        dx = x1 - x0
        dy = y1 - y0
        
        v_unsigned = math.hypot(dx, dy) / tracker.dt
        theta = math.atan2(dy, dx)
        angle_diff = math.atan2(math.sin(theta - yaw0), math.cos(theta - yaw0))
        sign = 1 if abs(angle_diff) < math.pi / 2 else -1
        v = sign * v_unsigned

        dyaw = yaw1 - yaw0
        delta = math.atan2(tracker.wheelbase * dyaw / tracker.dt, v) if v != 0 else 0
        deltas.append(delta)
        speeds.append(v)

    # Plot setup
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Trajectory comparison
    axs[0, 0].plot(ref_x, ref_y, 'b--', label='参考轨迹 (sample_arc)')
    axs[0, 0].plot(x, y, 'r-', label='执行轨迹 (SegmentTracker)')
    axs[0, 0].quiver(x[::5], y[::5], np.cos(yaw[::5]), np.sin(yaw[::5]),
                     angles='xy', scale_units='xy', scale=5, color='gray', width=0.005)
    axs[0, 0].scatter(x[0], y[0], c='green', label='起点')
    axs[0, 0].scatter(x[-1], y[-1], c='blue', label='终点')
    axs[0, 0].axis('equal')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 0].set_title("轨迹对比")
    axs[0, 0].set_xlabel("X")
    axs[0, 0].set_ylabel("Y")

    # Steering angle over time
    axs[0, 1].plot(np.rad2deg(deltas))
    axs[0, 1].set_title("每步转角变化 (deg)")
    axs[0, 1].set_xlabel("时间步")
    axs[0, 1].set_ylabel("转角 (°)")
    axs[0, 1].grid(True)

    # Speed over time
    axs[1, 0].plot(speeds)
    axs[1, 0].set_title("每步速度变化 (m/s)")
    axs[1, 0].set_xlabel("时间步")
    axs[1, 0].set_ylabel("速度")
    axs[1, 0].grid(True)

    # Empty or customizable
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("segment_tracking_comparison.png")
    plt.show()

if __name__ == "__main__":
    simulate_random_tracking()
