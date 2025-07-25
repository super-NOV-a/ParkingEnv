from arc_segment_tracker import sample_arc, SegmentTracker, reconstruct_reference_path
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams ["axes.unicode_minus"]=False
Point = Tuple[float, float]
Action = Tuple[float, float]  # (delta, arc_length)


def generate_infinity_actions(cycles=2) -> List[Action]:
    """
    构造一个 8 字形轨迹动作序列。
    每一圈由：左弯 → 右弯 组成。
    """
    actions = []
    left = (np.deg2rad(30), 1)   # 左转半圈
    right = (np.deg2rad(-30), 1) # 右转半圈
    for _ in range(cycles):
        actions.extend([left])
    for _ in range(2*cycles):
        actions.extend([right])
    for _ in range(cycles):
        actions.extend([left])
    return actions

def test_infinity_tracking():
    wheelbase = 2.5
    start_pose = (0.0, 0.0, 0.0)
    actions = generate_infinity_actions(cycles=12)

    # 参考轨迹（用于绘制）
    ref_path = reconstruct_reference_path(start_pose, actions, wheelbase)
    ref_x, ref_y = zip(*ref_path)

    # 初始化 SegmentTracker
    tracker = SegmentTracker(wheelbase=wheelbase, dt=0.1, segment_time=1.0)
    tracker.reset(start_pose)
    tracker.set_plan(actions)
    traj = tracker.run()
    x, y, yaw = zip(*traj)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(ref_x, ref_y, 'b--', label="参考轨迹")
    plt.plot(x, y, 'r-', label="执行轨迹")
    plt.quiver(x[::10], y[::10], np.cos(yaw[::10]), np.sin(yaw[::10]), 
               scale=10, width=0.005, color='gray')
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("∞ 字轨迹跟踪 (离散弧长 + SegmentTracker)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_infinity_tracking()
