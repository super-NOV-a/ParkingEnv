#!/usr/bin/env python3
# heatmap_hexbin.py
"""
Plot φ using random sampling + hexbin (concise).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)

K_DIST = 0.1  # 与主程序保持一致

def calc_phi(dist, yaw_err):
    phi_dist = 0.5 * np.exp(-K_DIST * dist)
    phi_yaw  = 0.5 * (1 - yaw_err / math.pi) * (dist <= 8.0)
    return phi_dist + phi_yaw

# -------- sample -----------------------------------------------------------
N = 50_0000
dist = rng.uniform(0, 30, N)
yaw  = rng.uniform(0, math.pi, N)
phi  = calc_phi(dist, yaw)

# -------- plot (hexbin) ----------------------------------------------------
plt.figure(figsize=(8, 4))
hb = plt.hexbin(dist, np.degrees(yaw), C=phi, gridsize=120, cmap='jet',
                reduce_C_function=np.mean)
plt.colorbar(hb, label='φ')
plt.xlabel('Distance (m)')
plt.ylabel('|Δψ| (deg)')
plt.title('Shaping potential φ — hexbin version')
plt.tight_layout()
plt.show()
