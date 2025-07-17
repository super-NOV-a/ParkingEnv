import numpy as np

def parking_energy(x, y, yaw, cx, cy, cyaw):
    c1 = 10
    c2 = 20
    R_min = 5

    rou = np.sqrt((x - cx)**2 + (y - cy)**2)
    theta_l = np.arctan2(y - cy, x - cx)
    delta_theta = yaw - cyaw

    # 安全 tan 值
    tan_delta_half = np.tan(delta_theta / 2)
    tan_theta_l = np.tan(theta_l)

    # 避免除以零：使用 np.where 做安全除法
    safe_tan_theta_l = np.where(tan_theta_l == 0, np.inf, tan_theta_l)

    distance_1 = rou * np.cos(theta_l) * (np.tan(theta_l) - tan_delta_half)
    distance_2 = rou * np.sin(theta_l) * (1 - tan_delta_half / safe_tan_theta_l)

    # 使用 np.where 替代 if-else
    difference_distance = np.where(
        np.sin(theta_l) == 0,
        distance_1,
        np.where(
            np.cos(theta_l) == 0,
            distance_2,
            (distance_1 + distance_2) / 2
        )
    )

    driving_distance = np.abs(rou * difference_distance)

    position_distance = rou + np.abs(delta_theta) * 1
    energy_distance = c1 / (1 + position_distance) * np.cos(delta_theta)
    # energy_distance = c1 / (1 + position_distance)

    R1 = rou / 2 * np.sin(delta_theta / 2)
    R2 = rou / 2 * np.sin(theta_l)
    R_difference = (R1**2 + R2**2) - R_min**2

    k = 20
    energy_difference = c2 / (1 + driving_distance) * (1 / (1 + np.exp(-k * (R_difference - 1)))) * np.cos(delta_theta)



    return energy_distance * 9 + energy_difference


def get_energy(all_nodes, pos):
    energy = 0
    for node in all_nodes:
        if node.depth == 0:
            weight = 3
        elif node.depth < 3:
            weight = 1 * (0.7 ** node.depth)
        else:
            weight = 1 * (0.5 ** node.depth)
        energy += weight * parking_energy(pos[0], pos[1], pos[2], node.x, node.y, node.yaw)
    return energy