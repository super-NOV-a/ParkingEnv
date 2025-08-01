import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import LineCollection

def calculate_vehicle_corners(length=5.0, width=2.0, rear_overhang=1.025):
    """
    计算车辆矩形四个角的坐标
    :param length: 车辆总长 (m)
    :param width: 车辆宽度 (m)
    :param rear_overhang: 后轴中心到车尾的长度 (m)
    :return: 车辆四个角点坐标数组
    """
    # 计算各部分尺寸
    front_overhang = length - rear_overhang  # 后轴中心到车头的长度
    
    # 车辆四个角点坐标（后轴中心在原点）
    corners = np.array([
        [-rear_overhang, -width/2],  # 后左
        [front_overhang, -width/2],   # 前左
        [front_overhang, width/2],    # 前右
        [-rear_overhang, width/2]     # 后右
    ])
    
    return corners

def line_segment_intersection(p1, p2, p3, p4):
    """
    计算两条线段的交点
    p1-p2: 第一条线段
    p3-p4: 第二条线段
    返回交点或None（如果无交点）
    """
    # 线段参数方程
    A = p2 - p1
    B = p4 - p3
    C = p3 - p1
    
    # 计算叉积
    cross_AB = np.cross(A, B)
    cross_AC = np.cross(A, C)
    cross_BC = np.cross(B, C)
    
    # 检查平行情况
    if abs(cross_AB) < 1e-10:
        return None
    
    # 计算参数t和u
    t = np.cross(C, B) / cross_AB
    u = np.cross(C, A) / cross_AB
    
    # 检查交点是否在线段上
    if 0 <= t <= 1 and 0 <= u <= 1:
        return p1 + t * A
    
    return None

def calculate_radar_intersections(vehicle_corners, num_rays=36):
    """计算雷达射线与车辆轮廓的交点"""
    radar_origin = np.array([0, 0])  # 雷达位于后轴中心
    intersections = []
    ray_directions = []
    
    # 创建封闭的车辆轮廓（添加第一个点使轮廓闭合）
    vehicle_contour = np.vstack([vehicle_corners, vehicle_corners[0]])
    
    # 生成等间隔的雷达射线角度
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    
    for angle in angles:
        # 计算射线方向向量
        direction = np.array([np.cos(angle), np.sin(angle)])
        ray_end = radar_origin + direction * 10  # 射线足够长
        
        closest_intersection = None
        min_distance = float('inf')
        
        # 检查射线与车辆每条边的交点
        for i in range(len(vehicle_contour) - 1):
            edge_start = vehicle_contour[i]
            edge_end = vehicle_contour[i + 1]
            
            intersection = line_segment_intersection(
                radar_origin, ray_end, edge_start, edge_end
            )
            
            if intersection is not None:
                # 计算交点距离雷达的距离
                distance = np.linalg.norm(intersection - radar_origin)
                
                # 保留最近的有效交点
                if distance < min_distance:
                    min_distance = distance
                    closest_intersection = intersection
        
        if closest_intersection is not None:
            intersections.append(closest_intersection)
            ray_directions.append(direction)
    
    return np.array(intersections), np.array(ray_directions)

def plot_vehicle_with_radar(num_rays=8, length=5.0, width=2.0, rear_overhang=1.025):
    """绘制车辆轮廓和雷达交点"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 计算车辆轮廓
    vehicle_corners = calculate_vehicle_corners(length, width, rear_overhang)
    
    # 绘制车辆矩形
    vehicle_rect = Polygon(vehicle_corners, closed=True, 
                          edgecolor='blue', facecolor='lightblue', alpha=0.7)
    ax.add_patch(vehicle_rect)
    
    # 标记后轴中心（雷达位置）
    ax.plot(0, 0, 'ro', markersize=8, label='Radar Position')
    
    # 计算雷达交点
    intersections, ray_directions = calculate_radar_intersections(
        vehicle_corners, num_rays
    )
    
    # 绘制雷达射线
    radar_origin = np.array([0, 0])
    ray_lines = []
    
    for i, direction in enumerate(ray_directions):
        # 射线从雷达中心到交点
        ray_line = np.vstack([radar_origin, intersections[i]])
        ray_lines.append(ray_line)
        
        # 标记交点
        ax.plot(intersections[i, 0], intersections[i, 1], 
                'go', markersize=8, alpha=0.7)
        
        # 添加角度标签
        angle = np.arctan2(direction[1], direction[0])
        label_pos = intersections[i] + direction * 0.3
        ax.text(label_pos[0], label_pos[1], 
                f'{np.rad2deg(angle):.0f}°', fontsize=9, ha='center')
    
    # 添加射线集合
    ray_collection = LineCollection(ray_lines, colors='red', 
                                   linewidths=1, alpha=0.5)
    ax.add_collection(ray_collection)
    
    # 绘制后轴中心到车尾的线
    rear_center = np.array([-rear_overhang, 0])
    ax.plot([0, rear_center[0]], [0, rear_center[1]], 
            'k--', linewidth=2, label=f'Rear Overhang ({rear_overhang}m)')
    ax.plot(rear_center[0], rear_center[1], 'ks', markersize=8, label='Rear Center')
    
    # 设置图形属性
    ax.set_title(f'Vehicle with 360° Radar ({num_rays} rays)\n'
                 f'Length: {length}m, Width: {width}m, Rear Overhang: {rear_overhang}m', 
                 fontsize=14)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整坐标轴范围以适应车辆尺寸
    buffer = 1.0  # 额外的显示空间
    x_min = min(-rear_overhang, min(intersections[:, 0])) - buffer
    x_max = max(length - rear_overhang, max(intersections[:, 0])) + buffer
    y_min = min(-width/2, min(intersections[:, 1])) - buffer
    y_max = max(width/2, max(intersections[:, 1])) + buffer
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # 添加图例
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# 测试不同雷达线数量
if __name__ == "__main__":
    # 车辆参数
    VEHICLE_LENGTH = 4.95
    VEHICLE_WIDTH = 2.0
    REAR_OVERHANG = 1.025  # 后轴中心到车尾的长度
    
    # 分别绘制4、8、16、36条雷达线的效果
    for num_rays in [36, 72, 144, 360]:
        plot_vehicle_with_radar(num_rays, VEHICLE_LENGTH, VEHICLE_WIDTH, REAR_OVERHANG)