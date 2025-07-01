import pygame
import numpy as np
import sys
import math

class Vehicle:
    """车辆类，包含自行车模型控制和雷达感知系统"""
    def __init__(self, length=5.0, width=2.0, num_radar=72):
        # 车辆尺寸 (米)
        self.length = length
        self.width = width
        
        # 车辆状态
        self.position = np.array([0.0, 0.0], dtype=np.float32)  # 后轴中心位置
        self.orientation = 0.0  # 朝向角度 (弧度)
        self.velocity = 0.0  # 速度 (米/秒)
        self.steering_angle = 0.0  # 前轮转向角 (弧度)
        
        # 自行车模型参数
        self.wheelbase = self.length * 0.6  # 轴距 (假设为车长的60%)
        self.max_steering = np.radians(30)  # 最大转向角 30度
        self.max_speed = 10.0  # 最大速度 (米/秒) 修改后
        self.acceleration = 2.0  # 加速度 (米/秒²) 修改后
        self.steering_rate = np.radians(90)  # 转向变化率 (度/秒) 修改后
        self.dt = 0.1  # 时间步长 (秒)
        
        # 雷达设置
        self.num_radar = num_radar
        self.radar_angles = np.linspace(0, 2*np.pi, num_radar, endpoint=False)
        
        # 存储交点信息
        self.intersection_points = []
        self.intersection_distances = []
        
        # 预计算交点
        self.calculate_intersections()
    
    def get_contour(self):
        """获取车辆轮廓的四个顶点（全局坐标系）"""
        half_length = self.length / 2
        half_width = self.width / 2
        
        # 局部坐标系下的四个顶点
        local_vertices = np.array([
            [-half_length, -half_width],  # 后左
            [half_length, -half_width],   # 前左
            [half_length, half_width],    # 前右
            [-half_length, half_width]    # 后右
        ])
        
        # 旋转矩阵
        rot_matrix = np.array([
            [np.cos(self.orientation), -np.sin(self.orientation)],
            [np.sin(self.orientation), np.cos(self.orientation)]
        ])
        
        # 旋转并平移顶点
        return np.dot(local_vertices, rot_matrix.T) + self.position
    
    def calculate_intersections(self):
        """计算所有雷达线与车身轮廓的交点"""
        self.intersection_points = []
        self.intersection_distances = []
        
        # 获取车辆轮廓
        contour = self.get_contour()
        
        # 创建闭合的轮廓线段 (4条边)
        edges = [
            (contour[0], contour[1]),  # 后到前左
            (contour[1], contour[2]),  # 前左到前右
            (contour[2], contour[3]),  # 前右到后右
            (contour[3], contour[0])   # 后右到后
        ]
        
        # 对每个雷达角度计算交点
        for angle in self.radar_angles:
            # 计算雷达线方向向量
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # 雷达线参数方程: P = position + t * direction, t > 0
            closest_intersection = None
            min_t = float('inf')
            
            # 检查与每条边的交点
            for edge in edges:
                # 解线段交点方程
                p1, p2 = edge
                edge_vec = p2 - p1
                
                # 构造线性方程组
                A = np.column_stack((-direction, edge_vec))
                b = self.position - p1
                
                try:
                    # 解方程 [t, s] = A⁻¹b
                    t, s = np.linalg.solve(A, b)
                    
                    # 检查交点是否在有效范围内
                    if t > 0 and 0 <= s <= 1:
                        if t < min_t:
                            min_t = t
                            closest_intersection = self.position + t * direction
                except np.linalg.LinAlgError:
                    # 无解或无穷解 (平行线)
                    continue
            
            if closest_intersection is not None:
                self.intersection_points.append(closest_intersection)
                self.intersection_distances.append(min_t)
            else:
                # 理论上应该总是有交点，但添加保护
                self.intersection_points.append(self.position)
                self.intersection_distances.append(0.0)
    
    def update(self, throttle, steering):
        """
        使用自行车模型更新车辆状态
        :param throttle: 油门控制 [-1, 1]，负值为刹车/倒车
        :param steering: 转向控制 [-1, 1]，负值为左转
        """
        # 1. 更新转向角 (限制在最大转向范围内)
        target_steering = steering * self.max_steering
        
        # 限制转向变化率
        steering_diff = target_steering - self.steering_angle
        max_steering_change = self.steering_rate * self.dt
        if abs(steering_diff) > max_steering_change:
            steering_change = np.sign(steering_diff) * max_steering_change
        else:
            steering_change = steering_diff
        
        self.steering_angle += steering_change
        self.steering_angle = np.clip(self.steering_angle, -self.max_steering, self.max_steering)
        
        # 2. 更新速度
        if throttle > 0:
            # 加速前进
            self.velocity = min(self.velocity + throttle * self.acceleration * self.dt, self.max_speed)
        elif throttle < 0:
            # 减速或倒车
            self.velocity = max(self.velocity + throttle * self.acceleration * self.dt, -self.max_speed)
        else:
            # 自然减速
            if self.velocity > 0:
                self.velocity = max(self.velocity - self.acceleration/2 * self.dt, 0)
            elif self.velocity < 0:
                self.velocity = min(self.velocity + self.acceleration/2 * self.dt, 0)
        
        # 3. 只有速度不为零时才更新位置和朝向
        if abs(self.velocity) > 0.01:
            # 计算角速度 (rad/s)
            angular_velocity = (self.velocity / self.wheelbase) * np.tan(self.steering_angle)
            
            # 更新朝向
            self.orientation += angular_velocity * self.dt
            self.orientation %= (2 * np.pi)  # 标准化角度到 [0, 2π]
            
            # 更新位置
            self.position[0] += self.velocity * np.cos(self.orientation) * self.dt
            self.position[1] += self.velocity * np.sin(self.orientation) * self.dt
        
        # 4. 更新雷达交点
        self.calculate_intersections()


if __name__=="__main__":
    # Pygame初始化
    pygame.init()
    width, height = 1000, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("交互式车辆模型 - 自行车模型与雷达感知")

    # 颜色定义
    BACKGROUND = (20, 30, 40)
    VEHICLE_COLOR = (70, 130, 180)  # 钢蓝色
    VEHICLE_OUTLINE = (30, 60, 100)
    RADAR_LINE_COLOR = (0, 200, 100, 100)  # 半透明绿色
    RADAR_POINT_COLOR = (255, 100, 100)   # 红色
    TEXT_COLOR = (220, 220, 220)
    GRID_COLOR = (50, 60, 70)

    # 创建车辆
    vehicle = Vehicle(length=5.0, width=2.0, num_radar=72)

    # 缩放因子 (1米 = 多少像素)
    scale = 50.0

    # 控制状态
    throttle = 0.0
    steering = 0.0

    # 字体
    font = pygame.font.SysFont(None, 28)
    title_font = pygame.font.SysFont(None, 36, bold=True)

    # 主游戏循环
    clock = pygame.time.Clock()
    running = True

    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 键盘控制
        keys = pygame.key.get_pressed()
        throttle = 0.0
        steering = 0.0
        
        # 油门/刹车控制
        if keys[pygame.K_UP]:
            throttle = 1.0
        if keys[pygame.K_DOWN]:
            throttle = -1.0
        
        # 转向控制
        if keys[pygame.K_LEFT]:
            steering = -1.0
        if keys[pygame.K_RIGHT]:
            steering = 1.0
        
        # 空格键刹车
        if keys[pygame.K_SPACE]:
            throttle = -0.5 if vehicle.velocity > 0 else 0.0
        
        # 重置车辆位置
        if keys[pygame.K_r]:
            vehicle.position = np.array([0.0, 0.0])
            vehicle.orientation = 0.0
            vehicle.velocity = 0.0
            vehicle.steering_angle = 0.0
        
        # 更新车辆状态
        vehicle.update(throttle, steering)
        
        # 清屏
        screen.fill(BACKGROUND)
        
        # 绘制网格
        grid_size = 50  # 网格大小 (像素)
        for x in range(0, width, grid_size):
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, height), 1)
        for y in range(0, height, grid_size):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (width, y), 1)
        
        # 绘制坐标原点
        origin_x, origin_y = width // 2, height // 2
        pygame.draw.circle(screen, (100, 100, 100), (origin_x, origin_y), 5)
        
        # 转换函数：世界坐标 -> 屏幕坐标
        def world_to_screen(world_pos):
            return (int(origin_x + world_pos[0] * scale), 
                    int(origin_y - world_pos[1] * scale))  # 注意：y轴反向
        
        # 绘制雷达线
        for i in range(vehicle.num_radar):
            start_screen = world_to_screen(vehicle.position)
            end_screen = world_to_screen(vehicle.intersection_points[i])
            
            # 使用半透明表面实现半透明效果
            radar_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            pygame.draw.line(radar_surface, RADAR_LINE_COLOR, start_screen, end_screen, 1)
            screen.blit(radar_surface, (0, 0))
            
            # 绘制交点
            pygame.draw.circle(screen, RADAR_POINT_COLOR, end_screen, 3)
        
        # 绘制车辆轮廓
        contour = vehicle.get_contour()
        screen_points = [world_to_screen(point) for point in contour]
        pygame.draw.polygon(screen, VEHICLE_COLOR, screen_points)
        pygame.draw.polygon(screen, VEHICLE_OUTLINE, screen_points, 2)
        
        # 绘制车辆朝向箭头
        arrow_length = vehicle.length * 0.8
        arrow_end = vehicle.position + arrow_length * np.array([
            np.cos(vehicle.orientation), 
            np.sin(vehicle.orientation)
        ])
        start_screen = world_to_screen(vehicle.position)
        end_screen = world_to_screen(arrow_end)
        
        # 计算箭头头部点
        arrow_size = 5
        angle = math.atan2(end_screen[1] - start_screen[1], end_screen[0] - start_screen[0])
        arrow_points = [
            end_screen,
            (end_screen[0] - arrow_size * math.cos(angle - math.pi/6), 
            end_screen[1] - arrow_size * math.sin(angle - math.pi/6)),
            (end_screen[0] - arrow_size * math.cos(angle + math.pi/6), 
            end_screen[1] - arrow_size * math.sin(angle + math.pi/6))
        ]
        pygame.draw.line(screen, (255, 50, 50), start_screen, end_screen, 3)
        pygame.draw.polygon(screen, (255, 50, 50), arrow_points)
        
        # 绘制车辆中心点
        pygame.draw.circle(screen, (50, 150, 255), world_to_screen(vehicle.position), 6)
        
        # 显示状态信息
        def draw_text(text, position, color=TEXT_COLOR, font=font):
            text_surface = font.render(text, True, color)
            screen.blit(text_surface, position)
        
        # 标题
        title = title_font.render("Interactive Vehicle Model - Bicycle Model with Radar Perception", True, (70, 180, 255))
        screen.blit(title, (width//2 - title.get_width()//2, 20))
        
        # 控制说明
        draw_text("Controls:", (20, 20))
        draw_text("Up: Accelerate", (20, 50))
        draw_text("Down: Brake/Reverse", (20, 80))
        draw_text("Left\Right: Steer", (20, 110))
        draw_text("Space: Emergency Brake", (20, 140))
        draw_text("R: Reset Position", (20, 170))
        
        # 车辆状态
        draw_text(f"Speed: {vehicle.velocity:.2f} m/s", (width - 250, 20))
        draw_text(f"Steering Angle: {np.degrees(vehicle.steering_angle):.1f}°", (width - 250, 50))
        draw_text(f"Orientation: {np.degrees(vehicle.orientation):.1f}°", (width - 250, 80))
        draw_text(f"Position: ({vehicle.position[0]:.2f}, {vehicle.position[1]:.2f})", (width - 250, 110))
        draw_text(f"Throttle: {throttle:.1f}, Steering: {steering:.1f}", (width - 250, 140))
        
        # 物理参数
        draw_text(f"Vehicle Size: {vehicle.length:.1f}m x {vehicle.width:.1f}m", (20, height - 100))
        draw_text(f"Max Speed: {vehicle.max_speed:.1f} m/s, Acceleration: {vehicle.acceleration:.1f} m/s²", (20, height - 70))
        draw_text(f"Number of Radar Lines: {vehicle.num_radar}", (20, height - 40))
        
        # 更新显示
        pygame.display.flip()
        
        # 控制帧率 (约10fps，对应0.1秒时间步长)
        clock.tick(10)

    # 退出Pygame
    pygame.quit()
    sys.exit()