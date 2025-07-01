import pygame
import numpy as np
import math
import random
from enum import Enum
from vehicle import Vehicle

class ParkingEnv:
    """泊车环境类，支持强化学习交互"""
    
    class ParkingSpotType(Enum):
        """泊车位类型"""
        PARALLEL = 0  # 平行车位
        PERPENDICULAR = 1  # 垂直车位
    
    def __init__(self, render_mode=None, spot_type=ParkingSpotType.PERPENDICULAR):
        """
        初始化泊车环境
        
        :param render_mode: 渲染模式 ('human' 或 None)
        :param spot_type: 泊车位类型 (PARALLEL 或 PERPENDICULAR)
        """
        # 车辆参数
        self.vehicle_length = 5.0
        self.vehicle_width = 2.0
        
        # 创建车辆
        self.vehicle = Vehicle(length=self.vehicle_length, width=self.vehicle_width, num_radar=0)
        
        # 泊车位类型
        self.spot_type = spot_type
        
        # 泊车位尺寸
        self.spot_length = self.vehicle_length * 1.5  # 车位长度
        self.spot_width = self.vehicle_width * 1.8    # 车位宽度
        
        # 目标位置（泊车位中心）
        self.target_position = np.array([0.0, 0.0], dtype=np.float32)
        self.target_orientation = 0.0  # 目标朝向（弧度）
        
        # 初始位置（根据泊车位类型设置）
        self.initial_position = self._get_initial_position()
        
        # 环境边界
        self.bounds = {
            'x_min': -20.0,
            'x_max': 20.0,
            'y_min': -15.0,
            'y_max': 15.0
        }
        
        # 障碍物（车位边界）
        self.obstacles = self._create_obstacles()
        
        # 渲染相关
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.scale = 30.0  # 1米 = 30像素
        
        # 状态空间和动作空间
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()
        
        # 环境参数
        self.max_steps = 200  # 最大步数
        self.current_step = 0
        self.success_threshold = 0.2  # 成功阈值（米）
        
        # 颜色定义
        self.colors = {
            'background': (20, 30, 40),
            'vehicle': (70, 130, 180),
            'vehicle_outline': (30, 60, 100),
            'target': (0, 200, 100, 150),
            'obstacle': (180, 60, 60),
            'text': (220, 220, 220),
            'grid': (50, 60, 70)
        }
    
    def _create_action_space(self):
        """创建动作空间（连续）"""
        # 动作空间: [油门, 转向]
        # 油门: [-1, 1] 负值为倒车
        # 转向: [-1, 1] 负值为左转
        return {
            'low': np.array([-1.0, -1.0]),
            'high': np.array([1.0, 1.0]),
            'shape': (2,)
        }
    
    def _create_observation_space(self):
        """创建观察空间"""
        # 状态: [车辆x, 车辆y, 车辆朝向sin, 车辆朝向cos, 
        #        车辆速度, 转向角, 目标x, 目标y, 目标朝向sin, 目标朝向cos]
        return {
            'low': np.array([
                self.bounds['x_min'], self.bounds['y_min'], -1.0, -1.0,
                -10.0, -np.radians(30), 
                self.bounds['x_min'], self.bounds['y_min'], -1.0, -1.0
            ]),
            'high': np.array([
                self.bounds['x_max'], self.bounds['y_max'], 1.0, 1.0,
                10.0, np.radians(30),
                self.bounds['x_max'], self.bounds['y_max'], 1.0, 1.0
            ]),
            'shape': (10,)
        }
    
    def _get_initial_position(self):
        """根据泊车位类型生成初始位置"""
        if self.spot_type == self.ParkingSpotType.PARALLEL:
            # 平行车位：车辆在车位前方稍远位置，有一定角度偏移
            offset_x = random.uniform(0.5, 2.0) * self.vehicle_length
            offset_y = random.uniform(-1.0, 1.0) * self.vehicle_width
            angle_offset = random.uniform(-np.pi/6, np.pi/6)  # -30°到30°
            return np.array([
                self.target_position[0] + offset_x,
                self.target_position[1] + offset_y
            ]), angle_offset
        else:
            # 垂直车位：车辆在车位侧前方，有一定角度偏移
            offset_x = random.uniform(0.5, 2.0) * self.vehicle_width
            offset_y = random.uniform(0.5, 2.0) * self.vehicle_length
            angle_offset = random.uniform(np.pi/4, np.pi/2)  # 45°到90°
            return np.array([
                self.target_position[0] + offset_x,
                self.target_position[1] + offset_y
            ]), angle_offset
    
    def _create_obstacles(self):
        """创建泊车位障碍物"""
        obstacles = []
        
        # 车位边界
        half_length = self.spot_length / 2
        half_width = self.spot_width / 2
        
        # 车位左边界
        obstacles.append({
            'type': 'line',
            'start': [self.target_position[0] - half_length, self.target_position[1] - half_width],
            'end': [self.target_position[0] + half_length, self.target_position[1] - half_width]
        })
        
        # 车位右边界
        obstacles.append({
            'type': 'line',
            'start': [self.target_position[0] - half_length, self.target_position[1] + half_width],
            'end': [self.target_position[0] + half_length, self.target_position[1] + half_width]
        })
        
        # 车位后边界
        obstacles.append({
            'type': 'line',
            'start': [self.target_position[0] - half_length, self.target_position[1] - half_width],
            'end': [self.target_position[0] - half_length, self.target_position[1] + half_width]
        })
        
        # 如果是垂直车位，添加前边界（平行车位不需要）
        if self.spot_type == self.ParkingSpotType.PERPENDICULAR:
            obstacles.append({
                'type': 'line',
                'start': [self.target_position[0] + half_length, self.target_position[1] - half_width],
                'end': [self.target_position[0] + half_length, self.target_position[1] + half_width]
            })
        
        return obstacles
    
    def _get_state(self):
        """获取当前状态"""
        # 车辆状态
        vehicle_x, vehicle_y = self.vehicle.position
        orientation_sin = np.sin(self.vehicle.orientation)
        orientation_cos = np.cos(self.vehicle.orientation)
        
        # 目标状态
        target_x, target_y = self.target_position
        target_orientation_sin = np.sin(self.target_orientation)
        target_orientation_cos = np.cos(self.target_orientation)
        
        return np.array([
            vehicle_x, vehicle_y, 
            orientation_sin, orientation_cos,
            self.vehicle.velocity, 
            self.vehicle.steering_angle,
            target_x, target_y,
            target_orientation_sin, target_orientation_cos
        ], dtype=np.float32)
    
    def reset(self):
        """重置环境"""
        # 重置车辆状态
        self.vehicle.position = self.initial_position[0].copy()
        self.vehicle.orientation = self.initial_position[1]
        self.vehicle.velocity = 0.0
        self.vehicle.steering_angle = 0.0
        
        # 重置步数
        self.current_step = 0
        
        # 重新生成初始位置
        self.initial_position = self._get_initial_position()
        
        # 重新创建障碍物
        self.obstacles = self._create_obstacles()
        
        # 如果需要渲染，初始化pygame
        if self.render_mode == 'human' and self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("泊车环境")
            self.clock = pygame.time.Clock()
        
        return self._get_state()
    
    def step(self, action):
        """执行一个时间步"""
        # 解析动作
        throttle, steering = action
        
        # 更新车辆
        self.vehicle.update(throttle, steering)
        
        # 获取当前状态
        state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查是否终止
        done = self._is_done()
        
        # 增加步数
        self.current_step += 1
        
        return state, reward, done, {}
    
    def _calculate_reward(self):
        """计算奖励"""
        # 位置误差
        position_error = np.linalg.norm(self.vehicle.position - self.target_position)
        
        # 朝向误差
        orientation_error = abs(self.vehicle.orientation - self.target_orientation)
        orientation_error = min(orientation_error, 2*np.pi - orientation_error)
        
        # 基础奖励：鼓励接近目标
        reward = -position_error * 0.5 - orientation_error * 0.5
        
        # 成功奖励
        if position_error < self.success_threshold and orientation_error < np.radians(5):
            reward += 100.0
        
        # 碰撞惩罚
        if self._check_collision():
            reward -= 50.0
        
        # 边界惩罚
        if self._check_out_of_bounds():
            reward -= 50.0
        
        # 速度惩罚（鼓励低速）
        reward -= abs(self.vehicle.velocity) * 0.1
        
        # 转向惩罚（鼓励小转向）
        reward -= abs(self.vehicle.steering_angle) * 0.1
        
        return reward
    
    def _is_done(self):
        """检查是否终止"""
        # 成功条件
        position_error = np.linalg.norm(self.vehicle.position - self.target_position)
        orientation_error = abs(self.vehicle.orientation - self.target_orientation)
        orientation_error = min(orientation_error, 2*np.pi - orientation_error)
        
        if position_error < self.success_threshold and orientation_error < np.radians(5):
            return True
        
        # 碰撞或越界
        if self._check_collision() or self._check_out_of_bounds():
            return True
        
        # 步数超过限制
        if self.current_step >= self.max_steps:
            return True
        
        return False
    
    def _check_collision(self):
        """检查是否与障碍物碰撞"""
        # 获取车辆轮廓
        contour = self.vehicle.get_contour()
        
        # 检查车辆是否与障碍物相交
        for obstacle in self.obstacles:
            if obstacle['type'] == 'line':
                for i in range(4):
                    # 检查车辆每条边与障碍物线段是否相交
                    if self._line_intersection(
                        contour[i], contour[(i+1)%4],
                        np.array(obstacle['start']), np.array(obstacle['end'])
                    ):
                        return True
        return False
    
    def _line_intersection(self, a1, a2, b1, b2):
        """检查两条线段是否相交"""
        # 计算方向向量
        v1 = a2 - a1
        v2 = b2 - b1
        
        # 计算行列式
        det = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(det) < 1e-10:
            return False  # 平行
        
        # 计算参数
        t = ((b1[0] - a1[0]) * v2[1] - (b1[1] - a1[1]) * v2[0]) / det
        u = ((b1[0] - a1[0]) * v1[1] - (b1[1] - a1[1]) * v1[0]) / det
        
        # 检查交点是否在线段上
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def _check_out_of_bounds(self):
        """检查是否超出边界"""
        x, y = self.vehicle.position
        return (x < self.bounds['x_min'] or x > self.bounds['x_max'] or 
                y < self.bounds['y_min'] or y > self.bounds['y_max'])
    
    def render(self):
        """渲染环境"""
        if self.render_mode != 'human':
            return
        
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # 清屏
        self.screen.fill(self.colors['background'])
        
        # 计算屏幕中心（目标位置）
        screen_center_x = self.screen.get_width() // 2
        screen_center_y = self.screen.get_height() // 2
        
        # 转换函数：世界坐标 -> 屏幕坐标
        def world_to_screen(world_pos):
            return (int(screen_center_x + world_pos[0] * self.scale), 
                    int(screen_center_y - world_pos[1] * self.scale))  # 注意：y轴反向
        
        # 绘制网格
        grid_size = int(self.scale)  # 网格大小 (1米)
        for x in range(-10, 11):
            pygame.draw.line(
                self.screen, self.colors['grid'],
                world_to_screen([x, -15]), world_to_screen([x, 15]),
                1
            )
        for y in range(-15, 16):
            pygame.draw.line(
                self.screen, self.colors['grid'],
                world_to_screen([-20, y]), world_to_screen([20, y]),
                1
            )
        
        # 绘制坐标轴
        pygame.draw.line(
            self.screen, (100, 100, 100),
            world_to_screen([-20, 0]), world_to_screen([20, 0]),
            2
        )
        pygame.draw.line(
            self.screen, (100, 100, 100),
            world_to_screen([0, -15]), world_to_screen([0, 15]),
            2
        )
        
        # 绘制目标车位
        self._draw_parking_spot(world_to_screen)
        
        # 绘制车辆
        self._draw_vehicle(world_to_screen)
        
        # 绘制障碍物
        self._draw_obstacles(world_to_screen)
        
        # 绘制状态信息
        self._draw_info(world_to_screen)
        
        # 更新显示
        pygame.display.flip()
        
        # 控制帧率
        self.clock.tick(30)
    
    def _draw_parking_spot(self, world_to_screen):
        """绘制泊车位"""
        half_length = self.spot_length / 2
        half_width = self.spot_width / 2
        
        # 车位轮廓
        points = [
            [self.target_position[0] - half_length, self.target_position[1] - half_width],
            [self.target_position[0] + half_length, self.target_position[1] - half_width],
            [self.target_position[0] + half_length, self.target_position[1] + half_width],
            [self.target_position[0] - half_length, self.target_position[1] + half_width]
        ]
        
        # 转换为屏幕坐标
        screen_points = [world_to_screen(p) for p in points]
        
        # 绘制半透明车位
        parking_surface = pygame.Surface((800, 600), pygame.SRCALPHA)
        pygame.draw.polygon(parking_surface, self.colors['target'], screen_points)
        self.screen.blit(parking_surface, (0, 0))
        
        # 绘制车位边界
        pygame.draw.polygon(self.screen, (0, 180, 0), screen_points, 2)
        
        # 绘制目标点
        pygame.draw.circle(self.screen, (0, 255, 0), world_to_screen(self.target_position), 5)
        
        # 绘制目标朝向箭头
        arrow_length = 2.0
        arrow_end = self.target_position + arrow_length * np.array([
            np.cos(self.target_orientation), 
            np.sin(self.target_orientation)
        ])
        pygame.draw.line(
            self.screen, (0, 255, 0),
            world_to_screen(self.target_position),
            world_to_screen(arrow_end),
            2
        )
    
    def _draw_vehicle(self, world_to_screen):
        """绘制车辆"""
        # 绘制车辆轮廓
        contour = self.vehicle.get_contour()
        screen_points = [world_to_screen(point) for point in contour]
        pygame.draw.polygon(self.screen, self.colors['vehicle'], screen_points)
        pygame.draw.polygon(self.screen, self.colors['vehicle_outline'], screen_points, 2)
        
        # 绘制车辆中心点
        pygame.draw.circle(self.screen, (50, 150, 255), world_to_screen(self.vehicle.position), 6)
        
        # 绘制车辆朝向箭头
        arrow_length = self.vehicle.length * 0.8
        arrow_end = self.vehicle.position + arrow_length * np.array([
            np.cos(self.vehicle.orientation), 
            np.sin(self.vehicle.orientation)
        ])
        pygame.draw.line(
            self.screen, (255, 50, 50),
            world_to_screen(self.vehicle.position),
            world_to_screen(arrow_end),
            2
        )
    
    def _draw_obstacles(self, world_to_screen):
        """绘制障碍物"""
        for obstacle in self.obstacles:
            if obstacle['type'] == 'line':
                start_screen = world_to_screen(obstacle['start'])
                end_screen = world_to_screen(obstacle['end'])
                pygame.draw.line(self.screen, self.colors['obstacle'], start_screen, end_screen, 3)
    
    def _draw_info(self, world_to_screen):
        """绘制信息文本"""
        # 创建字体
        font = pygame.font.SysFont(None, 24)
        
        # 车辆位置
        pos_text = f"位置: ({self.vehicle.position[0]:.2f}, {self.vehicle.position[1]:.2f})"
        text_surface = font.render(pos_text, True, self.colors['text'])
        self.screen.blit(text_surface, (10, 10))
        
        # 车辆朝向
        deg_orientation = np.degrees(self.vehicle.orientation) % 360
        orientation_text = f"朝向: {deg_orientation:.1f}°"
        text_surface = font.render(orientation_text, True, self.colors['text'])
        self.screen.blit(text_surface, (10, 40))
        
        # 车辆速度
        speed_text = f"速度: {self.vehicle.velocity:.2f} m/s"
        text_surface = font.render(speed_text, True, self.colors['text'])
        self.screen.blit(text_surface, (10, 70))
        
        # 转向角
        deg_steering = np.degrees(self.vehicle.steering_angle)
        steering_text = f"转向角: {deg_steering:.1f}°"
        text_surface = font.render(steering_text, True, self.colors['text'])
        self.screen.blit(text_surface, (10, 100))
        
        # 步数
        step_text = f"步数: {self.current_step}/{self.max_steps}"
        text_surface = font.render(step_text, True, self.colors['text'])
        self.screen.blit(text_surface, (10, 130))
        
        # 目标位置
        target_text = f"目标位置: ({self.target_position[0]:.2f}, {self.target_position[1]:.2f})"
        text_surface = font.render(target_text, True, (0, 255, 0))
        self.screen.blit(text_surface, (10, 160))
        
        # 泊车位类型
        spot_type = "垂直车位" if self.spot_type == self.ParkingSpotType.PERPENDICULAR else "平行车位"
        type_text = f"泊车位类型: {spot_type}"
        text_surface = font.render(type_text, True, self.colors['text'])
        self.screen.blit(text_surface, (10, 190))
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None


# 测试环境
if __name__ == "__main__":
    # 创建环境（垂直车位）
    env = ParkingEnv(render_mode='human', spot_type=ParkingEnv.ParkingSpotType.PERPENDICULAR)
    
    # 重置环境
    state = env.reset()
    
    # 手动控制测试
    running = True
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 获取按键
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
            throttle = -0.5 if env.vehicle.velocity > 0 else 0.0
        
        # 重置
        if keys[pygame.K_r]:
            state = env.reset()
        
        # 执行动作
        action = np.array([throttle, steering], dtype=np.float32)
        state, reward, done, _ = env.step(action)
        
        # 渲染
        env.render()
        
        # 检查是否结束
        if done:
            print(f"Episode finished! Reward: {reward:.2f}")
            state = env.reset()
    
    # 关闭环境
    env.close()