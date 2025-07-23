# parking_env_pkg/render.py
from __future__ import annotations
import math, pygame
from typing import List, Tuple, Sequence, Optional

from .utils import _normalize_angle, parking_corners

Vector = Tuple[float, float]

class PygameRenderer:
    """只做一件事：把传入的几何信息画到 Pygame 屏幕。"""

    def __init__(self, screen_size=(800, 800), scale=20):
        self.screen_size = screen_size
        self.scale = scale
        self.screen = None
        self.clock = pygame.time.Clock()

    def render(self, *,
               vehicle_poly,
               vehicle_state,
               target_info,
               obstacles: Sequence[Sequence[Vector]],
               lidar, lidar_max_range: float,
               step: int, max_steps: int,
               scenario_name: str,
               parking_length: float, parking_width: float):
        mode = "human"
        if self.screen is None and mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Parking Environment")

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                             and event.key == pygame.K_ESCAPE):
                pygame.quit(); self.screen = None; return

        cx, cy, cyaw, v, steer = vehicle_state
        w2s = self._world_to_screen
        self.screen.fill((255, 255, 255))

        # 1. obstacles
        for obs in obstacles:
            if len(obs) >= 3:
                pygame.draw.polygon(self.screen, (100, 100, 100),
                                    [w2s(cx, cy, cyaw, *p) for p in obs])
            elif len(obs) == 2:
                pygame.draw.line(self.screen, (100, 100, 100),
                                 w2s(cx, cy, cyaw, *obs[0]),
                                 w2s(cx, cy, cyaw, *obs[1]), 2)

        # 2. target slot
        tx, ty, tyaw = target_info
        pygame.draw.polygon(
            self.screen, (0, 200, 0),
            [w2s(cx, cy, cyaw, *p)
             for p in parking_corners(tx, ty, tyaw, parking_length, parking_width)],
            2,
        )

        # 3. lidar
        origin = (self.screen_size[0] // 2, self.screen_size[1] // 2)
        for ang, dist in zip(lidar.angles, lidar.ranges):
            lx, ly = dist * math.cos(ang), dist * math.sin(ang)
            end = (origin[0] + int(lx * self.scale),
                   origin[1] - int(ly * self.scale))
            col = dist / lidar_max_range
            pygame.draw.line(self.screen, (255 * (1 - col), 255 * col, 0),
                             origin, end, 1)
            if dist < lidar_max_range:
                pygame.draw.circle(self.screen, (0, 150, 0), end, 3)

        # 4. vehicle poly
        if vehicle_poly:
            pygame.draw.polygon(self.screen, (200, 0, 0),
                                [w2s(cx, cy, cyaw, *p)
                                 for p in vehicle_poly.exterior.coords], 2)

        # 6. vehicle heading line
        from shapely.geometry import Polygon
        if vehicle_poly and isinstance(vehicle_poly, Polygon):
            vehicle_center = vehicle_poly.centroid.coords[0]
            heading_vec = (math.cos(cyaw), math.sin(cyaw))
            head_len = parking_length * 0.5
            head_x = vehicle_center[0] + head_len * heading_vec[0]
            head_y = vehicle_center[1] + head_len * heading_vec[1]
            pygame.draw.line(self.screen, (0, 0, 255),
                             w2s(cx, cy, cyaw, *vehicle_center),
                             w2s(cx, cy, cyaw, head_x, head_y), 3)

        # 7. target slot heading line
        target_center = (tx, ty)
        target_vec = (math.cos(tyaw), math.sin(tyaw))
        head_len = parking_length * 0.5
        head_x = target_center[0] + head_len * target_vec[0]
        head_y = target_center[1] + head_len * target_vec[1]
        pygame.draw.line(self.screen, (0, 128, 255),
                         w2s(cx, cy, cyaw, *target_center),
                         w2s(cx, cy, cyaw, head_x, head_y), 2)

        # 5. HUD
        font = pygame.font.SysFont(None, 24)
        texts = [
            f"Speed: {v:.2f} m/s",
            f"Steer: {math.degrees(steer):.1f}°",
            f"Step: {step}/{max_steps}",
            f"Scenario: {scenario_name}",
        ]
        for i, t in enumerate(texts):
            self.screen.blit(font.render(t, True, (0, 0, 0)), (10, 10 + i * 22))
        pygame.display.flip()
        self.clock.tick(30)

    def _world_to_screen(self, cx, cy, cyaw, wx, wy):
        dx, dy = wx - cx, wy - cy
        cos_t, sin_t = math.cos(-cyaw), math.sin(-cyaw)
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        return (self.screen_size[0] // 2 + int(rx * self.scale),
                self.screen_size[1] // 2 - int(ry * self.scale))
