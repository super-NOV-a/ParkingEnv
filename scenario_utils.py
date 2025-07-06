# scenario_utils.py
import json
import numpy as np
from shapely.geometry import Point, LineString, Polygon


def load_json_feature_no_dilation(filename, interp_dist=0.1):
    """
    从 JSON 中读取原始 polyline 点，做等距插值，
    返回 [ [x0, y0], [x1, y1], ... ] 格式的列表。
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    # 假设路径在 data['Frames']['0']['PathNodes']
    raw = data['Frames']['0']['PathNodes']  # 例如 [[x,y],...]
    pts = np.array(raw)
    # 计算累积弧长
    deltas = np.diff(pts, axis=0)
    seg_lens = np.hypot(deltas[:, 0], deltas[:, 1])
    cumlen = np.insert(np.cumsum(seg_lens), 0, 0)

    # 插值
    total = cumlen[-1]
    distances = np.arange(0, total, interp_dist)
    xs = np.interp(distances, cumlen, pts[:, 0])
    ys = np.interp(distances, cumlen, pts[:, 1])
    return list(map(tuple, np.stack([xs, ys], axis=1)))


def poly_to_shapely(poly_pts):
    """多段线 pts → Shapely LineString（或 Polygon）"""
    if len(poly_pts) < 3:
        return LineString(poly_pts)
    else:
        # 如果闭合，则做 Polygon
        return Polygon(poly_pts) if poly_pts[0] == poly_pts[-1] else LineString(poly_pts)
