import cv2
import numpy as np
from collections import deque
from config import SCALE_FACTOR, FLOW_RATE_WINDOW

def scale_points(points):
    """Масштабирование точек относительно коэффициента SCALE_FACTOR"""
    return [(int(x * SCALE_FACTOR), int(y * SCALE_FACTOR)) for x, y in points]

def is_point_in_polygon(point, polygon):
    """Проверка принадлежности точки полигону"""
    return cv2.pointPolygonTest(np.array(polygon), point, False) >= 0

def estimate_speed(p1, p2, fps, pixels_per_meter, factor_declinis):
    """Расчет скорости объекта"""
    distance_pixels = np.linalg.norm(np.array(p2) - np.array(p1))
    distance_meters = distance_pixels / pixels_per_meter * factor_declinis
    return distance_meters * fps * 3.6  # км/ч

def init_flow_history():
    """Инициализация истории для расчета интенсивности потока"""
    return deque(maxlen=FLOW_RATE_WINDOW)