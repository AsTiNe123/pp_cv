import cv2
import numpy as np
from config import SCALED_SIZE, INFO_PANEL_HEIGHT, TOTAL_HEIGHT

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def create_display(self):
        """Создание комбинированного изображения (видео + панель)"""
        frame = np.zeros((TOTAL_HEIGHT, SCALED_SIZE[0], 3), dtype=np.uint8)
        return frame

    def draw_rois(self, frame, rois):
        """Отрисовка ROI и подзон"""
        for roi in rois:
            cv2.polylines(frame, [roi["polygon"]], True, roi["color"], 3)
            for lane in roi["lanes"]:
                cv2.polylines(frame, [lane["polygon"]], True, lane["color"], 2)

    def draw_objects(self, frame, boxes, lane_colors):
        """Отрисовка обнаруженных объектов"""
        for box, color in zip(boxes, lane_colors):
            x_center, y_center, w, h = box
            x1, y1 = int(x_center - w/2), int(y_center - h/2)
            x2, y2 = int(x_center + w/2), int(y_center + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    def draw_info_panel(self, frame, rois, elapsed_time, fps):
        """Отрисовка информационной панели"""
        panel_y = SCALED_SIZE[1] + 10
        cv2.rectangle(frame, (0, SCALED_SIZE[1]), (SCALED_SIZE[0], TOTAL_HEIGHT), (50, 50, 50), -1)
        
        # Заголовок
        cv2.putText(frame, "Traffic Analytics:", (10, panel_y), 
                   self.font, 0.7, (255, 255, 255), 2)
        panel_y += 30

        # Статистика
        for roi in rois:
            avg_speed = np.mean(roi["speeds"][-10:]) if roi["speeds"] else 0
            cv2.putText(frame, 
                       f"{roi['name']}: {avg_speed:.1f} km/h | Flow: {roi['flow_rate']} veh/h | Current: {roi['object_count']}",
                       (10, panel_y), self.font, 0.6, roi["color"], 1)
            panel_y += 25

            for lane in roi["lanes"]:
                lane_speed = np.mean(lane["speeds"][-5:]) if lane["speeds"] else 0
                cv2.putText(frame, 
                           f"  {lane['name']}: {lane_speed:.1f} km/h | Flow: {lane['flow_rate']} veh/h | Current: {lane['object_count']}",
                           (20, panel_y), self.font, 0.5, lane["color"], 1)
                panel_y += 20
            panel_y += 10

        # Время и FPS
        cv2.putText(frame, f"Elapsed time: {int(elapsed_time)}s | FPS: {fps:.1f}", 
                   (10, TOTAL_HEIGHT - 10), self.font, 0.5, (255, 255, 255), 1)