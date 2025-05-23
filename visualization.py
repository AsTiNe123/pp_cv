import cv2
import numpy as np
from config import SCALED_SIZE, INFO_PANEL_WIDTH, TOTAL_WIDTH


class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.panel_bg_color = (40, 40, 40)
        self.text_color = (255, 255, 255)

    def create_display(self):
        """Создание комбинированного изображения (видео + панель сбоку)"""
        frame = np.zeros((SCALED_SIZE[1], TOTAL_WIDTH, 3), dtype=np.uint8)
        # Заливаем панель цветом фона
        frame[:, SCALED_SIZE[0]:] = self.panel_bg_color
        return frame

    def draw_rois(self, frame, rois):
        """Отрисовка ROI и подзон (только на видео части)"""
        video_part = frame[:, :SCALED_SIZE[0]]
        for roi in rois:
            cv2.polylines(video_part, [roi["polygon"]], True, roi["color"], 3)
            for lane in roi["lanes"]:
                cv2.polylines(video_part, [lane["polygon"]], True, lane["color"], 2)

    def draw_objects(self, frame, boxes, lane_colors):
        """Отрисовка обнаруженных объектов (только на видео части)"""
        video_part = frame[:, :SCALED_SIZE[0]]
        for box, color in zip(boxes, lane_colors):
            x_center, y_center, w, h = box
            x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
            x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
            cv2.rectangle(video_part, (x1, y1), (x2, y2), color, 3)

    def draw_info_panel(self, frame, rois, elapsed_time, fps):
        """Отрисовка информационной панели сбоку"""
        panel_x = SCALED_SIZE[0] + 10
        text_y = 30

        # Заголовок
        cv2.putText(frame, "Traffic Analytics", (panel_x, text_y),
                    self.font, 0.8, (255, 255, 255), 2)
        text_y += 40

        # Общая информация
        cv2.putText(frame, f"Time: {int(elapsed_time)}s", (panel_x, text_y),
                    self.font, 0.6, self.text_color, 1)
        text_y += 25
        cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x, text_y),
                    self.font, 0.6, self.text_color, 1)
        text_y += 40

        # Статистика по ROI
        for roi in rois:
            avg_speed = np.mean(roi["speeds"][-10:]) if roi["speeds"] else 0
            cv2.putText(frame, roi['name'], (panel_x, text_y),
                        self.font, 0.7, roi["color"], 1)
            text_y += 30

            cv2.putText(frame, f"Speed: {avg_speed:.1f} km/h", (panel_x + 20, text_y),
                        self.font, 0.5, self.text_color, 1)
            text_y += 25

            cv2.putText(frame, f"Flow: {roi['flow_rate']} veh/h", (panel_x + 20, text_y),
                        self.font, 0.5, self.text_color, 1)
            text_y += 25

            cv2.putText(frame, f"Current: {roi['object_count']}", (panel_x + 20, text_y),
                        self.font, 0.5, self.text_color, 1)
            text_y += 15

            # Статистика по полосам
            for lane in roi["lanes"]:
                lane_speed = np.mean(lane["speeds"][-5:]) if lane["speeds"] else 0
                cv2.putText(frame, f"- {lane['name']}:", (panel_x + 30, text_y),
                            self.font, 0.5, lane["color"], 1)
                text_y += 20

                cv2.putText(frame, f"  {lane_speed:.1f} km/h | {lane['flow_rate']} veh/h | {lane['object_count']}",
                            (panel_x + 40, text_y), self.font, 0.4, self.text_color, 1)
                text_y += 20

            text_y += 15