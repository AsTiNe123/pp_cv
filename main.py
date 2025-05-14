from ultralytics import YOLO
import cv2
import math
import time
import numpy as np

# Настройки
ROI_LIST = [
    {
        "name": "ROI 1",
        "points": [(0, 350), (320, 350), (225, 480), (0, 480)],
        "color": (0, 255, 0)
    },
    {
        "name": "ROI 2",
        "points": [(320, 350), (225, 480), (640, 480), (640, 350)],
        "color": (0, 0, 255)
    }
]  # Список ROI с полигонами

PIXELS_PER_METER = 3
FACTOR_DECLINIS = 5


def estimate_speed(p1, p2, fps):
    distance_pixels = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    distance_meters = distance_pixels / PIXELS_PER_METER
    distance_meters *= FACTOR_DECLINIS
    return distance_meters * fps * 3.6  # км/ч


def is_point_in_polygon(point, polygon):
    """Проверяет, находится ли точка внутри полигона"""
    return cv2.pointPolygonTest(np.array(polygon), point, False) >= 0


model = YOLO("epoch7.pt")
cap = cv2.VideoCapture("video/traffic3.mp4")

# Подготовка данных ROI
for roi in ROI_LIST:
    roi["polygon"] = np.array(roi["points"], dtype=np.int32)
    roi["track_history"] = {}
    roi["speeds"] = []

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Рисуем все ROI (полигоны)
    for roi in ROI_LIST:
        cv2.polylines(frame, [roi["polygon"]], True, roi["color"], 2)
        cv2.putText(frame, roi["name"], tuple(roi["points"][0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi["color"], 2)

    # Детекция и трекинг
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x_center, y_center, w, h = box
            center_point = (float(x_center), float(y_center))
            x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
            x2, y2 = int(x_center + w / 2), int(y_center + h / 2)

            # Проверяем каждый ROI
            for roi in ROI_LIST:
                if is_point_in_polygon(center_point, roi["polygon"]):
                    # Инициализация истории для трека
                    if track_id not in roi["track_history"]:
                        roi["track_history"][track_id] = []

                    # Добавляем текущую позицию
                    roi["track_history"][track_id].append(center_point)

                    # Расчет скорости
                    speed = 0
                    if len(roi["track_history"][track_id]) > 1:
                        prev_pos = roi["track_history"][track_id][-2]
                        curr_pos = roi["track_history"][track_id][-1]
                        speed = estimate_speed(prev_pos, curr_pos, fps=fps)
                        roi["speeds"].append(speed)  # Сохраняем скорость для ROI

                    # Рисуем информацию для объекта
                    cv2.rectangle(frame, (x1, y1), (x2, y2), roi["color"], 2)
                    cv2.putText(frame, f"{speed:.1f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi["color"], 2)
                    break  # Выходим из цикла проверки ROI после первого совпадения

    # Отображаем статистику для каждого ROI
    stat_y = 30
    for roi in ROI_LIST:
        if roi["speeds"]:
            # Усредняем скорость за последние N измерений
            recent_speeds = roi["speeds"][-10:] if len(roi["speeds"]) > 10 else roi["speeds"]
            avg_speed = np.mean(recent_speeds)

            cv2.putText(frame, f"{roi['name']}: {avg_speed:.1f} km/h",
                        (10, stat_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi["color"], 2)
            stat_y += 30

    cv2.imshow("Polygon ROI Speed Estimation", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()