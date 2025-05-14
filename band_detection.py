from collections import defaultdict
import numpy as np
from ultralytics import YOLO
import cv2
import time
import math

PIXELS_PER_METER = 5


def estimate_speed(p1, p2, fps):
    """
    p1, p2: центры объекта в двух последовательных кадрах (в пикселях).
    fps: кадры в секунду.
    ppm: пикселей на метр (из калибровки).
    """
    distance_pixels = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    distance_meters = distance_pixels / PIXELS_PER_METER
    speed_mps = distance_meters * fps  # метры в секунду
    speed_kph = speed_mps * 3.6  # км/ч
    return speed_kph


# Настройки полос
LANE_BORDERS = [((100, 100), (150, 150)),
                ((100, 150), (150, 200)),
                ((100, 200), (150, 250))]  # Границы полос (по оси Y)
LANE_COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Цвета для каждой полосы

LANE_CENTERS = [(125, 125),
                (125, 175),
                (125, 225)]

frame_wid = 640
frame_hyt = 480

video_path = "video/traffic_flow.mp4"  # Убедитесь, что путь правильный
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Ошибка загрузки видео")
    exit()

# Словарь для хранения данных по полосам
lane_data = defaultdict(lambda: {
    'track_history': {},
    'speeds': []
})

model = YOLO("epoch7.pt", "v8")

track_history = {}
frame_count = 0
prev_time = time.time()

# Основной цикл обработки видео
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    annotated_frame = results[0].plot()
    FPS = int(1 / (time.time() - prev_time))

    # Рисуем границы полос
    for lane in LANE_BORDERS:

        cv2.line(annotated_frame, lane[0], lane[1], (0, 0, 255), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            center = (float(x), float(y))

            # Определяем полосу по Y-координате центра
            lane_id = np.digitize((x, y), LANE_CENTERS)  # Возвращает номер полосы (0, 1, 2...)
            lane_color = LANE_COLORS[lane_id % len(LANE_COLORS)]

            # Сохраняем историю позиций для полосы
            if track_id not in lane_data[lane_id]['track_history']:
                lane_data[lane_id]['track_history'][track_id] = []
            lane_data[lane_id]['track_history'][track_id].append(center)

            # Расчет скорости
            speed = 0
            if len(lane_data[lane_id]['track_history'][track_id]) > 1:
                prev_pos = lane_data[lane_id]['track_history'][track_id][-2]
                curr_pos = lane_data[lane_id]['track_history'][track_id][-1]
                speed = estimate_speed(prev_pos, curr_pos, FPS)
                lane_data[lane_id]['speeds'].append(speed)

            # Рисуем bounding box и скорость (цвет = полоса)
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                lane_color,
                2,
            )
            cv2.putText(
                annotated_frame,
                f"{track_id} {speed:.1f} км/ч",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                lane_color,
                2,
            )

    # Вывод средней скорости по полосам
    for i, y in enumerate(LANE_BORDERS):
        if i < len(LANE_BORDERS) - 1:
            lane_speeds = lane_data[i]['speeds']
            avg_speed = np.mean(lane_speeds) if lane_speeds else 0
            cv2.putText(
                annotated_frame,
                f"Lane {i+1}: {avg_speed:.1f} км/ч",
                (10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                LANE_COLORS[i],
                2,
            )

    cv2.imshow("Car Speed Estimation", annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break
    prev_time = time.time()
    # time.sleep(0.5)