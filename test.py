from ultralytics import YOLO
import cv2
import time
import math


PIXELS_PER_METER = 25

def estimate_speed(p1, p2, fps):
    """
    p1, p2: центры объекта в двух последовательных кадрах (в пикселях).
    fps: кадры в секунду.
    ppm: пикселей на метр (из калибровки).
    """
    distance_pixels = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    distance_meters = distance_pixels / PIXELS_PER_METER
    speed_mps = distance_meters * fps  # метры в секунду
    speed_kph = speed_mps       # км/ч
    return speed_kph

# Загружаем модель
model = YOLO("yolov8s.pt")  # или ваша custom модель

# Открываем видео
video_path = "video/traffic_flow.mp4"
cap = cv2.VideoCapture(video_path)

# Словарь для хранения предыдущих позиций автомобилей
track_history = {}
frame_count = 0
prev_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    FPS = int(1 / (time.time() - prev_time))
    avr_speed = 0
    count = 0
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            center = (float(x), float(y))

            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append(center)

            speed = 0
            if len(track_history[track_id]) > 2:
                point1, point2 = track_history[track_id][-3], track_history[track_id][-1]
                speed = estimate_speed(point1, point2, FPS)
                avr_speed += speed
                count += 1

            # Рисуем на annotated_frame (чтобы было видно)
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"{track_id} {speed:.1f} m/s",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        avr_speed = avr_speed / count if count != 0 else 0

    cv2.putText(frame, f"FPS: {FPS}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"avr. speed: {avr_speed:.1f} m/s", (140, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Car Speed Estimation", frame)
    time.sleep(1)
    if cv2.waitKey(1) == ord('q'):
        break
    prev_time = time.time()

cap.release()
cv2.destroyAllWindows()