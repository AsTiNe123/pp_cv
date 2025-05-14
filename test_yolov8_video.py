from ultralytics import YOLO
import cv2

# Загрузка обученной модели
model = YOLO('epoch7.pt')  # Укажите путь к вашей модели

# Открытие видеофайла
video_path = 'video/slow_traffic_small.mp4'  # Укажите путь к видео
cap = cv2.VideoCapture(video_path)

# Получение параметров видео (ширина, высота, FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Создание VideoWriter для сохранения результата
output_path = 'video/output_video.mp4'  # Укажите путь для сохранения
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Обработка видео кадр за кадром
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Применение модели к кадру
    results = model(frame, conf = 0.25)

    # Визуализация результатов
    annotated_frame = results[0].plot()

    # Сохранение кадра с детекциями
    out.write(annotated_frame)

    # Отображение результата в реальном времени (опционально)
    cv2.imshow('YOLOv8 Inference', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Результат сохранён в: {output_path}")