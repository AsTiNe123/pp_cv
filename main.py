import cv2
import time
from ultralytics import YOLO
from config import SCALED_SIZE
from roi_manager import ROIManager
from visualization import Visualizer
from utils import scale_points, estimate_speed
from config import ROI_CONFIG, PIXELS_PER_METER, FACTOR_DECLINIS

def main():
    # Инициализация
    model = YOLO("models/epoch7.pt")
    cap = cv2.VideoCapture("video/traffic3.mp4")
    roi_manager = ROIManager([{**roi, "points": scale_points(roi["points"])} 
                            for roi in ROI_CONFIG])

    visualizer = Visualizer()
    
    prev_time = time.time()
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Подготовка кадра
        frame = cv2.resize(frame, SCALED_SIZE)
        combined_frame = visualizer.create_display()
        combined_frame[:SCALED_SIZE[1], :] = frame
        
        # Расчет FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        elapsed_time = current_time - start_time

        # Сброс счетчиков
        roi_manager.update_counts()

        # Детекция и трекинг
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        lane_colors = []
        boxes = []

        if results[0].boxes.id is not None:
            det_boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(det_boxes, track_ids):
                roi, lane = roi_manager.process_detection(box, track_id, fps, current_time)
                if roi:
                    boxes.append(box)
                    lane_colors.append(lane["color"] if lane else roi["color"])

        # Обновление статистики
        roi_manager.update_flow_rates(current_time)

        # Отрисовка
        visualizer.draw_rois(frame, roi_manager.rois)
        visualizer.draw_objects(frame, boxes, lane_colors)
        visualizer.draw_info_panel(combined_frame, roi_manager.rois, elapsed_time, fps)

        cv2.imshow("Traffic Monitoring System", combined_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()