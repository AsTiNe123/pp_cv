from collections import defaultdict
from utils import init_flow_history, is_point_in_polygon
from config import PIXELS_PER_METER, FACTOR_DECLINIS
from numpy import array, int32

class ROIManager:
    def __init__(self, roi_config):
        self.rois = self._init_rois(roi_config)
        
    def _init_rois(self, config):
        """Инициализация структур данных для ROI"""
        rois = []
        for roi in config:
            roi_data = {
                **roi,
                "polygon": array(roi["points"], dtype=int32),
                "track_history": defaultdict(list),
                "speeds": [],
                "object_count": 0,
                "passed_objects": 0,
                "flow_rate": 0,
                "flow_history": init_flow_history()
            }
            
            # Инициализация подзон
            roi_data["lanes"] = [
                {
                    **lane,
                    "polygon": array(lane["points"], dtype=int32),
                    "track_history": defaultdict(list),
                    "speeds": [],
                    "object_count": 0,
                    "passed_objects": 0,
                    "flow_rate": 0,
                    "flow_history": init_flow_history()
                }
                for lane in roi["lanes"]
            ]
            rois.append(roi_data)
        return rois

    def update_counts(self):
        """Сброс счетчиков объектов перед обработкой нового кадра"""
        for roi in self.rois:
            roi["object_count"] = 0
            for lane in roi["lanes"]:
                lane["object_count"] = 0

    def update_flow_rates(self, current_time):
        """Обновление интенсивности потока"""
        for roi in self.rois:
            if roi["flow_history"]:
                time_window = max(1, current_time - roi["flow_history"][0])
                roi["flow_rate"] = int(len(roi["flow_history"]) / time_window * 3600)
            
            for lane in roi["lanes"]:
                if lane["flow_history"]:
                    time_window = max(1, current_time - lane["flow_history"][0])
                    lane["flow_rate"] = int(len(lane["flow_history"]) / time_window * 3600)

    def process_detection(self, box, track_id, fps, current_time):
        """Обработка обнаруженного объекта"""
        x_center, y_center, w, h = box
        center_point = (float(x_center), float(y_center))
        
        for roi in self.rois:
            if is_point_in_polygon(center_point, roi["polygon"]):
                roi["object_count"] += 1
                
                # Обновление истории трекинга
                if not roi["track_history"][track_id]:
                    roi["passed_objects"] += 1
                    roi["flow_history"].append(current_time)
                roi["track_history"][track_id].append(center_point)

                # Обработка подзон
                for lane in roi["lanes"]:
                    if is_point_in_polygon(center_point, lane["polygon"]):
                        lane["object_count"] += 1
                        
                        if not lane["track_history"][track_id]:
                            lane["passed_objects"] += 1
                            lane["flow_history"].append(current_time)
                        lane["track_history"][track_id].append(center_point)
                        return roi, lane
                return roi, None
        return None, None