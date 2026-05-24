from dataclasses import dataclass
from typing import Optional

import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    track_id: Optional[int]
    bbox_xyxy: np.ndarray
    confidence: float
    class_id: int
    class_name: str

    @property
    def area(self) -> float:
        w = self.bbox_xyxy[2] - self.bbox_xyxy[0]
        h = self.bbox_xyxy[3] - self.bbox_xyxy[1]
        return float(w * h)

    @property
    def bottom_y(self) -> float:
        return float(self.bbox_xyxy[3])

    @property
    def center(self) -> tuple:
        cx = (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2
        cy = (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2
        return (float(cx), float(cy))


class CarDetector:
    def __init__(self, model_name: str = "yolov8s.pt",
                 confidence_threshold: float = 0.3,
                 device: str = "cpu"):
        self.model = YOLO(model_name)
        self.conf_threshold = confidence_threshold
        self.device = device

    def detect_and_track(self, frame: np.ndarray,
                         persist: bool = True) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=persist,
            conf=self.conf_threshold,
            tracker="botsort.yaml",
            classes=list(VEHICLE_CLASSES.keys()),
            device=self.device,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        detections = []

        has_ids = boxes.id is not None

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            if cls_id not in VEHICLE_CLASSES:
                continue

            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i])
            track_id = int(boxes.id[i]) if has_ids else None

            det = Detection(
                track_id=track_id,
                bbox_xyxy=xyxy,
                confidence=conf,
                class_id=cls_id,
                class_name=VEHICLE_CLASSES[cls_id],
            )

            if det.area < 500:
                continue

            detections.append(det)

        return detections

    def reset_tracker(self):
        self.model.predictor = None
