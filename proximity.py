from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from detector import Detection

W_AREA = 0.7
W_BOTTOM_Y = 0.3


@dataclass
class FrameResult:
    frame_idx: int
    detections: list[Detection]
    proximity_scores: dict[int, float]
    max_score_this_frame: float
    max_score_track_id: Optional[int]


@dataclass
class ProximityResult:
    global_max_frame: int
    global_max_track_id: int
    global_max_score: float
    per_track_max: dict = field(default_factory=dict)
    per_track_history: dict = field(default_factory=dict)


class ProximityAnalyzer:
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = frame_width * frame_height

        self._history: dict[int, list[tuple[int, float]]] = {}
        self._per_track_max: dict[int, tuple[int, float]] = {}
        self._global_max_score: float = 0.0
        self._global_max_frame: int = -1
        self._global_max_track_id: int = -1
        self._all_scores: list[float] = []

    def compute_score(self, detection: Detection) -> float:
        normalized_area = detection.area / self.frame_area
        normalized_bottom = detection.bottom_y / self.frame_height
        return W_AREA * normalized_area + W_BOTTOM_Y * normalized_bottom

    def process_frame(self, frame_idx: int,
                      detections: list[Detection]) -> dict[int, float]:
        scores = {}

        for det in detections:
            if det.track_id is None:
                continue

            score = self.compute_score(det)
            scores[det.track_id] = score
            self._all_scores.append(score)

            if det.track_id not in self._history:
                self._history[det.track_id] = []
            self._history[det.track_id].append((frame_idx, score))

            if det.track_id not in self._per_track_max or \
                    score > self._per_track_max[det.track_id][1]:
                self._per_track_max[det.track_id] = (frame_idx, score)

            if score > self._global_max_score:
                self._global_max_score = score
                self._global_max_frame = frame_idx
                self._global_max_track_id = det.track_id

        return scores

    def get_result(self) -> ProximityResult:
        return ProximityResult(
            global_max_frame=self._global_max_frame,
            global_max_track_id=self._global_max_track_id,
            global_max_score=self._global_max_score,
            per_track_max=dict(self._per_track_max),
            per_track_history={k: list(v) for k, v in self._history.items()},
        )

    def get_proximity_percentile(self, score: float) -> float:
        if not self._all_scores:
            return 0.0
        sorted_scores = np.array(sorted(self._all_scores))
        idx = np.searchsorted(sorted_scores, score)
        return idx / len(sorted_scores) * 100.0
