import cv2
import numpy as np

from detector import Detection
from proximity import ProximityResult, ProximityAnalyzer
from metrics import MetricsStore

COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


class Visualizer:
    def __init__(self, frame_width: int, frame_height: int,
                 proximity_result: ProximityResult,
                 proximity_analyzer: ProximityAnalyzer,
                 metrics_store: MetricsStore,
                 total_frames: int):
        self.fw = frame_width
        self.fh = frame_height
        self.proximity = proximity_result
        self.analyzer = proximity_analyzer
        self.metrics = metrics_store
        self.total_frames = total_frames

    def render_frame(self, frame: np.ndarray, frame_idx: int,
                     detections: list[Detection],
                     scores: dict[int, float]) -> np.ndarray:
        out = frame.copy()

        is_global_closest = (frame_idx == self.proximity.global_max_frame)

        sorted_dets = sorted(detections, key=lambda d: scores.get(
            d.track_id if d.track_id is not None else -1, 0))

        for det in sorted_dets:
            tid = det.track_id
            score = scores.get(tid, 0.0) if tid is not None else 0.0

            is_this_global = (is_global_closest and
                              tid == self.proximity.global_max_track_id)

            color = self._get_box_color(score, is_this_global)
            thickness = 5 if is_this_global else self._get_box_thickness(score)

            x1, y1, x2, y2 = det.bbox_xyxy
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            if is_this_global:
                self._draw_glow_effect(out, det.bbox_xyxy, COLOR_MAGENTA)
                self._draw_overlay(out, det.bbox_xyxy, COLOR_MAGENTA, 0.2)

            id_str = str(tid) if tid is not None else "?"
            label1 = f"ID:{id_str} conf:{det.confidence:.2f}"
            score_str = f"{score:.3f}" if tid is not None else "N/A"
            label2 = f"area:{det.area:.0f} prox:{score_str}"
            self._draw_label(out, (x1, y1), label1, color)
            self._draw_label(out, (x1, y1 + 22), label2, color)

        if is_global_closest and self.proximity.global_max_frame >= 0:
            self._draw_closest_banner(out, frame_idx)
            self._draw_frame_border(out, COLOR_MAGENTA, 8)

        self._draw_hud(out, frame_idx, detections, scores)

        return out

    def _get_box_color(self, score: float,
                       is_global_closest: bool) -> tuple:
        if is_global_closest:
            return COLOR_MAGENTA

        pct = self.analyzer.get_proximity_percentile(score)

        if pct < 25:
            return COLOR_GREEN
        elif pct < 50:
            t = (pct - 25) / 25
            return self._lerp_color(COLOR_GREEN, COLOR_YELLOW, t)
        elif pct < 75:
            t = (pct - 50) / 25
            return self._lerp_color(COLOR_YELLOW, COLOR_ORANGE, t)
        else:
            t = (pct - 75) / 25
            return self._lerp_color(COLOR_ORANGE, COLOR_RED, t)

    def _get_box_thickness(self, score: float) -> int:
        pct = self.analyzer.get_proximity_percentile(score)
        if pct > 75:
            return 3
        return 2

    def _lerp_color(self, c1: tuple, c2: tuple, t: float) -> tuple:
        t = max(0.0, min(1.0, t))
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    def _draw_label(self, frame: np.ndarray, pos: tuple,
                    text: str, color: tuple) -> None:
        x, y = pos
        font_scale = 0.5
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)

        label_y = max(y - 5, th + 5)

        cv2.rectangle(frame, (x, label_y - th - 4),
                      (x + tw + 4, label_y + baseline), COLOR_BLACK, -1)
        cv2.putText(frame, text, (x + 2, label_y - 2),
                    FONT, font_scale, color, thickness, cv2.LINE_AA)

    def _draw_overlay(self, frame: np.ndarray, bbox: np.ndarray,
                      color: tuple, alpha: float) -> None:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.fw, x2), min(self.fh, y2)

        overlay = frame[y1:y2, x1:x2].copy()
        colored = np.full_like(overlay, color, dtype=np.uint8)
        blended = cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0)
        frame[y1:y2, x1:x2] = blended

    def _draw_glow_effect(self, frame: np.ndarray, bbox: np.ndarray,
                          color: tuple, layers: int = 4) -> None:
        x1, y1, x2, y2 = bbox
        overlay = frame.copy()

        for i in range(layers, 0, -1):
            expand = i * 5
            gx1 = max(0, x1 - expand)
            gy1 = max(0, y1 - expand)
            gx2 = min(self.fw - 1, x2 + expand)
            gy2 = min(self.fh - 1, y2 + expand)
            cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), color, 2)

        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_frame_border(self, frame: np.ndarray,
                           color: tuple, thickness: int) -> None:
        cv2.rectangle(frame, (0, 0),
                      (self.fw - 1, self.fh - 1), color, thickness)

    def _draw_closest_banner(self, frame: np.ndarray,
                             frame_idx: int) -> None:
        banner_h = 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.fw, banner_h),
                      COLOR_MAGENTA, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        text = f">>> CLOSEST APPROACH - Frame {frame_idx}, " \
               f"Car #{self.proximity.global_max_track_id} | " \
               f"Score: {self.proximity.global_max_score:.4f} <<<"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
        tx = (self.fw - tw) // 2
        ty = (banner_h + th) // 2
        cv2.putText(frame, text, (tx, ty), FONT, 0.7,
                    COLOR_WHITE, 2, cv2.LINE_AA)

        panel_x = self.fw - 320
        panel_y = banner_h + 10
        self._draw_panel(frame, panel_x, panel_y, 310, 100, [
            f"Frame: {frame_idx} / {self.total_frames}",
            f"Track ID: {self.proximity.global_max_track_id}",
            f"Proximity Score: {self.proximity.global_max_score:.4f}",
        ])

    def _draw_hud(self, frame: np.ndarray, frame_idx: int,
                  detections: list[Detection],
                  scores: dict[int, float]) -> None:
        lines = [
            f"Frame: {frame_idx}/{self.total_frames}",
            f"Cars detected: {len(detections)}",
        ]

        if scores:
            best_tid = max(scores, key=scores.get)
            best_score = scores[best_tid]
            lines.append(f"Closest this frame: #{best_tid} "
                         f"(score: {best_score:.3f})")
        else:
            lines.append("Closest this frame: N/A")

        if self.proximity.global_max_frame >= 0:
            lines.append(f"Global closest: Frame {self.proximity.global_max_frame}, "
                         f"Car #{self.proximity.global_max_track_id} "
                         f"({self.proximity.global_max_score:.3f})")
        else:
            lines.append("Global closest: N/A")

        if detections:
            avg_conf = sum(d.confidence for d in detections) / len(detections)
            lines.append(f"Avg confidence: {avg_conf:.3f}")

        self._draw_panel(frame, 10, 10, 350, 25 * len(lines) + 15, lines)

    def _draw_panel(self, frame: np.ndarray, x: int, y: int,
                    w: int, h: int, lines: list[str]) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, line in enumerate(lines):
            ty = y + 20 + i * 25
            cv2.putText(frame, line, (x + 8, ty), FONT, 0.5,
                        COLOR_WHITE, 1, cv2.LINE_AA)
