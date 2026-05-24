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
COLOR_CYAN = (255, 220, 0)   # gold/cyan for per-car closest
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


class Visualizer:
    def __init__(self, frame_width: int, frame_height: int,
                 proximity_result: ProximityResult,
                 proximity_analyzer: ProximityAnalyzer,
                 metrics_store: MetricsStore,
                 total_frames: int,
                 fps: float):
        self.fw = frame_width
        self.fh = frame_height
        self.proximity = proximity_result
        self.analyzer = proximity_analyzer
        self.metrics = metrics_store
        self.total_frames = total_frames
        self.fps = fps

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def render_frame(self, frame: np.ndarray, frame_idx: int,
                     detections: list[Detection],
                     scores: dict[int, float]) -> np.ndarray:
        out = frame.copy()

        is_global_closest_frame = (frame_idx == self.proximity.global_max_frame)

        # Draw lower-priority detections first so the closest ones paint on top
        sorted_dets = sorted(detections, key=lambda d: scores.get(
            d.track_id if d.track_id is not None else -1, 0))

        per_car_closest_dets = []   # cars at their personal closest this frame

        for det in sorted_dets:
            tid = det.track_id
            score = scores.get(tid, 0.0) if tid is not None else 0.0

            is_global = (
                is_global_closest_frame
                and tid == self.proximity.global_max_track_id
            )

            is_per_car_closest = (
                tid is not None
                and not is_global
                and tid in self.proximity.per_track_max
                and self.proximity.per_track_max[tid][0] == frame_idx
            )

            if is_per_car_closest:
                per_car_closest_dets.append(det)

            color, thickness = self._box_style(score, is_global, is_per_car_closest)

            x1, y1, x2, y2 = det.bbox_xyxy
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            if is_global:
                self._draw_glow_effect(out, det.bbox_xyxy, COLOR_MAGENTA)
                self._draw_overlay(out, det.bbox_xyxy, COLOR_MAGENTA, 0.2)
            elif is_per_car_closest:
                self._draw_glow_effect(out, det.bbox_xyxy, COLOR_CYAN, layers=3)
                self._draw_overlay(out, det.bbox_xyxy, COLOR_CYAN, 0.15)

            cx, cy = int(det.center[0]), int(det.center[1])
            id_str = str(tid) if tid is not None else "?"
            score_str = f"{score:.3f}" if tid is not None else "N/A"
            ts = self._frame_to_ts(frame_idx)

            label1 = f"Car #{id_str}  conf:{det.confidence:.2f}"
            label2 = f"pos:({cx},{cy})  area:{det.area:.0f}"
            label3 = f"prox:{score_str}  t:{ts}"

            label_color = color
            if is_per_car_closest:
                label3 = f"*** CLOSEST (prox:{score_str}  t:{ts}) ***"
                label_color = COLOR_CYAN
            if is_global:
                label3 = f"### GLOBAL CLOSEST (prox:{score_str}  t:{ts}) ###"
                label_color = COLOR_MAGENTA

            self._draw_label(out, (x1, y1), label1, label_color)
            self._draw_label(out, (x1, y1 + 22), label2, label_color)
            self._draw_label(out, (x1, y1 + 44), label3, label_color)

        # Full-frame banner for the global closest moment
        if is_global_closest_frame and self.proximity.global_max_frame >= 0:
            self._draw_global_banner(out, frame_idx)
            self._draw_frame_border(out, COLOR_MAGENTA, 8)

        # Small per-car banners for non-global closest moments
        for i, det in enumerate(per_car_closest_dets):
            self._draw_per_car_badge(out, det, frame_idx, i)

        self._draw_hud(out, frame_idx, detections, scores)

        return out

    # ------------------------------------------------------------------ #
    #  Helpers: time                                                       #
    # ------------------------------------------------------------------ #

    def _frame_to_ts(self, frame_idx: int) -> str:
        """Convert frame index to MM:SS.ff timestamp string."""
        if self.fps <= 0:
            return "00:00.00"
        total_secs = frame_idx / self.fps
        mins = int(total_secs // 60)
        secs = total_secs % 60
        return f"{mins:02d}:{secs:05.2f}"

    # ------------------------------------------------------------------ #
    #  Helpers: colors / styles                                            #
    # ------------------------------------------------------------------ #

    def _box_style(self, score: float, is_global: bool,
                   is_per_car_closest: bool) -> tuple:
        if is_global:
            return COLOR_MAGENTA, 5
        if is_per_car_closest:
            return COLOR_CYAN, 4

        pct = self.analyzer.get_proximity_percentile(score)
        if pct < 25:
            color = COLOR_GREEN
        elif pct < 50:
            color = self._lerp_color(COLOR_GREEN, COLOR_YELLOW, (pct - 25) / 25)
        elif pct < 75:
            color = self._lerp_color(COLOR_YELLOW, COLOR_ORANGE, (pct - 50) / 25)
        else:
            color = self._lerp_color(COLOR_ORANGE, COLOR_RED, (pct - 75) / 25)

        thickness = 3 if pct > 75 else 2
        return color, thickness

    def _lerp_color(self, c1: tuple, c2: tuple, t: float) -> tuple:
        t = max(0.0, min(1.0, t))
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    # ------------------------------------------------------------------ #
    #  Helpers: drawing                                                    #
    # ------------------------------------------------------------------ #

    def _draw_label(self, frame: np.ndarray, pos: tuple,
                    text: str, color: tuple) -> None:
        x, y = pos
        font_scale, thickness = 0.48, 1
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
        frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0)

    def _draw_glow_effect(self, frame: np.ndarray, bbox: np.ndarray,
                          color: tuple, layers: int = 4) -> None:
        x1, y1, x2, y2 = bbox
        overlay = frame.copy()
        for i in range(layers, 0, -1):
            expand = i * 5
            cv2.rectangle(overlay,
                          (max(0, x1 - expand), max(0, y1 - expand)),
                          (min(self.fw - 1, x2 + expand), min(self.fh - 1, y2 + expand)),
                          color, 2)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    def _draw_frame_border(self, frame: np.ndarray,
                           color: tuple, thickness: int) -> None:
        cv2.rectangle(frame, (0, 0),
                      (self.fw - 1, self.fh - 1), color, thickness)

    def _draw_global_banner(self, frame: np.ndarray, frame_idx: int) -> None:
        ts = self._frame_to_ts(frame_idx)
        banner_h = 52
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.fw, banner_h), COLOR_MAGENTA, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        text = (f">>> GLOBAL CLOSEST APPROACH  "
                f"Frame {frame_idx}  |  {ts}  |  "
                f"Car #{self.proximity.global_max_track_id}  |  "
                f"Score {self.proximity.global_max_score:.4f} <<<")
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.65, 2)
        tx = max(6, (self.fw - tw) // 2)
        ty = (banner_h + th) // 2
        cv2.putText(frame, text, (tx, ty), FONT, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)

        # Numeric readout panel top-right
        tid = self.proximity.global_max_track_id
        # Find the detection's center from per_track_max if available
        best_frame, best_score = self.proximity.per_track_max.get(tid, (frame_idx, 0))
        self._draw_panel(frame, self.fw - 295, banner_h + 8, 285, 105, [
            f"Timestamp : {ts}",
            f"Frame     : {frame_idx} / {self.total_frames}",
            f"Track ID  : {self.proximity.global_max_track_id}",
            f"Prox Score: {self.proximity.global_max_score:.4f}",
        ])

    def _draw_per_car_badge(self, frame: np.ndarray, det: Detection,
                            frame_idx: int, slot: int) -> None:
        """Small banner for a per-car closest moment (non-global)."""
        ts = self._frame_to_ts(frame_idx)
        cx, cy = int(det.center[0]), int(det.center[1])
        tid = det.track_id
        score = self.proximity.per_track_max.get(tid, (-1, 0.0))[1]

        badge_h = 28
        badge_y = self.fh - (slot + 1) * (badge_h + 4)
        badge_y = max(badge_y, 0)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, badge_y), (self.fw, badge_y + badge_h),
                      COLOR_CYAN, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        text = (f"  Car #{tid} closest: Frame {frame_idx}  |  "
                f"{ts}  |  pos:({cx},{cy})  |  score:{score:.4f}")
        cv2.putText(frame, text, (6, badge_y + 20), FONT, 0.52,
                    COLOR_BLACK, 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, frame_idx: int,
                  detections: list[Detection],
                  scores: dict[int, float]) -> None:
        ts = self._frame_to_ts(frame_idx)
        lines = [
            f"Frame: {frame_idx}/{self.total_frames}  [{ts}]",
            f"Cars detected: {len(detections)}",
        ]

        if scores:
            best_tid = max(scores, key=scores.get)
            best_score = scores[best_tid]
            lines.append(f"Closest this frame: Car #{best_tid}  "
                         f"(score:{best_score:.3f})")
        else:
            lines.append("Closest this frame: N/A")

        if self.proximity.global_max_frame >= 0:
            gts = self._frame_to_ts(self.proximity.global_max_frame)
            lines.append(f"Global closest: Frame {self.proximity.global_max_frame} "
                         f"[{gts}]")
            lines.append(f"  Car #{self.proximity.global_max_track_id}  "
                         f"score:{self.proximity.global_max_score:.3f}")
        else:
            lines.append("Global closest: N/A")

        if detections:
            avg_conf = sum(d.confidence for d in detections) / len(detections)
            lines.append(f"Avg confidence: {avg_conf:.3f}")

        self._draw_panel(frame, 10, 10, 370, 25 * len(lines) + 15, lines)

    def _draw_panel(self, frame: np.ndarray, x: int, y: int,
                    w: int, h: int, lines: list[str]) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x + 8, y + 20 + i * 25),
                        FONT, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
