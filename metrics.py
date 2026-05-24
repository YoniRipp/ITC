from dataclasses import dataclass, field
from typing import Optional
import json
import statistics

import numpy as np

from detector import Detection


@dataclass
class TrackMetrics:
    track_id: int
    first_seen: int
    last_seen: int
    total_frames: int
    expected_frames: int
    continuity: float
    avg_confidence: float
    confidence_std: float
    avg_area: float
    is_likely_false_positive: bool
    fp_reason: Optional[str]


class MetricsStore:
    def __init__(self, total_frames: int, frame_width: int, frame_height: int):
        self.total_frames = total_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = frame_width * frame_height

        self._all_confidences: list[float] = []
        self._detection_counts: list[int] = []
        self._track_data: dict[int, list[tuple[int, float, float, tuple]]] = {}
        self._finalized = False
        self._track_metrics: list[TrackMetrics] = []
        self._false_positives: list[TrackMetrics] = []

    def record_frame(self, frame_idx: int, detections: list[Detection]) -> None:
        self._detection_counts.append(len(detections))

        for det in detections:
            self._all_confidences.append(det.confidence)

            if det.track_id is not None:
                if det.track_id not in self._track_data:
                    self._track_data[det.track_id] = []
                self._track_data[det.track_id].append(
                    (frame_idx, det.confidence, det.area, det.center)
                )

    def finalize(self) -> None:
        self._track_metrics = []
        self._false_positives = []

        for track_id, entries in self._track_data.items():
            frames = [e[0] for e in entries]
            confs = [e[1] for e in entries]
            areas = [e[2] for e in entries]
            centers = [e[3] for e in entries]

            first_seen = min(frames)
            last_seen = max(frames)
            expected = last_seen - first_seen + 1
            continuity = len(frames) / expected if expected > 0 else 0.0

            avg_conf = statistics.mean(confs)
            conf_std = statistics.stdev(confs) if len(confs) > 1 else 0.0
            avg_area = statistics.mean(areas)

            is_fp = False
            fp_reason = None

            if len(frames) < 3:
                is_fp = True
                fp_reason = f"Short track ({len(frames)} frames)"
            elif avg_area < 500:
                is_fp = True
                fp_reason = f"Very small average area ({avg_area:.0f}px)"
            elif avg_conf < 0.35:
                is_fp = True
                fp_reason = f"Low average confidence ({avg_conf:.2f})"
            else:
                edge_margin_x = self.frame_width * 0.05
                edge_margin_y = self.frame_height * 0.05
                all_at_edge = all(
                    c[0] < edge_margin_x or c[0] > self.frame_width - edge_margin_x or
                    c[1] < edge_margin_y or c[1] > self.frame_height - edge_margin_y
                    for c in centers
                )
                if all_at_edge:
                    is_fp = True
                    fp_reason = "All detections at frame edge"

            tm = TrackMetrics(
                track_id=track_id,
                first_seen=first_seen,
                last_seen=last_seen,
                total_frames=len(frames),
                expected_frames=expected,
                continuity=continuity,
                avg_confidence=avg_conf,
                confidence_std=conf_std,
                avg_area=avg_area,
                is_likely_false_positive=is_fp,
                fp_reason=fp_reason,
            )
            self._track_metrics.append(tm)
            if is_fp:
                self._false_positives.append(tm)

        self._finalized = True

    def get_track_metrics(self) -> list[TrackMetrics]:
        return self._track_metrics

    def identify_false_positives(self) -> list[TrackMetrics]:
        return self._false_positives

    def get_detection_consistency(self) -> dict:
        if not self._detection_counts:
            return {"mean": 0, "std": 0, "zero_frames": 0,
                    "max_consecutive_zero": 0, "total_frames": 0}

        counts = self._detection_counts
        zero_frames = counts.count(0)

        max_consec_zero = 0
        current_consec = 0
        for c in counts:
            if c == 0:
                current_consec += 1
                max_consec_zero = max(max_consec_zero, current_consec)
            else:
                current_consec = 0

        return {
            "mean": statistics.mean(counts),
            "std": statistics.stdev(counts) if len(counts) > 1 else 0.0,
            "zero_frames": zero_frames,
            "max_consecutive_zero": max_consec_zero,
            "total_frames": len(counts),
        }

    def get_confidence_distribution(self) -> dict:
        if not self._all_confidences:
            return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0,
                    "histogram_bins": [], "histogram_counts": []}

        confs = self._all_confidences
        hist_counts, hist_edges = np.histogram(confs, bins=10, range=(0, 1))

        return {
            "mean": statistics.mean(confs),
            "median": statistics.median(confs),
            "std": statistics.stdev(confs) if len(confs) > 1 else 0.0,
            "min": min(confs),
            "max": max(confs),
            "histogram_bins": [f"{hist_edges[i]:.1f}-{hist_edges[i+1]:.1f}"
                               for i in range(len(hist_counts))],
            "histogram_counts": hist_counts.tolist(),
        }

    def generate_report(self) -> str:
        if not self._finalized:
            self.finalize()

        lines = []
        lines.append("DETECTION & TRACKING REPORT")
        lines.append("=" * 50)

        consistency = self.get_detection_consistency()
        lines.append(f"\nFrames processed: {consistency['total_frames']}")
        lines.append(f"Avg detections/frame: {consistency['mean']:.2f} "
                      f"(std: {consistency['std']:.2f})")
        lines.append(f"Frames with no detections: {consistency['zero_frames']}")
        lines.append(f"Max consecutive empty frames: "
                      f"{consistency['max_consecutive_zero']}")

        conf = self.get_confidence_distribution()
        lines.append(f"\nConfidence scores:")
        lines.append(f"  Mean: {conf['mean']:.3f}  Median: {conf['median']:.3f}  "
                      f"Std: {conf['std']:.3f}")
        lines.append(f"  Range: [{conf['min']:.3f}, {conf['max']:.3f}]")
        if conf['histogram_bins']:
            lines.append("  Distribution:")
            for bin_label, count in zip(conf['histogram_bins'],
                                        conf['histogram_counts']):
                bar = "#" * min(count, 50)
                lines.append(f"    {bin_label}: {count:4d} {bar}")

        valid_tracks = [t for t in self._track_metrics
                        if not t.is_likely_false_positive]
        lines.append(f"\nTracked vehicles: {len(valid_tracks)} "
                      f"(+ {len(self._false_positives)} likely false positives)")

        for tm in sorted(valid_tracks, key=lambda t: t.track_id):
            lines.append(f"\n  Car #{tm.track_id}:")
            lines.append(f"    Frames: {tm.first_seen} - {tm.last_seen} "
                          f"({tm.total_frames}/{tm.expected_frames} = "
                          f"{tm.continuity:.1%} continuity)")
            lines.append(f"    Avg confidence: {tm.avg_confidence:.3f} "
                          f"(std: {tm.confidence_std:.3f})")
            lines.append(f"    Avg bbox area: {tm.avg_area:.0f} px")

        if self._false_positives:
            lines.append("\nLikely false positives:")
            for fp in self._false_positives:
                lines.append(f"  Track #{fp.track_id}: {fp.fp_reason}")

        threshold_analysis = self._analyze_thresholds()
        lines.append("\nThreshold sensitivity analysis:")
        for thresh, count in sorted(threshold_analysis.items()):
            lines.append(f"  conf >= {thresh:.2f}: {count} detections")

        return "\n".join(lines)

    def _analyze_thresholds(self) -> dict[float, int]:
        thresholds = [0.25, 0.50, 0.75, 0.90]
        result = {}
        for t in thresholds:
            result[t] = sum(1 for c in self._all_confidences if c >= t)
        return result

    def export_json(self, filepath: str) -> None:
        if not self._finalized:
            self.finalize()

        data = {
            "summary": {
                "total_frames": len(self._detection_counts),
                "total_detections": len(self._all_confidences),
                "unique_tracks": len(self._track_metrics),
                "valid_tracks": len([t for t in self._track_metrics
                                     if not t.is_likely_false_positive]),
                "false_positives": len(self._false_positives),
            },
            "detection_consistency": self.get_detection_consistency(),
            "confidence_distribution": self.get_confidence_distribution(),
            "threshold_analysis": {
                str(k): v for k, v in self._analyze_thresholds().items()
            },
            "per_track": [
                {
                    "track_id": tm.track_id,
                    "first_seen": tm.first_seen,
                    "last_seen": tm.last_seen,
                    "total_frames": tm.total_frames,
                    "expected_frames": tm.expected_frames,
                    "continuity": round(tm.continuity, 4),
                    "avg_confidence": round(tm.avg_confidence, 4),
                    "confidence_std": round(tm.confidence_std, 4),
                    "avg_area": round(tm.avg_area, 1),
                    "is_likely_false_positive": tm.is_likely_false_positive,
                    "fp_reason": tm.fp_reason,
                }
                for tm in self._track_metrics
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
