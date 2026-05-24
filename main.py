import argparse
import sys
import time

import cv2

from detector import CarDetector
from proximity import ProximityAnalyzer, FrameResult
from visualizer import Visualizer
from metrics import MetricsStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Car Detection & Tracking with Closest-to-Camera Analysis"
    )
    parser.add_argument("input_video", type=str,
                        help="Path to input video file")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output video path (default: <input>_annotated.mp4)")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="YOLOv8 model name (default: yolov8s.pt)")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Inference device (default: cpu)")
    parser.add_argument("--export-json", type=str, default=None,
                        help="Path to export metrics JSON file")
    parser.add_argument("--no-display", action="store_true",
                        help="Suppress live preview window")
    return parser.parse_args()


def get_video_info(cap: cv2.VideoCapture) -> dict:
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def run_pass1(cap: cv2.VideoCapture, detector: CarDetector,
              proximity: ProximityAnalyzer,
              metrics: MetricsStore,
              total_frames: int) -> list[FrameResult]:
    frame_results = []
    frame_idx = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_and_track(frame, persist=True)
        scores = proximity.process_frame(frame_idx, detections)
        metrics.record_frame(frame_idx, detections)

        max_score = max(scores.values()) if scores else 0.0
        max_tid = max(scores, key=scores.get) if scores else None

        frame_results.append(FrameResult(
            frame_idx=frame_idx,
            detections=detections,
            proximity_scores=scores,
            max_score_this_frame=max_score,
            max_score_track_id=max_tid,
        ))

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            fps = frame_idx / elapsed if elapsed > 0 else 0
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"\r  Pass 1: frame {frame_idx}/{total_frames} "
                  f"({pct:.1f}%) - {fps:.1f} fps", end="", flush=True)

        frame_idx += 1

    print()
    return frame_results


def run_pass2(cap: cv2.VideoCapture, frame_results: list[FrameResult],
              visualizer: Visualizer, output_path: str,
              fps: float, no_display: bool,
              total_frames: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps,
                             (visualizer.fw, visualizer.fh))

    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_path_avi = output_path.rsplit(".", 1)[0] + ".avi"
        writer = cv2.VideoWriter(output_path_avi, fourcc, fps,
                                 (visualizer.fw, visualizer.fh))
        if not writer.isOpened():
            print(f"Error: Cannot create output video writer")
            return

    t0 = time.time()

    for i, result in enumerate(frame_results):
        ret, frame = cap.read()
        if not ret:
            break

        annotated = visualizer.render_frame(
            frame, result.frame_idx, result.detections,
            result.proximity_scores,
        )

        writer.write(annotated)

        if not no_display:
            display = annotated
            if annotated.shape[1] > 1280:
                scale = 1280 / annotated.shape[1]
                display = cv2.resize(annotated, None, fx=scale, fy=scale)
            cv2.imshow("Car Tracking", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n  Preview stopped by user")
                break

        if i % 100 == 0:
            elapsed = time.time() - t0
            render_fps = i / elapsed if elapsed > 0 else 0
            pct = i / total_frames * 100 if total_frames > 0 else 0
            print(f"\r  Pass 2: frame {i}/{total_frames} "
                  f"({pct:.1f}%) - {render_fps:.1f} fps", end="", flush=True)

    print()
    writer.release()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{args.input_video}'")
        sys.exit(1)

    info = get_video_info(cap)
    print(f"Video: {info['width']}x{info['height']}, "
          f"{info['fps']:.1f} FPS, {info['total_frames']} frames "
          f"({info['total_frames']/info['fps']:.1f}s)")

    detector = CarDetector(
        model_name=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
    )
    proximity = ProximityAnalyzer(info["width"], info["height"])
    metrics = MetricsStore(info["total_frames"],
                           info["width"], info["height"])

    print("\n=== PASS 1: Detection & Analysis ===")
    t0 = time.time()
    frame_results = run_pass1(cap, detector, proximity, metrics,
                              info["total_frames"])
    t1 = time.time()
    print(f"Pass 1 complete in {t1 - t0:.1f}s")

    metrics.finalize()
    prox_result = proximity.get_result()

    if prox_result.global_max_frame >= 0:
        print(f"\nClosest approach: Frame {prox_result.global_max_frame} "
              f"(Car #{prox_result.global_max_track_id}, "
              f"score {prox_result.global_max_score:.4f})")
    else:
        print("\nNo vehicles detected in the video.")

    output_path = args.output or \
        args.input_video.rsplit(".", 1)[0] + "_annotated.mp4"

    visualizer = Visualizer(
        frame_width=info["width"],
        frame_height=info["height"],
        proximity_result=prox_result,
        proximity_analyzer=proximity,
        metrics_store=metrics,
        total_frames=info["total_frames"],
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    detector.reset_tracker()

    print("\n=== PASS 2: Rendering ===")
    t2 = time.time()
    run_pass2(cap, frame_results, visualizer, output_path,
              info["fps"], args.no_display, info["total_frames"])
    t3 = time.time()
    print(f"Pass 2 complete in {t3 - t2:.1f}s")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(metrics.generate_report())
    print("=" * 60)
    print(f"\nOutput video: {output_path}")
    print(f"Total processing time: {t3 - t0:.1f}s")

    if args.export_json:
        metrics.export_json(args.export_json)
        print(f"Metrics exported: {args.export_json}")


if __name__ == "__main__":
    main()
