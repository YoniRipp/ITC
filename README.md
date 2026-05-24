# Car Detection & Tracking with Closest-to-Camera Analysis

A Python application that detects and tracks cars in video, visually highlights the moment a car is closest to the camera, and provides accuracy metrics.

## Quick Start

```bash
pip install -r requirements.txt
python main.py path/to/video.mp4
```

This produces an annotated video (`video_annotated.mp4`) and prints a detection report to the console.

### CLI Options

```
python main.py input.mp4 [options]

positional arguments:
  input_video           Path to input video file

options:
  -o, --output          Output video path (default: <input>_annotated.mp4)
  --model               YOLOv8 model name (default: yolov8s.pt)
  --confidence          Detection confidence threshold (default: 0.3)
  --device              Inference device: cpu, cuda, mps (default: cpu)
  --export-json PATH    Export metrics to JSON file
  --no-display          Suppress live preview window
```

### Example

```bash
# Basic usage
python main.py dashcam.mp4

# Higher accuracy model, GPU, JSON export
python main.py dashcam.mp4 --model yolov8m.pt --device cuda --export-json report.json

# Stricter confidence threshold
python main.py dashcam.mp4 --confidence 0.5 -o output.mp4
```

---

## How It Works

### Two-Pass Pipeline

The application uses a two-pass architecture:

**Pass 1 (Analysis):** Runs YOLOv8 detection + BoT-SORT tracking on every frame. Computes a proximity score for each detected car. Identifies the single frame where a car is closest to the camera (global maximum). Accumulates accuracy metrics.

**Pass 2 (Rendering):** Re-reads the video and draws annotations using the pre-computed results. Because Pass 1 already identified the global closest frame, Pass 2 can definitively mark it with special visuals.

Why two passes? You cannot mark the "closest frame in the entire video" during a single pass because you don't know which frame it is until you've processed every frame. Buffering all frames in memory is impractical (~14GB for a 3-minute 1080p video). Two passes solve this cleanly: analyze first, then render.

### Proximity Score

Each detected car receives a **proximity score** per frame:

```
score = 0.7 * (bbox_area / frame_area) + 0.3 * (bottom_y / frame_height)
```

- **Bounding box area** (weight 0.7): Under perspective projection (pinhole camera model), an object at half the distance occupies ~4x the pixel area. Larger box = closer to camera. This is the primary distance signal.
- **Bottom-edge y-coordinate** (weight 0.3): For objects on a ground plane, closer objects appear lower in the frame (higher y-value). This disambiguates cases where a large vehicle far away has similar area to a small vehicle nearby.

Both values are normalized to [0, 1] by frame dimensions, making the score resolution-independent.

### Visualization

| State | Visual |
|-------|--------|
| Normal detection | Green bounding box (2px), ID + confidence + area label |
| Increasing proximity | Color gradient: green -> yellow -> orange -> red (based on percentile) |
| Global closest frame | Magenta box (5px) + glow effect + full-width banner + frame border + numeric readout |
| HUD (always visible) | Top-left panel: frame counter, car count, current closest, global closest, avg confidence |

---

## 6. Architecture Rationale & Logic

### Why YOLOv8?

YOLOv8 is a **single-stage object detector** -- it performs detection in one forward pass through the network, unlike two-stage architectures (e.g., Faster R-CNN) that first propose regions then classify them. This makes it inherently fast and suitable for video processing.

Key reasons:

- **COCO-pretrained weights include vehicles out of the box.** Class 2 = car, class 5 = bus, class 7 = truck. No custom training required.
- **The `ultralytics` library provides a clean Python API.** Model loading, inference, tracking, and result parsing through a unified interface. Weights are downloaded automatically on first run.
- **Multiple model sizes for speed/accuracy tradeoff.** We default to `yolov8s` (small) which achieves 44.9 mAP@0.5:0.95 on COCO -- a good balance between accuracy and CPU inference speed.

### Why BoT-SORT for Tracking?

BoT-SORT is a multi-object tracker built into ultralytics. It assigns persistent integer IDs to detected objects across frames by combining:

- **Kalman filter motion prediction**: Predicts where each tracked object should appear in the next frame.
- **Appearance features**: A lightweight re-identification (ReID) vector for cosine similarity matching.
- **IoU matching**: Spatial overlap between predicted and observed bounding boxes.

Invoked simply via `model.track(frame, persist=True)`. No external tracker dependency needed.

### Why Bounding Box Area as Distance Proxy?

Under the **pinhole camera model** (perspective projection), an object at distance *d* produces an image whose linear dimensions scale as *1/d*. Since bounding box area = width x height, **area scales as 1/d^2**. A car twice as close occupies ~4x the pixel area.

This is combined with bottom-edge y-position (closer ground-plane objects appear lower in the frame) to form a weighted composite score. The approach is simple, requires no camera calibration, and works well for typical dashcam/security camera footage.

### How Accuracy Is Checked

Since we don't have ground truth labels for arbitrary user videos, accuracy is assessed through **self-consistency metrics**:

- **Confidence score distribution**: Mean, median, standard deviation, and per-decile histogram of all detection confidence scores. High mean confidence indicates the model is finding clear, unambiguous objects.
- **Track continuity**: For each tracked car, the ratio of frames where it was actually detected vs. the expected number of frames (last_seen - first_seen + 1). A continuity of 1.0 means the car was never lost; lower values indicate tracking gaps.
- **Detection consistency**: Standard deviation of per-frame detection count. Low std means stable, consistent detection.
- **False positive heuristics**: Tracks lasting < 3 frames (flicker noise), tracks with average area < 500px (too small to be a real car), tracks with average confidence < 0.35 (uncertain detections), and tracks appearing only at frame edges (partial objects).
- **Threshold sensitivity analysis**: How many detections survive at confidence thresholds 0.25, 0.50, 0.75, 0.90 -- reveals the precision-recall tradeoff.

---

## 7. How Accurate Is the Program?

### Detection Accuracy

| Model | Overall mAP@0.5:0.95 | Vehicle-Class mAP (est.) | Speed |
|-------|----------------------|--------------------------|-------|
| YOLOv8n | ~37.3 | ~55-65 | Fastest |
| YOLOv8s (default) | ~44.9 | ~65-75 | Fast |
| YOLOv8m | ~50.2 | ~70-80 | Moderate |

Vehicle-class mAP is higher than overall because cars and trucks are among the best-represented categories in COCO training data.

### Proximity Ranking Accuracy

The proximity score provides a **correct relative ranking** -- it reliably identifies which car is closest in a given frame, and which frame contains the overall closest approach. However:

- It is **not calibrated to real-world distance** (no meters/feet output)
- It is a **proxy** based on geometry, not a direct measurement
- It assumes a **static camera** with a roughly horizontal viewing angle

For the question "which frame has the closest car?", the answer is reliable under the conditions in Section 8.

### Confidence Threshold Effects

| Threshold | Precision | Recall | Behavior |
|-----------|-----------|--------|----------|
| 0.25 | Lower | Higher | Catches more cars, more false positives |
| 0.30 (default) | Good | Good | Balanced tradeoff |
| 0.50 | Higher | Lower | Very few false positives, may miss uncertain detections |
| 0.75 | Highest | Low | Only high-confidence detections survive |

### Main Accuracy Limitation

The "closest frame" determination depends on the detector finding the car in **every frame**. If the car is missed in the actual closest frame (due to motion blur, occlusion, or a momentary confidence dip), the program reports the next-closest frame where detection succeeded.

---

## 8. Strengths and Weaknesses

### Where It Works Well

| Condition | Why |
|-----------|-----|
| **Clear daytime footage** from a static camera | High contrast, consistent lighting, stable background. Confidence typically 0.8+. |
| **Standard vehicle types** (sedan, SUV, truck, bus) | Well-represented in COCO training data (tens of thousands of examples). |
| **Single or few cars** in the scene | Tracker maintains IDs reliably with few association ambiguities. |
| **Cars approaching head-on or receding** | Bounding box area changes monotonically with distance -- clean, unambiguous signal. |
| **Moderate vehicle speeds** (30-60 km/h) | At standard frame rates (24-30 fps), inter-frame displacement is small enough for Kalman filter prediction. |

### Where It Fails or Degrades

| Condition | What Goes Wrong | Severity |
|-----------|-----------------|----------|
| **Night / rain / fog / snow** | Low contrast reduces detection confidence. Cars missed intermittently, creating fragmented tracks. | High |
| **Heavy occlusion** (overlapping cars) | Tracker cannot distinguish overlapping detections. Causes ID switches -- a car's track_id changes mid-video. | High |
| **Unusual camera angles** (top-down, extreme side) | Box area no longer correlates with distance. Top-down view produces equal box sizes regardless of distance. | High |
| **Camera motion** (panning, zooming, shaking) | Zoom changes box area without distance change. Panning confuses motion model. Fundamental static-camera assumption violated. | High |
| **Non-standard vehicles** | Construction equipment, motorcycles, partially visible vehicles may not match COCO categories. | Medium |
| **Very small / distant cars** (< 30x30 px) | Detection confidence drops sharply. Feature map resolution insufficient at small scales. | Medium |
| **Cars moving laterally** (parallel to camera) | Box area stays roughly constant. Proximity score doesn't reflect actual distance changes at slight angles. | Low-Medium |
| **Size ambiguity** | A bus far away may have similar box area to a sedan nearby. The score cannot distinguish inherent object size from distance. | Medium |

---

## 9. If You Had One More Hour

### 1. Monocular Depth Estimation (MiDaS / Depth Anything)

Integrate a pretrained monocular depth model to generate per-pixel depth maps. Instead of using bounding box area as a proxy, sample the depth map within each detection to get an estimated distance.

**Why this is #1**: It directly addresses the single biggest limitation -- bounding box area is a coarse geometric proxy. A depth map provides pixel-level distance estimates that work regardless of camera angle, object size, or lateral motion.

**Effort**: ~45-60 minutes. MiDaS provides pretrained PyTorch models with simple inference APIs.

### 2. DeepSORT for Better Tracking

Replace BoT-SORT with DeepSORT, which uses a dedicated appearance-feature extractor (ReID network) to re-identify objects after occlusion.

**Why**: In multi-car scenarios, temporary occlusion causes the current tracker to assign new IDs. DeepSORT's appearance features allow re-association with the original ID, producing cleaner trajectories.

**Effort**: ~30-45 minutes. The `deep-sort-realtime` package provides a drop-in tracker.

### 3. Simple Web UI (Gradio)

Add a Gradio interface for drag-and-drop video upload, progress display, and output download -- no CLI knowledge required.

**Why**: Makes the tool accessible to non-technical users.

**Effort**: ~20-30 minutes. Gradio wraps existing Python functions with minimal code.

### 4. Speed / Velocity Estimation

Use tracked bounding box positions across consecutive frames plus known frame rate to compute relative velocity in pixels/second.

**Why**: A car that is far away but approaching quickly may be more relevant than a nearby stationary car. Velocity enables time-to-closest-approach predictions.

**Effort**: ~20-30 minutes. Centroid positions are already tracked.

---

## Project Structure

```
ITC/
├── main.py              # CLI entry point, two-pass orchestration
├── detector.py          # YOLOv8 model loading, detection + tracking, Detection dataclass
├── proximity.py         # Proximity scoring, per-track history, global max tracking
├── visualizer.py        # All drawing: boxes, color gradients, glow, HUD, banner
├── metrics.py           # Confidence stats, track continuity, false positive heuristics
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Dependencies

- **ultralytics** >= 8.0.0 -- YOLOv8 detection and tracking
- **opencv-python** >= 4.8.0 -- Video I/O and drawing
- **numpy** >= 1.24.0 -- Array operations
