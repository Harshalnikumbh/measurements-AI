# 📐 AI Body Measurement & Virtual Try-On System

> A production-grade Flask application that generates **accurate 3D body measurements** from front + side photos using **HMR2 (4D-Humans)**, MediaPipe pose validation, and a multi-layer measurement correction engine — paired with an AI **Virtual Try-On** feature powered by the LightX API.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Component Reference](#component-reference)
- [Measurement Pipeline](#measurement-pipeline)
- [Correction Engine](#correction-engine)
- [API Reference](#api-reference)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Virtual Try-On](#virtual-try-on)
- [Concurrency & Safety](#concurrency--safety)
- [Project Structure](#project-structure)

---

## Overview

This system takes two photos of a person (front-facing and side-facing), runs them through a 3D human mesh recovery pipeline (HMR2), and extracts precise clothing measurements — chest, waist, hip, shoulder, arm, thigh, knee, and more. A multi-layer correction engine then adjusts raw mesh readings based on BMI, age, body type, fat distribution, and user preferences.

---

## Key Features

| Feature | Description |
|---|---|
| 🧍 **3D Mesh Recovery** | HMR2 (4D-Humans) + ViTDet detector generates full body OBJ mesh from photos |
| 🤸 **Pose Validation** | MediaPipe verifies the person is standing straight before processing |
| 📏 **Full Measurement Suite** | Neck, chest, waist, hip, shoulder, armhole, thigh, knee, arm, body length |
| 🧠 **Multi-Layer Corrections** | BMI, age group, fat distribution, muscle level, activity level, fit preference |
| 👗 **Size Recommendation** | Rule-based clothing size (XS → XXXL) from three-measurement average |
| 🪞 **Virtual Try-On** | Upload person + clothing photo → AI-generated try-on via LightX API |
| 🔒 **Input Validation** | MIME check, cv2 decode, dimension/aspect ratio enforcement, schema validation |
| ⚡ **Parallel Processing** | Front and side mesh generation run concurrently via `ThreadPoolExecutor` |
| 🧵 **Thread-Safe Inference** | Single serialized worker queue prevents GPU/CPU race conditions |
| ⏱️ **Stage Timing** | `StageTimer` logs per-stage latency with request ID correlation |
| 🌐 **REST API** | Clean JSON API with structured error codes for all failure modes |

---

## Architecture

```
Browser / Client
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│                        Flask Application                          │
│                                                                   │
│   POST /process          POST /virtual-tryon-process             │
│        │                          │                              │
│        ▼                          ▼                              │
│ ┌─────────────────┐    ┌─────────────────────────┐              │
│ │ Input Validation│    │  VirtualTryOnService     │              │
│ │ • MIME check    │    │  • LightX API upload     │              │
│ │ • cv2 decode    │    │  • Order polling         │              │
│ │ • Dimensions    │    │  • Result download       │              │
│ │ • Schema fields │    └─────────────────────────┘              │
│ └────────┬────────┘                                              │
│          │                                                        │
│          ▼                                                        │
│ ┌─────────────────┐                                              │
│ │  PoseValidator  │  (MediaPipe Pose)                            │
│ │  Front + Side   │                                              │
│ └────────┬────────┘                                              │
│          │                                                        │
│          ▼                                                        │
│ ┌──────────────────────────────────────┐                         │
│ │     Parallel HMR2 Mesh Generation    │                         │
│ │                                      │                         │
│ │  ThreadPoolExecutor (max_workers=2)  │                         │
│ │  ┌────────────────┐ ┌──────────────┐ │                         │
│ │  │  front → .obj  │ │ side → .obj  │ │                         │
│ │  │  ViTDet detect │ │ ViTDet detect│ │                         │
│ │  │  HMR2 infer    │ │ HMR2 infer   │ │                         │
│ │  └────────────────┘ └──────────────┘ │                         │
│ │         ▲                  ▲         │                         │
│ │         └──── InferenceQueue ────────┘                         │
│ │               (serialized worker)    │                         │
│ └──────────────────┬───────────────────┘                         │
│                    │                                              │
│                    ▼                                              │
│ ┌──────────────────────────────────────────────────────┐         │
│ │        CompleteBodyMeasurementsCalculator            │         │
│ │                                                      │         │
│ │  1. Load + scale OBJ meshes (trimesh)               │         │
│ │  2. Slice mesh at percentage heights                │         │
│ │  3. Ramanujan ellipse circumference formula         │         │
│ │  4. Legacy weight-based adjustments (per-body-part) │         │
│ │  5. MeasurementCorrectionEngine (global scale)      │         │
│ │  6. Derived ratios (armhole, thigh, knee, arm)      │         │
│ │  7. Size recommendation                             │         │
│ └──────────────────────────────────────────────────────┘         │
│                    │                                              │
│                    ▼                                              │
│            JSON Response                                          │
└───────────────────────────────────────────────────────────────────┘
```

### Inference Thread Architecture

```
HTTP Request Thread(s)          InferenceWorker Thread (daemon)
        │                                │
        │  put_nowait(task)              │
        ├───────────────────────────────▶│
        │                                │  fn(*args) executes
        │  event.wait(timeout=300s)      │  (HMR2 model inference)
        │◀────── event.set() ────────────│
        │                                │
        │  result or error               │
        ▼                                │
   JSON Response                         │ (loops, picks next task)
```

### Measurement Calculation Flow

```
front.obj + side.obj
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              Mesh Slicing (trimesh)                 │
│                                                     │
│  Slice at % from top:                               │
│  • 7%  → neck                                       │
│  • 28% → chest                                      │
│  • 42% → waist                                      │
│  • 58% → hip                                        │
│                                                     │
│  width  = front mesh x-range at slice              │
│  depth  = side mesh z-range at slice               │
│                                                     │
│  Ramanujan Ellipse: π(3(a+b) - √((3a+b)(a+3b)))   │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
        Layer 1: Legacy weight/age/height adjustments
        (adjust_chest_by_weight, adjust_waist_*, etc.)
                           │
                           ▼
        Layer 2: MeasurementCorrectionEngine
        (global scale from BMI × age × muscle × activity × goal)
                           │
                           ▼
        Layer 3: Derived measurements
        (armhole = chest × ratio, thigh = hip × ratio, etc.)
                           │
                           ▼
        Layer 4: Targeted correction (final override for
        specific age/height/weight clusters)
                           │
                           ▼
              Final measurements JSON
```

---

## Component Reference

### `PoseValidator`
Validates standing posture using MediaPipe Pose landmarks.

| Method | Description |
|---|---|
| `check_front_view(path)` | Measures shoulder → hip → knee angle; accepts if ≥ 160° |
| `check_side_view(path)` | Measures shoulder → hip → ankle angle on more-visible side |
| `validate_images(front, side)` | Runs both checks; returns `(overall_success, detailed_results)` |

**Threshold:** `WAIST_ANGLE_THRESHOLD = 160°`

---

### `CompleteBodyMeasurementsCalculator`
Core measurement engine. Initialized with all user parameters.

| Parameter | Type | Options |
|---|---|---|
| `gender` | str | `male`, `female` |
| `weight` | float | kg |
| `height` | float | cm |
| `age` | int | 5–120 |
| `age_group` | str | `teen`, `adult`, `middle_age`, `senior` |
| `fat_distribution` | str | `upper`, `middle`, `lower`, `even` |
| `muscle_level` | str | `low`, `moderate`, `high`, `very_high` |
| `activity_level` | str | `sedentary`, `light`, `moderate`, `active`, `very_active` |
| `shoulder_type` | str | `narrow`, `average`, `broad`, `very_broad` |
| `measurement_goal` | str | `health`, `clothing`, `fitness`, `general` |
| `fit_preference` | str | `tight`, `regular`, `loose`, `oversized` |

**Key method:** `calculate_all_measurements(front_obj, side_obj) → Dict`

---

### `MeasurementCorrectionEngine`
Applies a single multiplicative global scale derived from all user characteristics.

```
final_delta = BASE_DELTA[target] × AGE_SCALE × BMI_SCALE × MUSCLE_SCALE × ACTIVITY_SCALE × GOAL_SCALE

Capped at MAX_DELTA per measurement type.
```

| Scale Table | Values |
|---|---|
| `BMI_SCALE` | underweight=0.0, normal=1.0, overweight=1.3, obese=1.6 |
| `AGE_SCALE` | teen=0.0, adult=1.0, middle_age=1.2, senior=1.4 |
| `MUSCLE_SCALE` | low=1.1, moderate=1.0, high=0.7, very_high=0.6 |
| `ACTIVITY_SCALE` | sedentary=1.1 → very_active=0.6 |
| `FIT_SCALE` | tight=0.97, regular=1.0, loose=1.05, oversized=1.10 |

Hard safety caps: `waist ≤ 2.0 cm`, `hip ≤ 3.0 cm`, `armhole ≤ 2.0 cm`

---

### `MeasurementCorrector`
Legacy table-based corrections applied by `(body_type, bmi_category)` lookup before the engine runs.

Covers all combinations of:
- Male types: `inverted_triangle`, `triangle`, `rectangle`, `trapezoid`
- Female types: `hourglass`, `pear`, `apple`, `inverted_triangle`, `rectangle`
- BMI categories: `underweight`, `normal`, `overweight`, `obese`

---

### `BodyTypeClassifier`
Classifies body shape from chest, waist, hip, and shoulder measurements.

| Gender | Body Types |
|---|---|
| Male | `inverted_triangle`, `triangle`, `rectangle`, `trapezoid` |
| Female | `hourglass`, `pear`, `apple`, `inverted_triangle`, `rectangle` |

---

### `ClothingSizeRecommender`
Rule-based size from average of chest, waist, hip measurements (in inches).

| Size | Avg Measurement Range |
|---|---|
| XS | < 29" avg |
| S | ~29–31" avg |
| M | ~31–33" avg |
| L | ~33–35" avg |
| XL | ~35–37" avg |
| XXL | ~37–41" avg |
| XXXL | > 41" avg |

---

### `VirtualTryOnService`
Handles the full LightX API flow.

| Method | Description |
|---|---|
| `get_upload_url(path)` | Requests pre-signed S3 URL from LightX |
| `upload_image(url, path)` | PUTs image bytes to pre-signed URL |
| `start_virtual_tryon(person_url, outfit_url, type)` | Submits try-on job, returns `order_id` |
| `check_status(order_id, max_attempts=60)` | Polls every 3s; returns result URL when `status=active` |
| `download_result_image(url, path)` | Streams result image to disk |

**Clothing type:** `0 = upper body`, `1 = lower body`, `2 = full outfit`

---

### `StageTimer`
Lightweight per-request stage profiler with request ID correlation.

```python
timer = StageTimer(req_id)
with timer.stage("pose_validation"):
    ...
with timer.stage("hmr2_parallel"):
    ...
timer.log_summary()
# [abc12345] ╕ TOTAL 18.34s ┊ pose_validation=0.82s | hmr2_parallel=15.20s | ...
```

---

## Measurement Pipeline

### Output Measurements

| Measurement | Unit | Notes |
|---|---|---|
| `neck` | cm / inches | Circumference |
| `chest` | cm / inches | Full bust/chest circumference |
| `upper_chest` | cm / inches | Female only |
| `lower_chest` | cm / inches | Female only |
| `waist` | cm / inches | Natural waist circumference |
| `hip` | cm / inches | Full hip circumference |
| `shoulder` | cm / inches | Biacromial width |
| `armhole` | cm / inches | Derived from chest × ratio |
| `upper_thigh` | cm / inches | Derived from hip × ratio |
| `knee` | cm / inches | Derived from thigh × ratio |
| `arm.total_length` | cm / inches | Full arm length |
| `arm.hand_to_elbow` | cm / inches | Forearm |
| `arm.shoulder_to_elbow` | cm / inches | Upper arm |
| `body_length` | cm / inches | Nape to waist |
| `metadata.bmi` | float | Calculated BMI |
| `metadata.bmi_category` | string | underweight/normal/overweight/obese |
| `metadata.body_type` | string | Classified shape |
| `metadata.recommended_size` | string | XS – XXXL |

### Ellipse Formula (Ramanujan)

```
Circumference ≈ π × (3(a+b) − √((3a+b)(a+3b)))

where:
  a = adjusted half-width  (from mesh x-range)
  b = adjusted half-depth  (from mesh z-range)
  
Adjustment ratios vary by measurement type and gender.
```

---

## Correction Engine

The correction pipeline has **4 sequential layers**:

```
Layer 1 — Legacy Adjustments (per measurement, per demographic cluster)
  Hardcoded offsets/scales for specific (age, height, weight) clusters
  e.g. adjust_chest_by_weight(), adjust_waist_by_weight_female()

Layer 2 — MeasurementCorrectionEngine (single global scale)
  delta = BASE_DELTA × AGE × BMI × MUSCLE × ACTIVITY × GOAL
  Applied to waist (primary target), armhole (shoulder type)
  Final fit_preference scale applied to all non-protected measurements

Layer 3 — Derived Ratio Calculations
  armhole = chest × ratio (0.42 male / 0.44 female, adjusted per cluster)
  upper_thigh = hip × ratio (0.55 male / 0.60–0.72 female)
  knee = thigh × ratio (0.72 male / 0.63–0.75 female)

Layer 4 — Targeted Correction (final override)
  For specific age/height/weight clusters, hard-set critical measurements
  to validated target values within 0.5-inch tolerance
```

---

## API Reference

### `POST /process`

Generate body measurements from front and side images.

**Form Fields:**

| Field | Type | Required | Values |
|---|---|---|---|
| `front_image` | File | ✅ | JPG, PNG, JPEG |
| `side_image` | File | ✅ | JPG, PNG, JPEG |
| `gender` | string | ✅ | `male`, `female` |
| `height` | float | ✅ | e.g. `175` |
| `height_unit` | string | ✅ | `cm`, `m` |
| `weight` | float | ✅ | e.g. `70` |
| `weight_unit` | string | ✅ | `kg`, `lbs` |
| `age` | int | ✅ | 5–120 |
| `body_type` | string | ✅ | `slim`, `avg`, `athletic`, `heavy`, `curvy` |
| `age_group` | string | ✅ | `teen`, `adult`, `middle_age`, `senior` |
| `fat_distribution` | string | ✅ | `upper`, `middle`, `lower`, `even` |
| `muscle_level` | string | ✅ | `low`, `moderate`, `high`, `very_high` |
| `activity_level` | string | ✅ | `sedentary`, `light`, `moderate`, `active`, `very_active` |
| `shoulder_type` | string | ✅ | `narrow`, `average`, `broad`, `very_broad` |
| `measurement_goal` | string | ✅ | `health`, `clothing`, `fitness`, `general` |
| `fit_preference` | string | ✅ | `tight`, `regular`, `loose`, `oversized` |

**Image Constraints:**
- Min dimension: 200px (shorter side)
- Max dimension: 4096px (longer side)
- Max size: 12 megapixels
- Aspect ratio: 0.25 – 1.5 (portrait / near-square only)
- Auto-resized to max 1024px long side before inference

**Success Response:**
```json
{
  "success": true,
  "measurements": {
    "chest": {"circumference": {"cm": 96.52, "inches": 38.0}},
    "waist": {"circumference": {"cm": 81.28, "inches": 32.0}},
    "hip":   {"circumference": {"cm": 99.06, "inches": 39.0}},
    "shoulder": {"width": {"cm": 42.5, "inches": 16.73}},
    "armhole": {"circumference": {"cm": 42.5, "inches": 16.73}},
    "upper_thigh": {"circumference": {"cm": 57.15, "inches": 22.5}},
    "knee": {"circumference": {"cm": 38.1, "inches": 15.0}},
    "arm": {
      "total_length": {"cm": 67, "inches": 26},
      "hand_to_elbow": {"cm": 31.5, "inches": 12.4},
      "shoulder_to_elbow": {"cm": 36.5, "inches": 14.4}
    },
    "body_length": {"length": {"cm": 38.1, "inches": 15.0}},
    "metadata": {
      "bmi": 22.5,
      "bmi_category": "normal",
      "body_type": "hourglass",
      "recommended_size": "M"
    }
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "code": "INVALID_HEIGHT_RANGE",
  "error": "Height must be between 100 and 250 cm.",
  "request_id": "abc12345"
}
```

---

### `POST /virtual-tryon-process`

Run virtual try-on with LightX API.

**Form Fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `person_image` | File | ✅ | Full-body photo of person |
| `clothing_image` | File | ✅ | Photo of clothing item |
| `clothing_type` | int | ✅ | `0`=upper, `1`=lower, `2`=full |

**Success Response:**
```json
{
  "success": true,
  "result_image_url": "/tryon-result/tryon_result_20240101_120000.jpg",
  "download_url": "/download-tryon/tryon_result_20240101_120000.jpg",
  "filename": "tryon_result_20240101_120000.jpg"
}
```

---

### `GET /tryon-result/<filename>`
Serve the try-on result image inline.

### `GET /download-tryon/<filename>`
Download the try-on result image as attachment.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/body-measurement-app.git
cd body-measurement-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install base dependencies
pip install flask opencv-python mediapipe trimesh numpy pillow \
            requests python-dotenv werkzeug piexif

# Install HMR2 (4D-Humans)
pip install git+https://github.com/shubham-goel/4D-Humans.git

# Install Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Download HMR2 model weights (runs automatically on first use)
# Weights download to ~/.cache/4DHumans/
```

---

## Configuration

Create a `.env` file in the project root:

```env
LIGHTX_API_KEY=your_lightx_api_key_here
```

**App defaults** (configurable in code):

| Config | Default | Description |
|---|---|---|
| `UPLOAD_FOLDER` | `uploads/` | Temporary upload storage |
| `OUTPUT_FOLDER` | `output/` | Temporary OBJ mesh storage |
| `TRYON_FOLDER` | `tryon_results/` | Try-on result storage |
| `MAX_CONTENT_LENGTH` | 16 MB | Max upload file size |
| `IMG_TARGET_LONG_SIDE` | 1024 px | Resize before inference |
| `IMG_MAX_MEGAPIXELS` | 12 MP | Pixel budget cap |
| `WAIST_ANGLE_THRESHOLD` | 160° | Minimum standing angle |
| `MAX_INFERENCE_QUEUE` | 4 | Max queued inference requests |

---

## Usage

```bash
# Start the server
python measurement.py

# Server runs at: http://0.0.0.0:5000
```

### Example API call (curl)

```bash
curl -X POST http://localhost:5000/process \
  -F "front_image=@/path/to/front.jpg" \
  -F "side_image=@/path/to/side.jpg" \
  -F "gender=female" \
  -F "height=164" \
  -F "height_unit=cm" \
  -F "weight=60" \
  -F "weight_unit=kg" \
  -F "age=28" \
  -F "body_type=avg" \
  -F "age_group=adult" \
  -F "fat_distribution=even" \
  -F "muscle_level=moderate" \
  -F "activity_level=moderate" \
  -F "shoulder_type=average" \
  -F "measurement_goal=clothing" \
  -F "fit_preference=regular"
```

### Python client example

```python
import requests

files = {
    'front_image': open('front.jpg', 'rb'),
    'side_image': open('side.jpg', 'rb'),
}
data = {
    'gender': 'female', 'height': '164', 'height_unit': 'cm',
    'weight': '60', 'weight_unit': 'kg', 'age': '28',
    'body_type': 'avg', 'age_group': 'adult',
    'fat_distribution': 'even', 'muscle_level': 'moderate',
    'activity_level': 'moderate', 'shoulder_type': 'average',
    'measurement_goal': 'clothing', 'fit_preference': 'regular'
}

resp = requests.post('http://localhost:5000/process', files=files, data=data)
measurements = resp.json()['measurements']
print(f"Chest: {measurements['chest']['circumference']['inches']} inches")
print(f"Size:  {measurements['metadata']['recommended_size']}")
```

---

## Virtual Try-On

```
User uploads:  person_image + clothing_image + clothing_type
                        │
                        ▼
           VirtualTryOnService.get_upload_url()
           → LightX returns pre-signed S3 URL
                        │
                        ▼
           VirtualTryOnService.upload_image()
           → PUT image bytes to S3
                        │
                        ▼
           VirtualTryOnService.start_virtual_tryon()
           → POST to /aivirtualtryon → returns order_id
                        │
                        ▼
           VirtualTryOnService.check_status()
           → Poll /order-status every 3s (max 60 attempts = 3 min)
           → Returns result URL when status == "active"
                        │
                        ▼
           VirtualTryOnService.download_result_image()
           → Save result to tryon_results/
                        │
                        ▼
           Return { result_image_url, download_url }
```

---

## Concurrency & Safety

### Inference Queue
A single daemon worker thread serializes all HMR2 model calls to prevent race conditions:

```python
run_inference(fn, *args, timeout=300)
# ├── Raises queue.Full  → HTTP 503 "Server busy"
# └── Raises TimeoutError → HTTP 504 "Processing timed out"
```

### Image Validation Pipeline
Every upload goes through 4 checks before touching the model:

```
1. Extension check      → .jpg/.png/.jpeg only
2. MIME type check      → image/jpeg or image/png only
3. cv2.imdecode()       → confirms real image bytes (not spoofed)
4. Dimension check      → min 200px, max 4096px, max 12MP, aspect 0.25–1.5
5. normalize_image()    → strips alpha, applies EXIF rotation, resizes to 1024px
```

### Cleanup
All temporary files (uploaded images, OBJ meshes) are deleted in `finally` blocks after every request, regardless of success or failure.

---

## Project Structure

```
body-measurement-app/
├── measurement.py              # Main Flask app (all components)
├── .env                        # API keys (not committed)
├── requirements.txt
├── templates/
│   ├── index.html              # Main measurement UI
│   ├── virtual_try_on.html     # Try-on UI
│   ├── about.html
│   └── contact.html
├── uploads/                    # Temp: uploaded images (auto-cleaned)
├── output/                     # Temp: OBJ mesh files (auto-cleaned)
└── tryon_results/              # Persistent: try-on result images
```

---

## Error Codes Reference

| Code | HTTP | Meaning |
|---|---|---|
| `MISSING_GENDER` | 400 | `gender` field not provided |
| `INVALID_GENDER` | 400 | Not `male` or `female` |
| `MISSING_HEIGHT` / `INVALID_HEIGHT_FORMAT` | 400 | Height missing or non-numeric |
| `INVALID_HEIGHT_RANGE` | 400 | Outside 100–250 cm |
| `MISSING_WEIGHT` / `INVALID_WEIGHT_FORMAT` | 400 | Weight missing or non-numeric |
| `INVALID_WEIGHT_RANGE` | 400 | Outside 30–300 kg |
| `MISSING_AGE` / `INVALID_AGE_FORMAT` | 400 | Age missing or non-integer |
| `INVALID_AGE_RANGE` | 400 | Outside 5–120 years |
| `INVALID_BODY_TYPE` | 400 | Not in allowed set |
| `IMAGE_SAVE_FAILED` | 500 | cv2.imwrite failed |
| *(queue full)* | 503 | Inference queue at capacity |
| *(timeout)* | 504 | HMR2 inference exceeded 300s |

---

## Dependencies

```
flask
opencv-python
mediapipe
trimesh
numpy
pillow
piexif
requests
python-dotenv
werkzeug
torch
4D-Humans (HMR2)
detectron2
```