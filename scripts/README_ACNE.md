# ACNE04 → YOLOv8 (academic use only)

ACNE04 license is **academic / non-commercial**. Keep weights and data internal.

## What you already have

- HuggingFace dataset: `Layered-Labs/ACNE04` (1457 images)
- Base weights: `skin-analysis/docs/yolov8n.pt` (COCO pretrained — must be fine-tuned)

### ACNE04 schema (inspected)

| Field | Type | Notes |
|-------|------|--------|
| `image` | PIL | face photo |
| `severity` | string | e.g. mild / moderate |
| `level` | int | severity grade |
| `lesion_boxes` | dict of lists | `xmin/ymin/xmax/ymax` (absolute xyxy) |
| `split` | string | `train` / `test` (used for YOLO val) |

One class for detection: `acne` (all lesion boxes).

## Google Colab (recommended if no modern NVIDIA GPU)

Upload / open this notebook and run all cells with a **GPU** runtime:

`skin-analysis/docs/ACNE04_YOLOv8_Colab.ipynb`

It exports ACNE04, trains YOLOv8n, and downloads `acne_yolov8n.pt`.

## Setup (local)

```bash
cd backend
.venv\Scripts\activate
pip install -r requirements.txt
```

HF downloads go to `E:\hf` (short path; C: is often full).

## 1) Inspect schema

```bash
python scripts/inspect_acne04.py
```

## 2) Export to YOLO format

Full export:

```bash
python scripts/export_acne04_yolo.py --force
```

Smoke export (fast):

```bash
python scripts/export_acne04_yolo.py --force --max-samples 100
```

Output:

```
backend/data/acne04/
  images/train|val/
  labels/train|val/
  acne04.yaml
  export_meta.json
```

## 3) Train

```bash
pip install ultralytics
python scripts/train_acne_yolo.py --epochs 50 --device cpu
```

Best weights are copied to `backend/models/acne_yolov8n.pt`.

## 4) Next (API)

Wire `acne_yolov8n.pt` into the pipeline: run detect on forehead/cheek crops → lesion count → severity 0–4 → JSON + UI.
