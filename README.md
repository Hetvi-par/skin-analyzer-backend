# Skin Analysis — Backend

FastAPI service for face landmarks (MediaPipe) and skin masking (ResNet18 ONNX).

## Setup

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### Models

Place models under `backend/models/`:

| File | Purpose |
|------|---------|
| `face_landmarker.task` | MediaPipe Face Landmarker |
| `resnet18.onnx` | Face parsing / skin mask (19 classes, 512×512) |

Download Face Landmarker if missing:

```text
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API root: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/health
- Initial analyze: `POST http://localhost:8000/api/v1/analyze/initial`

## Analysis pipeline

1. **MediaPipe** detects face mesh landmarks
2. **ResNet18 ONNX** produces a 19-class face-parsing map
3. Skin mask is built from `skin` + `nose` classes
4. **Region crops** from landmark polygons ∩ skin mask:
   forehead, left cheek, right cheek, nose, chin
5. **CIELAB color metrics** — redness (a*) and pigmentation (dark L*) on a 0–4 scale
6. **YOLOv8 acne** — lesion detection on region crops → severity 0–4
7. **Texture heuristics** — pores (Laplacian/high-pass on cheeks/nose) and wrinkles
   (horizontal edges on forehead/chin) → visibility 0–4
8. **Skin tone** — Lab median on forehead/cheeks → ITA depth + warm/cool undertone
9. Response includes previews, region crops, color, acne, texture, and tone

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/initial" ^
  -F "file=@face.jpg"
```

## Project structure

```
backend/
├── app/
│   ├── api/           # Routes
│   ├── core/          # Settings
│   ├── schemas/       # Response models
│   ├── services/      # Landmarks, masking, pipeline
│   ├── utils/         # Image helpers
│   └── main.py
├── models/            # .task + .onnx weights
├── uploads/
└── requirements.txt
```
