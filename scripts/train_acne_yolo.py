"""Fine-tune YOLOv8n on exported ACNE04 (academic use only).

Prereqs:
  1. pip install ultralytics datasets
  2. python scripts/export_acne04_yolo.py
  3. Place base weights at docs/yolov8n.pt (or pass --weights)

Usage (from backend/):
  .venv\\Scripts\\python scripts/train_acne_yolo.py
  .venv\\Scripts\\python scripts/train_acne_yolo.py --epochs 30 --imgsz 640

Best weights are copied to models/acne_yolov8n.pt for the API.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
DEFAULT_DATA = BACKEND_ROOT / "data" / "acne04" / "acne04.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "docs" / "yolov8n.pt"
DEFAULT_OUT_MODEL = BACKEND_ROOT / "models" / "acne_yolov8n.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train acne YOLOv8 on ACNE04")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu / 0 / 0,1 …")
    parser.add_argument("--project", type=Path, default=BACKEND_ROOT / "runs" / "acne")
    parser.add_argument("--name", default="yolov8n_acne04")
    parser.add_argument("--out-model", type=Path, default=DEFAULT_OUT_MODEL)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Missing {args.data}. Run: python scripts/export_acne04_yolo.py"
        )
    if not args.weights.exists():
        raise SystemExit(
            f"Missing base weights at {args.weights}. "
            "Put yolov8n.pt under skin-analysis/docs/ or pass --weights."
        )

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )

    # Ultralytics saves best.pt under project/name/weights/
    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (
        args.project / args.name
    )
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        # Fallback: last.pt
        best = run_dir / "weights" / "last.pt"
    if not best.exists():
        raise SystemExit(f"Training finished but no weights found under {run_dir}")

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.out_model)
    print(f"Copied {best} → {args.out_model}")


if __name__ == "__main__":
    main()
