"""Export Layered-Labs/ACNE04 to Ultralytics YOLO detect format.

Creates:
  data/acne04/
    images/{train,val}/...
    labels/{train,val}/...
    acne04.yaml

Usage (from backend/):
  .venv\\Scripts\\python scripts/export_acne04_yolo.py
  .venv\\Scripts\\python scripts/export_acne04_yolo.py --val-ratio 0.15 --max-samples 200

License: ACNE04 is academic use only — keep this prototype internal.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = BACKEND_ROOT / "data" / "acne04"

# Short path on E: — C: is full, long Windows paths break datasets locks
_HF_HOME = Path("E:/hf")
_HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_HOME))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from datasets import load_dataset  # noqa: E402
from PIL import Image  # noqa: E402


def _as_list(value) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _extract_boxes(row: dict) -> list[list[float]]:
    """Return list of [x1, y1, x2, y2] in absolute pixel coords.

    ACNE04 uses parallel lists:
      lesion_boxes = {xmin:[…], ymin:[…], xmax:[…], ymax:[…]}
    """
    raw = row.get("lesion_boxes") or row.get("bboxes") or row.get("boxes")
    if isinstance(raw, dict) and {"xmin", "ymin", "xmax", "ymax"} <= set(raw):
        xs = _as_list(raw["xmin"])
        ys = _as_list(raw["ymin"])
        xe = _as_list(raw["xmax"])
        ye = _as_list(raw["ymax"])
        n = min(len(xs), len(ys), len(xe), len(ye))
        return [[float(xs[i]), float(ys[i]), float(xe[i]), float(ye[i])] for i in range(n)]

    if raw is None:
        return []
    boxes = _as_list(raw)
    if not boxes:
        return []
    first = boxes[0]
    if isinstance(first, dict):
        out = []
        for b in boxes:
            if {"xmin", "ymin", "xmax", "ymax"} <= set(b):
                out.append([b["xmin"], b["ymin"], b["xmax"], b["ymax"]])
            elif {"x1", "y1", "x2", "y2"} <= set(b):
                out.append([b["x1"], b["y1"], b["x2"], b["y2"]])
            elif "bbox" in b:
                out.append(list(b["bbox"][:4]))
        return out
    if isinstance(first, (list, tuple)) and len(first) >= 4:
        return [list(b[:4]) for b in boxes]
    return []


def _xyxy_to_yolo(
    box: list[float],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float] | None:
    """Normalize absolute xyxy box to YOLO cx,cy,w,h."""
    if len(box) < 4:
        return None
    x1, y1, x2, y2 = map(float, box[:4])
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw < 1 or bh < 1:
        return None
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h
    return (
        min(max(cx, 0.0), 1.0),
        min(max(cy, 0.0), 1.0),
        min(max(nw, 1e-6), 1.0),
        min(max(nh, 1e-6), 1.0),
    )


def _get_image(row: dict) -> Image.Image:
    for key in ("image", "img", "jpg", "png"):
        if key in row and row[key] is not None:
            img = row[key]
            if isinstance(img, Image.Image):
                return img.convert("RGB")
            return Image.open(img).convert("RGB")
    raise KeyError(f"No image field in row keys={list(row.keys())}")


def _write_yaml(out_dir: Path) -> None:
    yaml_path = out_dir / "acne04.yaml"
    # Ultralytics resolves paths relative to the yaml file location.
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: acne",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {yaml_path}")


def export(
    out_dir: Path,
    val_ratio: float,
    seed: int,
    max_samples: int | None,
    force: bool,
) -> None:
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("Loading Layered-Labs/ACNE04 …")
    ds = load_dataset("Layered-Labs/ACNE04")
    # HF ships a single "train" split; use the row-level `split` column (train/test).
    base = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    train_idx: list[int] = []
    val_idx: list[int] = []
    for i in range(len(base)):
        row_split = str(base[i].get("split") or "").lower()
        if row_split in ("test", "val", "validation"):
            val_idx.append(i)
        else:
            train_idx.append(i)

    # Fallback if `split` column is missing/empty
    if not val_idx:
        all_idx = list(range(len(base)))
        random.Random(seed).shuffle(all_idx)
        n_val = max(1, int(len(all_idx) * val_ratio))
        val_idx = all_idx[:n_val]
        train_idx = all_idx[n_val:]

    if max_samples is not None:
        train_idx = train_idx[:max_samples]
        val_idx = val_idx[: max(1, max_samples // 5)]

    splits = {
        "train": base.select(train_idx),
        "val": base.select(val_idx),
    }
    print(f"Using row split column -> train={len(train_idx)} val={len(val_idx)}")

    stats = {"train": 0, "val": 0, "boxes": 0, "skipped_no_box": 0}
    schema_sample: dict | None = None

    for split_name, split in splits.items():
        print(f"Exporting {split_name}: {len(split)}")
        for i in range(len(split)):
            row = split[i]
            if schema_sample is None:
                schema_sample = {k: type(v).__name__ for k, v in row.items()}

            img = _get_image(row)
            w, h = img.size
            boxes = _extract_boxes(row)
            yolo_lines: list[str] = []
            for box in boxes:
                yolo = _xyxy_to_yolo(box, w, h)
                if yolo is None:
                    continue
                cx, cy, nw, nh = yolo
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                stats["boxes"] += 1

            if not yolo_lines:
                stats["skipped_no_box"] += 1
                continue

            stem = Path(str(row.get("file_name") or f"{split_name}_{i:05d}")).stem
            stem = f"{split_name}_{stem}"
            img_path = out_dir / "images" / split_name / f"{stem}.jpg"
            lbl_path = out_dir / "labels" / split_name / f"{stem}.txt"
            img.save(img_path, quality=95)
            lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            stats[split_name] += 1

    _write_yaml(out_dir)
    meta = {
        "source": "Layered-Labs/ACNE04",
        "license_note": "academic use only",
        "class_names": ["acne"],
        "schema_sample_types": schema_sample,
        "stats": stats,
        "val_ratio": val_ratio,
        "seed": seed,
        "max_samples": max_samples,
    }
    (out_dir / "export_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("Done:", json.dumps(stats, indent=2))
    print(f"Schema field types: {schema_sample}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ACNE04 → YOLO")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per split (useful for a smoke export)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing out dir before export",
    )
    args = parser.parse_args()
    export(args.out, args.val_ratio, args.seed, args.max_samples, args.force)


if __name__ == "__main__":
    main()
