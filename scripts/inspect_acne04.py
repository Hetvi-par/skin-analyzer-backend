"""Inspect Layered-Labs/ACNE04 schema (run once before export).

Usage (from backend/):
  .venv\\Scripts\\python scripts/inspect_acne04.py
"""

from __future__ import annotations

import os
from pathlib import Path

# Short path on E: — C: is full, and long Windows paths break datasets locks
_HF_HOME = Path("E:/hf")
_HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_HOME))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from datasets import load_dataset  # noqa: E402


def main() -> None:
    print(f"HF_HOME={os.environ['HF_HOME']}")
    print("Loading Layered-Labs/ACNE04 …")
    ds = load_dataset("Layered-Labs/ACNE04")
    print("\n=== Dataset splits ===")
    print(ds)

    for split_name, split in ds.items():
        print(f"\n=== Split: {split_name} ({len(split)} rows) ===")
        print("Features:", split.features)
        sample = split[0]
        print("Keys:", list(sample.keys()))
        for key, value in sample.items():
            if key in ("image", "img"):
                img = value
                size = getattr(img, "size", None)
                mode = getattr(img, "mode", None)
                print(f"  {key}: PIL size={size} mode={mode}")
            else:
                text = repr(value)
                if len(text) > 240:
                    text = text[:240] + "…"
                print(f"  {key}: {text}")
        # Peek a few more rows for box shape consistency
        for i in range(1, min(4, len(split))):
            row = split[i]
            boxes = (
                row.get("lesion_boxes")
                or row.get("bboxes")
                or row.get("objects")
                or row.get("annotations")
            )
            severity = row.get("severity") or row.get("label") or row.get("grade")
            print(f"  row[{i}] severity={severity!r} boxes_type={type(boxes).__name__}")


if __name__ == "__main__":
    main()
