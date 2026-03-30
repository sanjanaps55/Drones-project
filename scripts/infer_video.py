#!/usr/bin/env python3
"""
Run object detection on a video and save an output video with boxes drawn.

Uses Ultralytics predict(); outputs are written under results/inference/<run_name>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Install: pip install -r requirements-training.txt", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Video inference with saved visualizations")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: current directory).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Trained weights (.pt), e.g. models/yolov8n_visdrone_best.pt",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input video (mp4, avi, etc.).",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu, 0, mps, ...")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--run-name",
        type=str,
        default="video_detect",
        help="Subfolder under results/inference/",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Also save YOLO txt labels per frame (optional).",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = (root / weights).resolve()
    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        return 1

    source = Path(args.source)
    if not source.is_file():
        print(f"Video not found: {source}", file=sys.stderr)
        return 1

    out_base = root / "results" / "inference"
    out_base.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    model.predict(
        source=str(source.resolve()),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        project=str(out_base),
        name=args.run_name,
        exist_ok=True,
    )

    print(f"Done. Look for annotated video under: {out_base / args.run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
