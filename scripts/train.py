#!/usr/bin/env python3
"""
Module 2 — Train Ultralytics YOLOv8 on the prepared VisDrone YOLO dataset.

Prerequisites:
  - Run scripts/prepare_dataset.py so dataset/data.yaml, images/, and labels/ exist.
  - pip install -r requirements-training.txt

Metrics and curves are logged under runs/ (Ultralytics default). Best weights are
copied to models/ for later evaluation, export, and compression.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

try:
    from ultralytics import YOLO
except ImportError as e:
    print("Install Ultralytics: pip install -r requirements-training.txt", file=sys.stderr)
    raise SystemExit(1) from e


def load_train_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on VisDrone (YOLO format)")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root (contains dataset/, configs/, runs/). Default: cwd.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML with defaults (default: <project-root>/configs/train.yaml)",
    )
    parser.add_argument("--model", type=str, default=None, help="Checkpoint e.g. yolov8n.pt")
    parser.add_argument("--data", type=str, default=None, help="dataset YAML path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="e.g. 0, cpu, mps")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None, help="early stopping patience")
    parser.add_argument("--run-name", type=str, default=None, help="Ultralytics run name")
    parser.add_argument(
        "--no-copy-best",
        action="store_true",
        help="Do not copy best.pt to models/ (see weights_out in config).",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    cfg_path = args.config or (root / "configs" / "train.yaml")
    if not cfg_path.is_file():
        print(f"Missing config: {cfg_path}", file=sys.stderr)
        return 1

    cfg = load_train_config(cfg_path)

    model_name = args.model or cfg["model"]
    data_arg = args.data or cfg["data"]
    data_path = Path(data_arg)
    if not data_path.is_absolute():
        data_path = (root / data_path).resolve()
    if not data_path.is_file():
        print(f"Dataset YAML not found: {data_path}", file=sys.stderr)
        return 1

    epochs = args.epochs if args.epochs is not None else int(cfg["epochs"])
    batch = args.batch if args.batch is not None else int(cfg["batch"])
    imgsz = args.imgsz if args.imgsz is not None else int(cfg["imgsz"])
    workers = args.workers if args.workers is not None else int(cfg["workers"])
    patience = args.patience if args.patience is not None else int(cfg["patience"])
    device = args.device if args.device is not None else cfg.get("device")
    if device is None or (isinstance(device, str) and str(device).lower() == "null"):
        device = None  # Ultralytics auto

    project = root / str(cfg["project"]).replace("\\", "/")
    name = args.run_name or str(cfg["name"])
    weights_out = root / str(cfg["weights_out"]).replace("\\", "/")

    print(f"Data:   {data_path}")
    print(f"Model:  {model_name}")
    print(f"epochs={epochs} batch={batch} imgsz={imgsz} device={device!r}")

    model = YOLO(model_name)
    model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        workers=workers,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
    )

    save_dir = Path(getattr(model.trainer, "save_dir", project / name))
    best_pt = save_dir / "weights" / "best.pt"
    if not args.no_copy_best and best_pt.is_file():
        weights_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, weights_out)
        print(f"Copied best weights → {weights_out}")

    print(f"Training finished. Run artifacts: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
