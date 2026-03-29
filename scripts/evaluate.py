#!/usr/bin/env python3
"""
Module 3 - Evaluation pipeline for trained detectors.

Computes:
  - mAP@0.5, mAP@0.5:0.95
  - precision, recall, F1
Also saves:
  - metrics JSON
  - append row to CSV
  - sample prediction images
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import yaml

try:
    from ultralytics import YOLO
except ImportError as e:
    print("Install requirements first: pip install -r requirements-training.txt", file=sys.stderr)
    raise SystemExit(1) from e


def _resolve_data_path(project_root: Path, data_arg: str) -> Path:
    p = Path(data_arg)
    return p if p.is_absolute() else (project_root / p).resolve()


def _resolve_split_dir(data_yaml: Path, split: str) -> Path:
    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    base = data_yaml.parent
    path_root = data.get("path")
    if path_root:
        root = Path(path_root)
        if not root.is_absolute():
            root = (base / root).resolve()
    else:
        root = base

    split_rel = data.get(split)
    if not split_rel:
        raise ValueError(f"Split '{split}' is missing in {data_yaml}")

    split_path = Path(split_rel)
    if not split_path.is_absolute():
        split_path = (root / split_path).resolve()
    return split_path


def _append_csv_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--weights", type=str, required=True, help=".pt model path")
    parser.add_argument("--data", type=str, default="dataset/data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default=None, help="e.g. 0, cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--model-name", type=str, default=None, help="Label used in reports")
    parser.add_argument("--num-sample-preds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--csv-path", type=str, default="results/metrics_summary.csv")
    args = parser.parse_args()

    root = args.project_root.resolve()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = (root / weights).resolve()
    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        return 1

    data_yaml = _resolve_data_path(root, args.data)
    if not data_yaml.is_file():
        print(f"Dataset YAML not found: {data_yaml}", file=sys.stderr)
        return 1

    model_name = args.model_name or weights.stem
    results_dir = (root / args.results_dir).resolve()
    eval_dir = results_dir / "evaluations" / model_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        save_json=False,
    )

    rd = dict(metrics.results_dict)
    precision = float(rd.get("metrics/precision(B)", 0.0))
    recall = float(rd.get("metrics/recall(B)", 0.0))
    map50 = float(rd.get("metrics/mAP50(B)", 0.0))
    map5095 = float(rd.get("metrics/mAP50-95(B)", 0.0))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    speed = dict(getattr(metrics, "speed", {}))
    sp_pre = float(speed.get("preprocess", 0.0))
    sp_inf = float(speed.get("inference", 0.0))
    sp_post = float(speed.get("postprocess", 0.0))
    sp_total = sp_pre + sp_inf + sp_post
    fps = (1000.0 / sp_total) if sp_total > 0 else 0.0

    summary = {
        "model": model_name,
        "weights": str(weights),
        "data": str(data_yaml),
        "split": args.split,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP50": map50,
        "mAP50_95": map5095,
        "latency_pre_ms": sp_pre,
        "latency_inf_ms": sp_inf,
        "latency_post_ms": sp_post,
        "latency_total_ms": sp_total,
        "fps_est": fps,
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }

    summary_json = eval_dir / "metrics_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = (root / csv_path).resolve()
    _append_csv_row(
        csv_path,
        {
            "model": model_name,
            "split": args.split,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mAP50": map50,
            "mAP50_95": map5095,
            "latency_pre_ms": sp_pre,
            "latency_inf_ms": sp_inf,
            "latency_post_ms": sp_post,
            "latency_total_ms": sp_total,
            "fps_est": fps,
            "weights": str(weights),
            "save_dir": str(getattr(metrics, "save_dir", "")),
        },
    )

    split_dir = _resolve_split_dir(data_yaml, args.split)
    image_paths = [p for p in split_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if image_paths:
        rng = random.Random(args.seed)
        picks = rng.sample(image_paths, k=min(args.num_sample_preds, len(image_paths)))
        preds_dir = eval_dir / "sample_predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        model.predict(
            source=[str(p) for p in picks],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save=True,
            project=str(preds_dir),
            name="preds",
            exist_ok=True,
            verbose=False,
        )

    print("Evaluation complete")
    print(json.dumps(summary, indent=2))
    print(f"Saved JSON: {summary_json}")
    print(f"Updated CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
