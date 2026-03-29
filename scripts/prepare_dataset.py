#!/usr/bin/env python3
"""
Module 1 — VisDrone DET → YOLO format dataset preparation.

Steps:
  1. Read VisDrone annotation lines (comma-separated).
  2. Skip ignored boxes (score==0) and unmapped categories.
  3. Map VisDrone categories → consolidated YOLO classes (see configs/visdrone.yaml).
  4. Convert boxes to YOLO: class x_center y_center width height (normalized 0–1).
  5. Copy images and write .txt labels under dataset/images/* and dataset/labels/*.
  6. Emit dataset/data.yaml for training (Ultralytics YOLO).

See configs/VISDRONE_SETUP.md for download and folder layout.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths (resolve relative to project root = cwd or --output-root)
# ---------------------------------------------------------------------------


def _project_paths(output_root: Path) -> dict[str, Path]:
    root = output_root.resolve()
    return {
        "root": root,
        "dataset": root / "dataset",
        "images_train": root / "dataset" / "images" / "train",
        "images_val": root / "dataset" / "images" / "val",
        "images_test": root / "dataset" / "images" / "test",
        "labels_train": root / "dataset" / "labels" / "train",
        "labels_val": root / "dataset" / "labels" / "val",
        "labels_test": root / "dataset" / "labels" / "test",
        "data_yaml": root / "dataset" / "data.yaml",
        "results": root / "results",
    }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_visdrone_config(config_path: Path) -> tuple[list[str], dict[int, int | None]]:
    """Load class names and VisDrone id → YOLO id map from YAML."""
    with open(config_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    names: list[str] = list(raw["names"])
    raw_map = raw["visdrone_to_yolo"]
    mapping: dict[int, int | None] = {}
    for k, v in raw_map.items():
        key = int(k)
        if v is None:
            mapping[key] = None
        else:
            mapping[key] = int(v)
    return names, mapping


# ---------------------------------------------------------------------------
# VisDrone → YOLO conversion (single image)
# ---------------------------------------------------------------------------


def parse_visdrone_line(parts: list[str]) -> tuple[float, float, float, float, int, int] | None:
    """
    Parse one annotation row into (left, top, w, h, score, category).
    Returns None if the row is malformed.
    """
    if len(parts) < 6:
        return None
    try:
        left, top, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        score = int(float(parts[4]))
        category = int(float(parts[5]))
    except (ValueError, IndexError):
        return None
    return left, top, bw, bh, score, category


def visdrone_to_yolo_lines(
    ann_path: Path,
    img_w: int,
    img_h: int,
    visdrone_to_yolo: dict[int, int | None],
    min_box_side_px: float = 1.0,
) -> list[str]:
    """
    Convert a VisDrone .txt file to YOLO label lines (strings).
    Drops invalid boxes, score==0, and categories not in the map.
    """
    lines_out: list[str] = []
    text = ann_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return lines_out

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        parsed = parse_visdrone_line(parts)
        if parsed is None:
            continue
        left, top, bw, bh, score, category = parsed

        # VisDrone GT: score 0 means ignore this annotation.
        if score == 0:
            continue

        yolo_cls = visdrone_to_yolo.get(category)
        if yolo_cls is None:
            continue

        if bw < min_box_side_px or bh < min_box_side_px:
            continue
        if bw <= 0 or bh <= 0:
            continue

        # Clip box to image bounds, then convert to YOLO normalized xywh.
        x1, y1 = left, top
        x2, y2 = left + bw, top + bh
        x1 = max(0.0, min(x1, img_w - 1.0))
        y1 = max(0.0, min(y1, img_h - 1.0))
        x2 = max(0.0, min(x2, img_w - 1.0))
        y2 = max(0.0, min(y2, img_h - 1.0))

        bw_clip = x2 - x1
        bh_clip = y2 - y1
        if bw_clip < min_box_side_px or bh_clip < min_box_side_px:
            continue

        xc = (x1 + x2) / 2.0 / img_w
        yc = (y1 + y2) / 2.0 / img_h
        wn = bw_clip / img_w
        hn = bh_clip / img_h

        # Final sanity: strictly inside (0,1) with small epsilon for numerical edge cases.
        if wn <= 0 or hn <= 0 or xc <= 0 or yc <= 0 or xc >= 1 or yc >= 1:
            continue

        lines_out.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    return lines_out


# ---------------------------------------------------------------------------
# Process one VisDrone split folder (contains images/ and annotations/)
# ---------------------------------------------------------------------------


def process_split(
    source_dir: Path,
    images_out: Path,
    labels_out: Path,
    visdrone_to_yolo: dict[int, int | None],
    copy_mode: str = "copy",
) -> tuple[int, int]:
    """
    Convert all pairs in source_dir/images + source_dir/annotations.
    Returns (num_images_copied, num_label_files_written).
    """
    img_dir = source_dir / "images"
    ann_dir = source_dir / "annotations"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing images folder: {img_dir}")
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Missing annotations folder: {ann_dir}")

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in exts
    )
    n_img = 0
    n_lbl = 0

    for img_path in image_files:
        stem = img_path.stem
        ann_path = ann_dir / f"{stem}.txt"
        if not ann_path.is_file():
            # Skip images with no annotation file (unusual for VisDrone DET).
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]

        yolo_lines = visdrone_to_yolo_lines(ann_path, w, h, visdrone_to_yolo)
        dst_img = images_out / img_path.name
        if copy_mode == "symlink":
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            try:
                dst_img.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dst_img)
        else:
            shutil.copy2(img_path, dst_img)

        lbl_path = labels_out / f"{stem}.txt"
        lbl_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        n_img += 1
        n_lbl += 1

    return n_img, n_lbl


# ---------------------------------------------------------------------------
# data.yaml for Ultralytics
# ---------------------------------------------------------------------------


def write_data_yaml(paths: dict[str, Path], names: list[str]) -> None:
    """Write Ultralytics-style data.yaml.

    Do not set ``path:``. Ultralytics uses the YAML file's parent folder as the
    dataset root; ``path: .`` would be resolved from the process cwd and breaks
    training when cwd is the project root (it would look for ./images/val).
    """
    content = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    paths["data_yaml"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["data_yaml"], "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Visualization (YOLO labels + image)
# ---------------------------------------------------------------------------


def yolo_line_to_pixel_box(
    line: str, img_w: int, img_h: int
) -> tuple[int, float, float, float, float] | None:
    """Parse one YOLO label line; return (cls, xc, yc, w, h) in pixels (xyxy corners)."""
    parts = line.split()
    if len(parts) != 5:
        return None
    cls = int(float(parts[0]))
    xc, yc, wn, hn = map(float, parts[1:])
    bw = wn * img_w
    bh = hn * img_h
    x1 = int(round((xc * img_w) - bw / 2))
    y1 = int(round((yc * img_h) - bh / 2))
    x2 = int(round((xc * img_w) + bw / 2))
    y2 = int(round((yc * img_h) + bh / 2))
    return cls, float(x1), float(y1), float(x2), float(y2)


def visualize_random_samples(
    paths: dict[str, Path],
    class_names: list[str],
    split: str,
    num_samples: int,
    seed: int,
    out_dir: Path,
) -> None:
    """Save grid of random images with YOLO boxes drawn (BGR via OpenCV → file)."""
    split = split.lower()
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")

    img_dir = paths["dataset"] / "images" / split
    lbl_dir = paths["dataset"] / "labels" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"No images at {img_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    if not imgs:
        raise FileNotFoundError(f"No images found in {img_dir}")

    rng = random.Random(seed)
    picks = rng.sample(imgs, k=min(num_samples, len(imgs)))

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = _distinct_colors(len(class_names))

    for idx, img_path in enumerate(picks):
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.is_file():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parsed = yolo_line_to_pixel_box(line, w, h)
                if parsed is None:
                    continue
                cls_id, x1, y1, x2, y2 = parsed
                x1i = int(max(0, min(x1, w - 1)))
                y1i = int(max(0, min(y1, h - 1)))
                x2i = int(max(0, min(x2, w - 1)))
                y2i = int(max(0, min(y2, h - 1)))
                color = colors[cls_id % len(colors)]
                cv2.rectangle(im, (x1i, y1i), (x2i, y2i), color, 2)
                name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
                cv2.putText(
                    im,
                    name,
                    (x1i, max(0, y1i - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        out_path = out_dir / f"sample_{split}_{idx:03d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), im)

    print(f"Wrote {len(picks)} visualizations to {out_dir}")


def _distinct_colors(n: int) -> list[tuple[int, int, int]]:
    """Generate n BGR colors for drawing."""
    if n <= 0:
        return []
    base = [
        (255, 99, 71),
        (60, 179, 113),
        (30, 144, 255),
        (255, 191, 0),
        (147, 112, 219),
        (0, 165, 255),
        (203, 192, 255),
    ]
    out: list[tuple[int, int, int]] = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="VisDrone DET → YOLO dataset preparation")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd(),
        help="Project root (contains dataset/, configs/). Default: current working directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to visdrone.yaml (default: <output-root>/configs/visdrone.yaml)",
    )
    parser.add_argument(
        "--train-source",
        type=Path,
        default=None,
        help="VisDrone train folder with images/ and annotations/ subfolders.",
    )
    parser.add_argument(
        "--val-source",
        type=Path,
        default=None,
        help="VisDrone val folder with images/ and annotations/ subfolders.",
    )
    parser.add_argument(
        "--test-source",
        type=Path,
        default=None,
        help="Optional test split (images only is OK; labels skipped if missing .txt).",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="How to place images into dataset/images (default copy).",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only run visualization on existing dataset/ (no conversion).",
    )
    parser.add_argument("--split", type=str, default="train", help="Split for --visualize-only.")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    output_root = args.output_root.resolve()
    config_path = args.config or (output_root / "configs" / "visdrone.yaml")
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    names, vis_map = load_visdrone_config(config_path)
    paths = _project_paths(output_root)

    if args.visualize_only:
        viz_dir = paths["results"] / "dataset_preview"
        visualize_random_samples(
            paths, names, args.split, args.num_samples, args.seed, viz_dir
        )
        return 0

    if args.train_source is None and args.val_source is None:
        print("Provide --train-source and/or --val-source, or use --visualize-only.", file=sys.stderr)
        return 1

    total_img = 0
    total_lbl = 0
    if args.train_source is not None:
        n_i, n_l = process_split(
            args.train_source.resolve(),
            paths["images_train"],
            paths["labels_train"],
            vis_map,
            copy_mode=args.copy_mode,
        )
        print(f"Train: {n_i} images, {n_l} label files → {paths['images_train']}")
        total_img += n_i
        total_lbl += n_l

    if args.val_source is not None:
        n_i, n_l = process_split(
            args.val_source.resolve(),
            paths["images_val"],
            paths["labels_val"],
            vis_map,
            copy_mode=args.copy_mode,
        )
        print(f"Val:   {n_i} images, {n_l} label files → {paths['images_val']}")
        total_img += n_i
        total_lbl += n_l

    if args.test_source is not None:
        # Test may have no public labels; copy images and write labels when .txt exists.
        src = args.test_source.resolve()
        img_dir = src / "images"
        ann_dir = src / "annotations"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing images folder: {img_dir}")
        paths["images_test"].mkdir(parents=True, exist_ok=True)
        paths["labels_test"].mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        n_test = 0
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts):
            dst = paths["images_test"] / img_path.name
            if args.copy_mode == "symlink":
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                try:
                    dst.symlink_to(img_path.resolve())
                except OSError:
                    shutil.copy2(img_path, dst)
            else:
                shutil.copy2(img_path, dst)
            stem = img_path.stem
            ann_path = ann_dir / f"{stem}.txt"
            if ann_path.is_file():
                im = cv2.imread(str(img_path))
                if im is None:
                    continue
                h, w = im.shape[:2]
                lines = visdrone_to_yolo_lines(ann_path, w, h, vis_map)
                (paths["labels_test"] / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
            else:
                (paths["labels_test"] / f"{stem}.txt").write_text("", encoding="utf-8")
            n_test += 1
        print(f"Test: {n_test} images → {paths['images_test']}")

    write_data_yaml(paths, names)
    print(f"Wrote {paths['data_yaml']} (nc={len(names)})")
    print("Done. Optional: run with --visualize-only to preview boxes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
