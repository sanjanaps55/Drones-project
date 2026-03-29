# VisDrone DET: download and layout

## What you need

Use the **VisDrone2019-DET** image detection release (not video/MOT unless you adapt the parser).

Official repository: [https://github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)

## Download

1. Open the VisDrone dataset page (GitHub README links to **Baidu Netdisk** and **Google Drive** mirrors).
2. Download at minimum:
   - **VisDrone2019-DET-train** (training images + annotations)
   - **VisDrone2019-DET-val** (validation)
3. Optional: **VisDrone2019-DET-test-dev** (images only; labels may be restricted—use for inference-only checks).

Extract archives so each split looks like:

```text
VisDrone2019-DET-train/
  images/          # *.jpg
  annotations/     # *.txt (same stem as image)
```

```text
VisDrone2019-DET-val/
  images/
  annotations/
```

## Annotation format (DET)

Each line in an annotation file:

`bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion`

- Box is **top-left** `(x, y)` plus **width/height** in pixels.
- `score`: in ground truth, `0` means **ignore this box**; `1` means use it.
- `category`: `0` = ignored region, `11` = others (skipped in our pipeline per config).

## Prepare YOLO dataset

From the project root (folder that contains `scripts/` and `dataset/`):

```powershell
python scripts/prepare_dataset.py --train-source "PATH\VisDrone2019-DET-train" --val-source "PATH\VisDrone2019-DET-val" --output-root .
```

This fills `dataset/images/{train,val}` and `dataset/labels/{train,val}` and writes `dataset/data.yaml` for Ultralytics YOLO.

## Visualize samples

```powershell
python scripts/prepare_dataset.py --visualize-only --split train --num-samples 12 --seed 42 --output-root .
```

Requires images and YOLO labels already under `dataset/`.
