# UAV Garbage Detection with Explainable AI

A two-stage pipeline for detecting and classifying litter in UAV (drone) imagery, combining **YOLOv8** for detection, **EfficientNet-B0** for fine-grained classification, and **Grad-CAM** for explainability — so predictions aren't just accurate, they're interpretable.

## Pipeline

1. **Detect** — a YOLOv8 model (fine-tuned on garbage/litter classes) scans a UAV image and draws bounding boxes around candidate litter objects.
2. **Crop & Classify** — each detected region is cropped and passed through an EfficientNet-B0 classifier (fine-tuned on TACO-style categories) to assign a fine-grained litter category.
3. **Explain** — Grad-CAM is applied to the EfficientNet-B0 classifier to generate a heatmap over each crop, showing which pixels most influenced the predicted class.

```
UAV Image → YOLOv8 (detect) → crop each box → EfficientNet-B0 (classify) → Grad-CAM (explain)
```

## Classes

The classifier recognizes 15 litter categories (TACO-style taxonomy):

`Bottle, Bottle_cap, Broken_glass, Can, Carton, Cigarette, Cup, Lid, Other_plastic, Paper, Plastic_bag_and_wrapper, Pop_tab, Straw, Styrofoam_piece, Unlabeled_litter`

## Data

Inference is demonstrated on the **UAVVaste** dataset (drone-captured litter images), unzipped from `data/raw/uavvaste.zip` in the original workflow.

## Models

This repo contains the **inference notebook**; it expects two pretrained model checkpoints (not included here due to size):

- `models/yolov8_garbage/weights/best.pt` — YOLOv8 detector fine-tuned on garbage/litter
- `models/efficientnet_taco_finetuned.pth` — EfficientNet-B0 classifier fine-tuned on TACO-style classes

Update the checkpoint paths in the notebook to point to wherever you store your trained weights (e.g. Google Drive, local disk, or a released model artifact).

## Setup

```bash
pip install -r requirements.txt
```

Then open `notebooks/UAV_Garbage_Inference.ipynb` and run the cells in order:

1. Load the YOLOv8 and EfficientNet-B0 checkpoints
2. Run YOLO detection on a UAV image
3. Crop each detected object and classify it with EfficientNet-B0
4. Generate a Grad-CAM overlay for the classifier's decision on a chosen crop

## Tech Stack

- **Detection:** YOLOv8 (Ultralytics)
- **Classification:** EfficientNet-B0 (torchvision)
- **Explainability:** Grad-CAM (`pytorch-grad-cam`)
- **Core:** PyTorch, OpenCV, Pillow, Matplotlib

## Why Explainability Matters Here

Litter classification from aerial imagery can be ambiguous — small objects, partial occlusion, and lighting variation all affect predictions. Grad-CAM overlays make it possible to visually verify *why* the model classified an object a certain way (e.g. confirming it focused on the object itself rather than background clutter), which matters for trust in any downstream automated waste-monitoring system.

---
Built as an applied AI/ML project exploring UAV-based environmental monitoring and explainable computer vision.
