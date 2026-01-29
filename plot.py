"""
Plot YOLO training history from a JSON file.

Expected JSON keys:
- "train_loss": list[float]
- "val_loss":   list[float]
Optional:
- "lr":         list[float]  (ignored here)

Outputs (PDF):
- plots/loss_train_vs_val.pdf
- plots/train_loss.pdf
- plots/val_loss.pdf
- plots/calibration_<title>.pdf
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from tqdm import tqdm

from test_script import save_predictions
from train_habrok import Config, get_data_loaders
from utils import intersection_over_union
from collections import Counter


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


# -----------------------------
# Training-history plotting
# -----------------------------
def load_history(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path.resolve()}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "train_loss" not in data or "val_loss" not in data:
        raise KeyError('JSON must contain keys "train_loss" and "val_loss".')

    train_loss = data["train_loss"]
    val_loss = data["val_loss"]

    if not isinstance(train_loss, list) or not isinstance(val_loss, list):
        raise TypeError('"train_loss" and "val_loss" must be lists.')

    if not train_loss or not val_loss:
        raise ValueError('"train_loss" and "val_loss" must be non-empty lists.')

    return data


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_plot(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_train_vs_val(train_loss: list[float], val_loss: list[float], out_dir: Path) -> None:
    epochs_train = range(1, len(train_loss) + 1)
    epochs_val = range(1, len(val_loss) + 1)

    fig = plt.figure()
    plt.plot(list(epochs_train), train_loss, label="Train loss")
    plt.plot(list(epochs_val), val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_plot(fig, out_dir / "loss_train_vs_val.pdf")


def plot_single(
    series: list[float],
    title: str,
    ylabel: str,
    filename: str,
    out_dir: Path,
) -> None:
    epochs = range(1, len(series) + 1)

    fig = plt.figure()
    plt.plot(list(epochs), series)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    save_plot(fig, out_dir / filename)


# -----------------------------
# Calibration curve
# -----------------------------
def _load_results_json(json_path: str | os.PathLike) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_calibration_curve(
    json_path: str | os.PathLike,
    title: str = "YOLOv1 Calibration",
    iou_threshold: float = 0.5,
) -> None:
    data = _load_results_json(json_path)

    y_true: list[int] = []
    confidence_scores: list[float] = []

    for sample in data:
        preds = sample.get("predictions", [])
        gts = sample.get("ground_truths", [])

        # Sort by confidence so high-conf preds "claim" GTs first.
        preds.sort(key=lambda x: x["conf"], reverse=True)
        used_gts = [False] * len(gts)

        for pred in preds:
            p_box = torch.tensor(pred["bbox"])
            p_class = pred["class"]
            p_conf = pred["conf"]

            best_iou = 0.0
            best_gt_idx = -1

            for i, gt in enumerate(gts):
                if used_gts[i] or gt["class"] != p_class:
                    continue

                iou = intersection_over_union(
                    p_box.unsqueeze(0),
                    torch.tensor(gt["bbox"]).unsqueeze(0),
                    box_format="midpoint",
                ).item()

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_threshold and best_gt_idx != -1:
                y_true.append(1)  # TP
                used_gts[best_gt_idx] = True
            else:
                y_true.append(0)  # FP

            confidence_scores.append(p_conf)

    confidence_scores = np.clip(confidence_scores, 0, 1)

    prob_true, prob_pred = calibration_curve(y_true, confidence_scores, n_bins=10)

    # Bin counts for error bars
    bins = np.linspace(0, 1, 11)
    conf_array = np.array(confidence_scores)
    n_samples: list[int] = []
    for i in range(10):
        idx = np.where((conf_array >= bins[i]) & (conf_array < bins[i + 1]))[0]
        n_samples.append(len(idx))

    # Binomial standard error
    errors = [
        np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        for p, n in zip(prob_true, n_samples)
    ]

    # Percent conversion
    prob_true_pct = np.array(prob_true) * 100
    prob_pred_pct = np.array(prob_pred) * 100
    errors_pct = np.array(errors) * 100

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        prob_pred_pct,
        prob_true_pct,
        yerr=errors_pct,
        marker="o",
        linestyle="-",
        color="b",
        capsize=5,
        label="YOLOv1",
    )
    plt.plot([0, 100], [0, 100], color="gray", linestyle="--", label="Perfect Calibration")

    plt.xlabel("Mean Confidence (%)")
    plt.ylabel("Accuracy / Precision (%)")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    ensure_dir(Path("plots"))
    out_name = f"calibration_{title.replace(' ', '_')}.pdf"
    plt.savefig(Path("plots") / out_name, format="pdf", bbox_inches="tight")
    plt.show()

    print(f"Processed {len(y_true)} detections.")
    ece = np.sum(
        np.abs(np.array(prob_true) - np.array(prob_pred)) * (np.array(n_samples) / len(y_true))
    )
    print(f"ECE: {ece:.4f}")


# -----------------------------
# Visualization saving
# -----------------------------
def save_images_from_json(
    json_path: str | os.PathLike,
    loader,
    output_folder: str = "visual_results",
    VOC_CLASSES: list[str] | None = None,
    num_predictions: int = 100,
    threshold: float = 0.4,
) -> None:
    """
    json_path: Path to your results.json
    loader: The test_loader (instance of torch.utils.data.DataLoader)
    output_folder: Directory where images will be saved
    VOC_CLASSES: List of class names for labeling
    """
    ensure_dir(Path(output_folder))

    dataset = loader.dataset
    results_data = _load_results_json(json_path)
    print(f"Processing {len(results_data)} entries from JSON...")

    for entry in tqdm(results_data):
        img_id = entry["img_id"]
        if img_id >= num_predictions:
            break

        preds = entry["predictions"]

        try:
            sample = dataset[img_id]
            image_tensor = sample["image"]
        except Exception as e:
            print(f"Could not load image index {img_id}: {e}")
            continue

        boxes_to_draw: list[list[float]] = []
        for p in preds:
            if p["conf"] > threshold:
                boxes_to_draw.append([p["class"], p["conf"]] + p["bbox"])

        save_predictions(
            image=image_tensor,
            boxes=boxes_to_draw,
            class_labels=VOC_CLASSES,
            output_folder=output_folder,
            img_name=f"pred_image_{img_id}",
        )

    print(f"\nFinished! Images saved to {output_folder}")

# COCO size thresholds (pixels^2)
SMALL_AREA = 32 * 32
MEDIUM_AREA = 96 * 96

def box_area_pixels(box, img_w, img_h):
    # box: [xc, yc, w, h] normalized
    return (box[2] * img_w) * (box[3] * img_h)

def categorize_gt(box, img_w, img_h):
    area = box_area_pixels(box, img_w, img_h)
    if area < SMALL_AREA:
        return "small"
    elif area < MEDIUM_AREA:
        return "medium"
    else:
        return "large"

def compute_map_for_subset(pred_boxes, gt_boxes, iou_threshold=0.5, num_classes=20):
    epsilon = 1e-6
    APs = []

    for c in range(num_classes):
        detections = [b for b in pred_boxes if b[1] == c]
        gts = [b for b in gt_boxes if b[1] == c]

        if len(gts) == 0:
            continue

        detections.sort(key=lambda x: x[2], reverse=True)

        gt_counter = Counter([gt[0] for gt in gts])
        gt_used = {k: torch.zeros(v) for k, v in gt_counter.items()}

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        for i, det in enumerate(detections):
            img_gts = [gt for gt in gts if gt[0] == det[0]]

            best_iou = 0
            best_gt_idx = -1

            for idx, gt in enumerate(img_gts):
                iou = intersection_over_union(
                    torch.tensor(det[3:]),
                    torch.tensor(gt[3:]),
                    box_format="midpoint"
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= iou_threshold:
                img_id = det[0]
                if gt_used[img_id][best_gt_idx] == 0:
                    TP[i] = 1
                    gt_used[img_id][best_gt_idx] = 1
                else:
                    FP[i] = 1
            else:
                FP[i] = 1

        TP_cum = torch.cumsum(TP, dim=0)
        FP_cum = torch.cumsum(FP, dim=0)

        recalls = TP_cum / (len(gts) + epsilon)
        precisions = TP_cum / (TP_cum + FP_cum + epsilon)

        recalls = torch.cat((torch.tensor([0.0]), recalls))
        precisions = torch.cat((torch.tensor([1.0]), precisions))

        APs.append(torch.trapz(precisions, recalls))

    return sum(APs) / len(APs) if APs else 0.0

def load_and_split_by_size(json_path, img_w, img_h):
    with open(json_path, "r") as f:
        data = json.load(f)

    pred_all = []
    gt_small, gt_medium, gt_large = [], [], []

    for entry in data:
        img_id = entry["img_id"]

        for p in entry["predictions"]:
            pred_all.append([
                img_id,
                p["class"],
                p["conf"],
                *p["bbox"]
            ])

        for gt in entry["ground_truths"]:
            size = categorize_gt(gt["bbox"], img_w, img_h)
            record = [img_id, gt["class"], 1.0, *gt["bbox"]]

            if size == "small":
                gt_small.append(record)
            elif size == "medium":
                gt_medium.append(record)
            else:
                gt_large.append(record)

    return pred_all, gt_small, gt_medium, gt_large


def calculate_map(json_path: str | os.PathLike, iou_threshold: float = 0.5, num_classes: int = 20) -> float:
    '''    Calculate mAP for small, medium, and large objects separately.'''

    preds, gt_s, gt_m, gt_l = load_and_split_by_size(
        "results.json", 448, 448
    )
    gt_all = gt_s + gt_m + gt_l
    print("mAP (small): ", compute_map_for_subset(preds, gt_s).item())
    print("mAP (medium):", compute_map_for_subset(preds, gt_m).item())
    print("mAP (large): ", compute_map_for_subset(preds, gt_l).item())
    print("mAP (all):    ", compute_map_for_subset(preds, gt_all).item())



def main() -> None:
    json_path = Path("training_history_20260128_223854.json")
    out_dir = Path("plots")

    history = load_history(json_path)
    ensure_dir(out_dir)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    plot_train_vs_val(train_loss, val_loss, out_dir)
    plot_single(train_loss, "Training Loss", "Loss", "train_loss.pdf", out_dir)
    plot_single(val_loss, "Validation Loss", "Loss", "val_loss.pdf", out_dir)

    print(f"Saved plots to: {out_dir.resolve()}")

    plot_calibration_curve("results.json", iou_threshold=0.5)

    _, _, dataset = get_data_loaders(Config())
    save_images_from_json(
        "results.json",
        dataset,
        VOC_CLASSES=VOC_CLASSES,
        num_predictions=445,
        threshold=0.05,
    )

    calculate_map("results.json", iou_threshold=0.5, num_classes=len(VOC_CLASSES))


if __name__ == "__main__":
    main()
