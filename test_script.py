import json
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
# You'll need an IoU and NMS function - I'll assume they are in your utils
from utils import intersection_over_union, non_max_suppression, cellboxes_to_boxes

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Use 'Agg' backend to save images without a GUI/display
import matplotlib
matplotlib.use('Agg')
import os

from src.model.detector import YOLOv1
from train_habrok import build_targets_yolov1, get_data_loaders, Config  # Assuming test_loader is defined in train_habrok.py


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]



def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20):
    # pred_boxes format: [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = [box for box in pred_boxes if box[1] == c]
        ground_truths = [box for box in true_boxes if box[1] == c]

        # Count how many GT boxes exist for each image
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # Sort by confidence
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only compare with GT boxes from the same image
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            best_iou = 0
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format="midpoint"
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cum = torch.cumsum(TP, dim=0)
        FP_cum = torch.cumsum(FP, dim=0)
        recalls = TP_cum / (total_true_bboxes + epsilon)
        precisions = TP_cum / (TP_cum + FP_cum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def save_predictions(image, boxes, class_labels, output_folder, img_name):
    """
    image: Tensor of shape (3, H, W)
    boxes: List of Tensors or Lists [class_pred, prob_score, xc, yc, w, h]
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Convert tensor to numpy and fix format
    # permute: (C, H, W) -> (H, W, C)
    image = image.permute(1, 2, 0).cpu().numpy()
    
    # IMPORTANT: If your data was normalized (0-1), ensure it stays in range.
    # If you used ImageNet normalization, you'd need to multiply by std and add mean here.
    image = np.clip(image, 0, 1) 

    height, width, _ = image.shape
    
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    for box in boxes:
        # Convert to list if it's a tensor
        if torch.is_tensor(box):
            box = box.tolist()
            
        class_idx = int(box[0])
        prob = box[1]
        xc, yc, w, h = box[2], box[3], box[4], box[5]

        # 2. Convert Midpoint (normalized 0-1) to Corner (pixel coordinates)
        # matplotlib.patches.Rectangle needs the bottom-left corner (x, y)
        pixel_w = w * width
        pixel_h = h * height
        upper_left_x = (xc * width) - (pixel_w / 2)
        upper_left_y = (yc * height) - (pixel_h / 2)

        # 3. Create the rectangle
        rect = patches.Rectangle(
            (upper_left_x, upper_left_y),
            pixel_w,
            pixel_h,
            linewidth=2,
            edgecolor="lime", # Using lime for better visibility
            facecolor="none",
        )
        ax.add_patch(rect)

        # 4. Add the label text
        label_text = f"{class_labels[class_idx]} {prob:.2f}"
        ax.text(
            upper_left_x,
            upper_left_y - 5, # Position text slightly above the box
            s=label_text,
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox={"facecolor": "lime", "alpha": 0.5, "pad": 1},
        )

    plt.axis("off")
    save_path = os.path.join(output_folder, f"{img_name}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig) # Explicitly close figure to free memory on HPC

def get_bboxes(loader, model, iou_threshold, threshold, device, S=7, B=2, C=20):
    all_json_data = [] # New list to store data for JSON
    all_pred_boxes = []
    all_true_boxes = []
    model.eval()
    train_idx = 0

    for batch in tqdm(loader):
        # 1. Correct the 'images' key from your previous error
        x = batch["images"].to(device)
        boxes = [b.to(device) for b in batch["boxes"]]
        labels = [l.to(device) for l in batch["labels"]]

        # 2. Build the target grid
        # This converts list of tensors -> [Batch, 7, 7, 30]
        targets = build_targets_yolov1(boxes, labels, S=S, B=B, C=C, device=device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        
        # Now both predictions and targets are in the same [B, 7, 7, 30] format
        true_bboxes = cellboxes_to_boxes(targets, S=S, B=B, C=C)
        bboxes = cellboxes_to_boxes(predictions, S=S, B=B, C=C)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format="midpoint",
            )
            image_entry = {
                "img_id": train_idx,
                "predictions": [],
                "ground_truths": []
            }
            if train_idx < 100:  # Save predictions for first 100 images only to limit output
                save_predictions(
                    image=x[idx], 
                    boxes=nms_boxes, 
                    class_labels=VOC_CLASSES, 
                    output_folder="images", 
                    img_name=f"pred_{train_idx}"
                )

            for box in nms_boxes:
                image_entry["predictions"].append({
                    "class": int(box[0]),
                    "conf": round(float(box[1]), 4),
                    "bbox": [round(float(val), 4) for val in box[2:]]
                })
                if box[1] > threshold:
                    all_pred_boxes.append([train_idx] + box.tolist())

            for box in true_bboxes[idx]:
                # many boxes will be empty (prob=0), only keep real ones
                if box[1] > threshold:
                    image_entry["ground_truths"].append({
                        "class": int(box[0]),
                        "bbox": [round(float(val), 4) for val in box[2:]]
                    })
                    all_true_boxes.append([train_idx] + box.tolist())
            
            all_json_data.append(image_entry)
            train_idx += 1

    # Save to file
    with open("results.json", "w") as f:
        json.dump(all_json_data, f, indent=4)

    print(f"\nSaved results for {train_idx} images to results.json")

    return all_pred_boxes, all_true_boxes


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Load your DETECTION model (not the Pretrain one)
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    
    # Load the Post-train checkpoint
    checkpoint = torch.load("checkpoints/checkpoint_best_prev.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    train_loader, val_loader, test_loader = get_data_loaders(Config())  # Assuming this function returns test_loader too
    # Get boxes
    pred_boxes, true_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.05, device=DEVICE)

    # Calculate mAP
    map_score = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20)
    print(f"Mean Average Precision: {map_score}")

if __name__ == "__main__":
    main()