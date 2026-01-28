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

from src.model.YOLOv1 import YOLOv1


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
    boxes: List of [class_pred, prob_score, x1, y1, x2, y2]
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert tensor to numpy for plotting
    image = image.permute(1, 2, 0).cpu().numpy()
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box in boxes:
        class_idx = int(box[0])
        prob = box[1]
        # Coordinates (assuming midpoint format normalized 0-1)
        # You may need to rescale these by image.shape[0] and image.shape[1]
        x, y, w, h = box[2], box[3], box[4], box[5]
        
        # Calculate top-left for matplotlib
        upper_left_x = (x - w / 2) * image.shape[1]
        upper_left_y = (y - h / 2) * image.shape[0]
        rect = patches.Rectangle(
            (upper_left_x, upper_left_y),
            w * image.shape[1],
            h * image.shape[0],
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.text(
            upper_left_x,
            upper_left_y,
            s=f"{class_labels[class_idx]} {prob:.2f}",
            color="white",
            verticalalignment="top",
            bbox={"color": "red", "pad": 0},
        )

    plt.axis("off")
    plt.savefig(os.path.join(output_folder, f"{img_name}.png"), bbox_inches='tight')
    plt.close()
    
def get_bboxes(loader, model, iou_threshold, threshold, device, S=7, B=2, C=20):
    model.eval()
    all_pred_boxes = []
    all_true_boxes = []
    train_idx = 0

    print(">>> Predicting bboxes...")
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        # Convert raw tensor to a list of bounding boxes
        # Format: [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        bboxes = cellboxes_to_boxes(predictions, S=S, B=B, C=C)
        true_bboxes = cellboxes_to_boxes(labels, S=S, B=B, C=C)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format="midpoint",
            )

            for box in nms_boxes:
                all_pred_boxes.append([train_idx] + box)

            for box in true_bboxes[idx]:
                # only keep actual objects (prob > threshold)
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1
            
    return all_pred_boxes, all_true_boxes

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Load your DETECTION model (not the Pretrain one)
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    
    # Load the Post-train checkpoint
    checkpoint = torch.load("checkpoints/checkpoint_best.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Get boxes
    pred_boxes, true_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE)

    # Calculate mAP
    map_score = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint")
    print(f"Mean Average Precision: {map_score}")