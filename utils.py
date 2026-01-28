import torch

def cellboxes_to_boxes(out, S=7):
    """
    Converts YOLO output to bounding boxes relative to the entire image.
    Assumes 'out' is (batch, S, S, 30) for YOLOv1.
    """
    converted_bboxes = out.reshape(out.shape[0], S, S, -1)
    confidences1 = converted_bboxes[..., 0:1]
    confidences2 = converted_bboxes[..., 5:6]
    
    # Take the box with the higher confidence
    best_confidence, best_box = torch.max(torch.cat([confidences1, confidences2], dim=-1), dim=-1, keepdim=True)
    
    # Extract x, y, w, h based on which box was "best"
    # This is a simplified version; in full YOLO you might process both boxes for NMS
    bboxes1 = converted_bboxes[..., 1:5]
    bboxes2 = converted_bboxes[..., 6:10]
    
    # Logic to select the best box's coordinates
    best_bboxes = torch.where(best_box.unsqueeze(-1) == 0, bboxes1, bboxes2)
    
    # Add cell indices to coordinates to make them global
    cell_indices = torch.arange(S).repeat(out.shape[0], S, 1).unsqueeze(-1).to(out.device)
    
    x = (best_bboxes[..., 0:1] + cell_indices) / S
    y = (best_bboxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3)) / S
    w_y = best_bboxes[..., 2:4] / S  # widths/heights are already relative to image usually
    
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    
    # Get predicted class
    predicted_class = converted_bboxes[..., 10:].argmax(-1, keepdim=True)
    
    return torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    """
    bboxes: list of lists [[class, confidence, x, y, w, h], ...]
    """
    # Filter by confidence threshold first
    bboxes = [box for box in bboxes if box[1] > threshold]
    # Sort by confidence descending
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        # Remove all other boxes of the same class that overlap too much
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    # Calculate intersection area
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)