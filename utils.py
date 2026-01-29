import torch

def cellboxes_to_boxes(out, S=7, B=2, C=20):
    """
    Converts YOLO output [batch, S, S, 30] to [batch, S*S, 6]
    Output format: [class_idx, confidence, x, y, w, h]
    """
    batch_size = out.shape[0]
    device = out.device

    # 1. Extract classes and confidences
    # out shape: [batch, 7, 7, 30]
    classes = out[..., :C].argmax(-1).unsqueeze(-1)  # [batch, 7, 7, 1]
    
    conf1 = out[..., C:C+1]    # [batch, 7, 7, 1]
    conf2 = out[..., C+5:C+6]  # [batch, 7, 7, 1]
    
    # 2. Select the best box (B1 or B2)
    # best_box_idx will be 0 if box1 is better, 1 if box2 is better
    _, best_box_idx = torch.max(torch.cat([conf1, conf2], dim=-1), dim=-1, keepdim=True)
    best_conf = torch.max(conf1, conf2) # Higher confidence value
    
    # 3. Extract x, y, w, h for both boxes
    box1 = out[..., C+1:C+5]
    box2 = out[..., C+6:C+10]
    
    # Selection logic that avoids dimension errors
    best_boxes = torch.where(best_box_idx == 0, box1, box2)

    # 4. Global Coordinate Transformation
    # We need to add the cell offset (0..6) to the cell-relative (x, y)
    
    # Create a 1D range [0, 1, 2, 3, 4, 5, 6]
    cell_range = torch.arange(S).to(device)
    
    # Create the 2D grid offsets
    # x_off: [[0,1,2,3,4,5,6], [0,1,2,3,4,5,6], ...] -> columns
    # y_off: [[0,0,0...], [1,1,1...], ...] -> rows
    x_off = cell_range.repeat(S, 1).unsqueeze(0).unsqueeze(-1)      # [1, 7, 7, 1]
    y_off = cell_range.repeat(S, 1).t().unsqueeze(0).unsqueeze(-1)  # [1, 7, 7, 1]

    # Calculate global x, y
    # (local_x + cell_column_index) / 7
    x = (best_boxes[..., 0:1] + x_off) / S
    y = (best_boxes[..., 1:2] + y_off) / S
    
    # w, h are already image-relative (0..1) in YOLOv1
    w = best_boxes[..., 2:3]
    h = best_boxes[..., 3:4]

    # 5. Final Assembly
    # Concatenate: [class, conf, x, y, w, h]
    # Final shape: [batch, 7, 7, 6]
    converted_bboxes = torch.cat((classes.float(), best_conf, x, y, w, h), dim=-1)
    
    # Flatten grid dimensions: [batch, 49, 6]
    return converted_bboxes.reshape(batch_size, S * S, 6)

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