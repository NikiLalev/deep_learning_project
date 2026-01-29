import torch
import torch.nn as nn

def xywh_to_xyxy(xywh):
    x, y, w, h = xywh.unbind(-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)

def iou_xyxy(box1, box2, eps=1e-6):
    """Calculates IoU for boxes in [x1,y1,x2,y2] format."""

    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * (box1[..., 3] - box1[..., 1]).clamp(min=0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * (box2[..., 3] - box2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter + eps
    return inter / union

class YOLOv1Loss(nn.Module):
    """
    Paper-closest YOLOv1 loss for preds/targets shaped [B,S,S,C + B*5]
    Layout per cell:
      class scores: 0:C
      box k: conf at C + k*5, coords at C + k*5 + 1..4 (x_cell,y_cell,w_img,h_img)
    """
    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.lc = lambda_coord
        self.lno = lambda_noobj

    def forward(self, preds, targets):
        Bsz, S, _, D = preds.shape
        C = self.C
        device = preds.device

        # object indicator per cell (paper: 1^obj_i). We use target conf1 slot as your encoding.
        obj = (targets[..., C] > 0).float()                 # [B,S,S]
        noobj = 1.0 - obj                                   # [B,S,S]

        # -------- Class loss (paper eq: sum_i 1^obj_i sum_c (p_i(c)-p̂_i(c))^2 ) --------
        pred_class = preds[..., :C]                          # raw scores (no softmax)
        class_loss = ((pred_class - targets[..., :C]) ** 2 * obj.unsqueeze(-1)).sum()

        # -------- Decode predicted boxes (cell-relative x,y; image-relative w,h) --------
        # coords are at [conf, x, y, w, h]
        # box1
        pred_conf1 = preds[..., C + 0]                       # raw conf
        pred_xy1   = preds[..., C + 1:C + 3]                 # raw x,y (cell)
        pred_wh1   = preds[..., C + 3:C + 5]                 # raw w,h (img)

        # box2
        pred_conf2 = preds[..., C + 5]
        pred_xy2   = preds[..., C + 6:C + 8]
        pred_wh2   = preds[..., C + 8:C + 10]

        # Paper doesn't mandate sigmoids; but we must keep x,y in [0,1) per cell.
        # This is common & keeps training stable.
        pred_xy1 = torch.sigmoid(pred_xy1)
        pred_xy2 = torch.sigmoid(pred_xy2)

        # w,h should be >=0 in paper loss (sqrt). Use clamp to avoid abs kink.
        pred_wh1 = pred_wh1.clamp(min=0.0)
        pred_wh2 = pred_wh2.clamp(min=0.0)

        pred_box1 = torch.cat([pred_xy1, pred_wh1], dim=-1)  # [B,S,S,4]
        pred_box2 = torch.cat([pred_xy2, pred_wh2], dim=-1)

        # Targets (same for both predictors)
        tgt_box = targets[..., C + 1:C + 5]                  # [B,S,S,4] (x_cell,y_cell,w_img,h_img)

        # -------- IoU to decide responsibility (paper: choose best predictor) --------
        grid_i = torch.arange(S, device=device).view(1, S, 1, 1).expand(Bsz, S, S, 1)
        grid_j = torch.arange(S, device=device).view(1, 1, S, 1).expand(Bsz, S, S, 1)

        def cell_to_img_xywh(box_xywh):
            x_cell = box_xywh[..., 0:1]
            y_cell = box_xywh[..., 1:2]
            w_img  = box_xywh[..., 2:3]
            h_img  = box_xywh[..., 3:4]
            x_img = (grid_j + x_cell) / S
            y_img = (grid_i + y_cell) / S
            return torch.cat([x_img, y_img, w_img, h_img], dim=-1)

        pred1_img = cell_to_img_xywh(pred_box1)
        pred2_img = cell_to_img_xywh(pred_box2)
        tgt_img   = cell_to_img_xywh(tgt_box)

        iou1 = iou_xyxy(xywh_to_xyxy(pred1_img), xywh_to_xyxy(tgt_img))  # [B,S,S]
        iou2 = iou_xyxy(xywh_to_xyxy(pred2_img), xywh_to_xyxy(tgt_img))  # [B,S,S]

        resp1 = (iou1 >= iou2).float() * obj                  # [B,S,S]
        resp2 = (iou2 >  iou1).float() * obj

        # -------- Localization loss (paper: x,y and sqrt(w),sqrt(h) for responsible) --------
        def loc_loss(pred_box, resp):
            # x,y
            xy = ((pred_box[..., 0:2] - tgt_box[..., 0:2]) ** 2 * resp.unsqueeze(-1)).sum()
            # sqrt(w), sqrt(h)
            pred_sqrt = torch.sqrt(pred_box[..., 2:4] + 1e-6)
            tgt_sqrt  = torch.sqrt(tgt_box[..., 2:4].clamp(min=0.0) + 1e-6)
            wh = ((pred_sqrt - tgt_sqrt) ** 2 * resp.unsqueeze(-1)).sum()
            return xy + wh

        loc = self.lc * (loc_loss(pred_box1, resp1) + loc_loss(pred_box2, resp2))

        # -------- Confidence loss (paper: (C - Ĉ)^2; target C is IoU for responsible) --------
        # Paper uses MSE; we keep conf raw (no sigmoid) for paper-closest behavior.
        conf_loss_obj = (
            ((pred_conf1 - iou1) ** 2 * resp1).sum()
            + ((pred_conf2 - iou2) ** 2 * resp2).sum()
        )

        # no-object confidence loss for BOTH predictors in empty cells
        conf_loss_noobj = self.lno * (
            ((pred_conf1 - 0.0) ** 2 * noobj).sum()
            + ((pred_conf2 - 0.0) ** 2 * noobj).sum()
        )

        total = class_loss + loc + conf_loss_obj + conf_loss_noobj
        return total / Bsz
