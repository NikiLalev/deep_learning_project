import torch
import torch.nn as nn
from src.model.detector import YOLOv1

# Inspired by the YOLO 9000 paper: https://arxiv.org/abs/1612.08242
class YOLOImproved(YOLOv1):
    def __init__(self, split_size=13, num_classes=20):
        # 1. Initialize YOLOv1 backbone (first 20 layers)
        # 5 anchors based on the YOLO9000 paper "We choose k = 5"
        # Split size 13 for 416x416 input images since we downsample by 32x (416/32=13)
        super(YOLOImproved, self).__init__(split_size, num_boxes=5, num_classes=num_classes)
        
        # 2. Remove parts of YOLOv1 we don't need
        if hasattr(self, 'fcl'): del self.fcl
        if hasattr(self, 'detection_layers'): del self.detection_layers

        # 3. Define Anchors (From YOLOv2 Config, k=5) - https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg
        self.anchors = [
            (0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), 
            (7.88282, 3.52778), (9.77052, 9.16828)
        ]
        self.num_anchors = len(self.anchors)
        
        # 4. New Detection Head (Fully Convolutional)
        def _conv_block(in_c, out_c, size):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, size, stride=1, padding=size//2, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1, inplace=True),
            )

        # "Three 3 x 3 convolutional layers with 1024 filters"
        self.conv_head = nn.Sequential(
            _conv_block(1024, 1024, size=3),
            _conv_block(1024, 1024, size=3),
            _conv_block(1024, 1024, size=3)
        )

        # "Followed by a final 1 x 1 convolutional layer"
        # "For VOC we predict 5 boxes with 5 coordinates each and 20 classes per box so 125 filters"
        self.final_conv_1x1 = nn.Conv2d(
            in_channels=1024, 
            out_channels=self.num_anchors * (5 + self.C), 
            kernel_size=1
        )

    def forward(self, x):
        # Input: [Batch, 3, 416, 416] - assuming 416x416 input images
        # 1. Features
        x = self.features(x)
        
        # 2. Head
        x = self.conv_head(x)
        x = self.final_conv_1x1(x) 
        
        # 3. Reshape Logic
        # Current shape is [Batch, Channels, Grid_Y, Grid_X]
        # We want to access data like: x[batch, row, col, anchor_id]
        
        # Step 1: Move (Grid_Y, Grid_X) to the middle, Channels to the end.
        x = x.permute(0, 2, 3, 1) # Shape: [Batch, 125, 13, 13] -> [Batch, 13, 13, 125]
        
        # Step 2: Unpack the 125 channels into 5 anchors of 25 values each
        batch_size = x.size(0)
        grid_size = x.size(1)
        
        x = x.reshape(batch_size, grid_size, grid_size, self.num_anchors, 5 + self.C)
        # Final Shape: [Batch, 13, 13, 5, 25]
        return x

