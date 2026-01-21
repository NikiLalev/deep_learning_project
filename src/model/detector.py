import torch
import torch.nn as nn
from src.model.classification_imagenet import YOLOPretrain


class YOLOv1(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20):
        super(YOLOv1, self).__init__()
        
        # Grid size, number of bounding boxes, number of classes
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        def _conv_block(in_c, out_c, kernel_size, stride=1):
            # Darknet uses pad=1 aka padding = kernel_size // 2. Reference: https://github.com/pjreddie/darknet/blob/master/src/parser.c
            padding = kernel_size // 2
            return nn.Sequential(
                # Bias is False because we use BatchNorm. Reason for using batch norm: https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_c),
                # Paper uses Leaky ReLU for all layers except the final layer where linear activation is used
                nn.LeakyReLU(0.1, inplace=True),
            )

        # 1. Same as YOLOPretrain
        self.features = nn.Sequential(
            # Conv 1: 7x7, in_channels=3 - 224x224x3, out_channels=64 - 112x112x64, stride=2
            _conv_block(3, 64, kernel_size=7, stride=2),
            # MaxPool 1: 2x2, stride=2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv 2: 3x3, stride=1
            _conv_block(64, 192, kernel_size=3),
            # MaxPool 2: 2x2, stride=2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv 3-6: (1x1, 3x3) x2, stride=1
            _conv_block(192, 128, kernel_size=1),
            _conv_block(128, 256, kernel_size=3),
            _conv_block(256, 256, kernel_size=1),
            _conv_block(256, 512, kernel_size=3),
            # MaxPool 3: 2x2, stride=2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv 7-14: (1x1, 3x3) x4, stride 1
            *[
                layer
                for _ in range(4)
                for layer in (
                    _conv_block(512, 256, kernel_size=1),
                    _conv_block(256, 512, kernel_size=3),
                )
            ],
            # Conv 15: 1x1, stride=1
            _conv_block(512, 512, kernel_size=1),
            # Conv 16: 3x3, stride=1
            _conv_block(512, 1024, kernel_size=3),
            # MaxPool 4: 2x2, stride=2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv 17-20: (1x1, 3x3) x2, stride=1
            _conv_block(1024, 512, kernel_size=1),
            _conv_block(512, 1024, kernel_size=3),
            _conv_block(1024, 512, kernel_size=1),
            _conv_block(512, 1024, kernel_size=3),
        )

        # 2. Detection layers - 4 additional convolutional layers
        self.detection_layers = nn.Sequential(
            # Conv 21 3x3, stride=1
            _conv_block(1024, 1024, kernel_size=3, stride=1),
            # Conv 22 3x3, stride=2
            _conv_block(1024, 1024, kernel_size=3, stride=2),
            # Conv 23 3x3, stride=1
            _conv_block(1024, 1024, kernel_size=3, stride=1),
            # Conv 24 3x3, stride=1
            _conv_block(1024, 1024, kernel_size=3, stride=1),
        )

        # 3. 2 additional fully connected layers
        self.fcl = nn.Sequential(
            nn.Flatten(),
            # Fully Connected 1
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            # "A dropout layer with rate = .5 after the first connected layer prevents co-adaptation between layers"
            nn.Dropout(0.5),
            # Fully Connected 2 with linear activation
            # For PASCAL (S=7, B=2, C=20), Output = S * S * (C + B * 5) = 7 * 7 * (20 + 2 * 5) = 1470
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.detection_layers(x)
        x = self.fcl(x)
        # Reshape so (Batch, 1470) becomes (Batch, 7, 7, 30)
        x = x.view(-1, self.S, self.S, (self.C + self.B * 5))
        return x

    def load_pretrain_weights(self, pretrain_model):
        """
        Loads weights from the pretrained classification model
        into the first 20 layers of the detector.
        """
        pretrain_dict = pretrain_model.state_dict()
        model_dict = self.state_dict()

        # Get the feature weights from the pretrained model
        # that match the detector's feature layers.
        pretrained_dict = {
            k: v
            for k, v in pretrain_dict.items()
            if k in model_dict and "features" in k
        }

        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # Load the new state dict
        self.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)} layers from pretraining.")
