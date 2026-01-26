import torch
import torch.nn as nn

class YOLOPretrain(nn.Module):
    def __init__(self, num_classes=1000):
        super(YOLOPretrain, self).__init__()
        
        def _conv_block(in_c, out_c, kernel_size, stride=1):
            # Darknet uses pad=1 aka padding = kernel_size // 2. Reference: https://github.com/pjreddie/darknet/blob/master/src/parser.c
            padding = kernel_size // 2
            return nn.Sequential(
                # Bias is False because we use BatchNorm. Reason for using batch norm: https://github.com/pjreddie/darknet/blob/master/cfg/extraction.cfg
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1, inplace=True)
            )

        # First 20 layers from figure 3 in the oroginal YOLO paper
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
            *[layer for _ in range(4) for layer in (
                _conv_block(512, 256, kernel_size=1),
                _conv_block(256, 512, kernel_size=3)
            )],

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

        # Final Classifier Layer
        self.classifier = nn.Sequential(
            # Average pool layer (from 1024x7x7 to 1024x1x1)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # Fully connected layer
            nn.Linear(1024, num_classes),
            nn.LeakyReLU(0.1, inplace=True)            
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x