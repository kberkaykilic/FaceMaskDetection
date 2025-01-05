import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_activation=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1) if use_activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPBlock, self).__init__()
        self.split = ConvBlock(in_channels, out_channels // 2, 1, 1, 0)
        self.blocks = nn.Sequential(
            *[ConvBlock(out_channels // 2, out_channels // 2, 3, 1, 1) for _ in range(num_blocks)]
        )
        self.concat = ConvBlock(out_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.split(x)
        x2 = self.blocks(x1)
        return self.concat(torch.cat([x1, x2], dim=1))

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels // 2, 1, 1, 0)
        self.pooling = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in [5, 9, 13]])
        self.concat = ConvBlock(in_channels * 2, out_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        pooled = [pool(x1) for pool in self.pooling]
        return self.concat(torch.cat([x1, *pooled], dim=1))

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.stem = ConvBlock(3, 32, 3, 1, 1)
        self.stage1 = CSPBlock(32, 64, 1)
        self.stage2 = CSPBlock(64, 128, 3)
        self.stage3 = CSPBlock(128, 256, 3)
        self.stage4 = CSPBlock(256, 512, 1)
        self.sppf = SPPF(512, 512)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.sppf(x4)
        return x1, x2, x3, x4, x5

class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        self.topdown1 = CSPBlock(512, 256, 1)
        self.topdown2 = CSPBlock(256, 128, 1)
        self.bottomup1 = CSPBlock(128, 256, 1)
        self.bottomup2 = CSPBlock(256, 512, 1)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features
        t1 = self.topdown1(F.interpolate(x5, scale_factor=2) + x4)
        t2 = self.topdown2(F.interpolate(t1, scale_factor=2) + x3)
        b1 = self.bottomup1(F.interpolate(t2, scale_factor=0.5) + x2)
        b2 = self.bottomup2(F.interpolate(b1, scale_factor=0.5) + x1)
        return t2, b1, b2

class DetectionHead(nn.Module):
    def __init__(self, num_classes):
        super(DetectionHead, self).__init__()
        self.detect_layers = nn.ModuleList([
            nn.Conv2d(128, num_classes, 1),
            nn.Conv2d(256, num_classes, 1),
            nn.Conv2d(512, num_classes, 1)
        ])

    def forward(self, features):
        return [layer(feature) for layer, feature in zip(self.detect_layers, features)]

class YOLOv8(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv8, self).__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = DetectionHead(num_classes)

    def forward(self, x):
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        return self.head(neck_features)
