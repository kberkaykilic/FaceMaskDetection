import torch
import torch.nn as nn
import torchvision


class SSDLite(nn.Module):
    def __init__(self, num_classes=3):
        super(SSDLite, self).__init__()

        self.backbone = self._get_backbone()
        self.extras = self._get_extras()
        self.loc_layers = self._get_loc_layers()
        self.conf_layers = self._get_conf_layers(num_classes)
        self._initialize_weights()

    def _get_backbone(self):
        mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
        return nn.Sequential(*list(mobilenet.features[:14]))

    def _get_extras(self):
        return nn.ModuleList([
            self._create_extra_layer(96, 256, 512),
            self._create_extra_layer(512, 128, 256),
            self._create_extra_layer(256, 128, 256)
        ])

    def _create_extra_layer(self, in_channels, mid_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def _get_loc_layers(self):
        return nn.ModuleList([
            self._create_loc_conf_layer(96, 24),
            self._create_loc_conf_layer(512, 24),
            self._create_loc_conf_layer(256, 24),
            self._create_loc_conf_layer(256, 24)
        ])

    def _get_conf_layers(self, num_classes):
        return nn.ModuleList([
            self._create_loc_conf_layer(96, num_classes * 6),
            self._create_loc_conf_layer(512, num_classes * 6),
            self._create_loc_conf_layer(256, num_classes * 6),
            self._create_loc_conf_layer(256, num_classes * 6)
        ])

    def _create_loc_conf_layer(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features = [self.backbone(x)]
        for extra_layer in self.extras:
            features.append(extra_layer(features[-1]))

        loc_preds, conf_preds = [], []
        for i, feature in enumerate(features):
            loc_preds.append(self.loc_layers[i](feature).permute(0, 2, 3, 1).contiguous())
            conf_preds.append(self.conf_layers[i](feature).permute(0, 2, 3, 1).contiguous())

        loc_preds = torch.cat([o.view(o.size(0), -1) for o in loc_preds], 1)
        conf_preds = torch.cat([o.view(o.size(0), -1) for o in conf_preds], 1)

        return loc_preds, conf_preds
