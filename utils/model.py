import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class ChestXrayDenseNet121(nn.Module):
    def __init__(self, num_classes=19, weights=DenseNet121_Weights.DEFAULT, dropout=0.2):
        super(ChestXrayDenseNet121, self).__init__()

        # Load pretrained DenseNet-121
        self.base_model = densenet121(weights=weights)

        # Freeze early layers if needed (optional)
        # for param in self.base_model.features[:6].parameters():
        #     param.requires_grad = False

        # Replace classifier head
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
    

class ChestXrayDenseNet121WithMask(nn.Module):
    def __init__(self, num_classes=19, weights=DenseNet121_Weights.DEFAULT, dropout=0.2):
        super(ChestXrayDenseNet121WithMask, self).__init__()

        # Load pretrained DenseNet-121
        self.base_model = densenet121(weights=weights)

        # === Modify the first conv layer to accept 6-channel input ===
        old_conv = self.base_model.features.conv0
        self.base_model.features.conv0 = nn.Conv2d(
            in_channels=6,  # ← 3 for RGB + 3 for masks
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # Optional: initialize the new 6-channel conv weights
        with torch.no_grad():
            self.base_model.features.conv0.weight[:, :3] = old_conv.weight  # copy RGB weights
            self.base_model.features.conv0.weight[:, 3:] = old_conv.weight[:, :3]  # duplicate for masks

        # === Replace classifier head ===
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


class EfficientNetV2_MultiLabel(nn.Module):
    def __init__(self, num_classes=19):  # change this for your case
        super(EfficientNetV2_MultiLabel, self).__init__()
        
        # Load pretrained EfficientNetV2
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.backbone = efficientnet_v2_s(weights=weights)
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)