import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights

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
