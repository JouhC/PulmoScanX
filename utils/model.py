import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class ChestXrayDenseNet121(nn.Module):
    def __init__(self, num_classes=19, weights=DenseNet121_Weights.DEFAULT):
        super(ChestXrayDenseNet121, self).__init__()
        self.model = densenet121(weights=weights)
        
        # Replace the final classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)