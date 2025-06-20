import torch
import torchvision.transforms as transforms
import kornia.augmentation as K
from torchvision.transforms import InterpolationMode

#resize = (384, 384)
resize = (224, 224)

class RandomApplyWrapper(torch.nn.Module):
    def __init__(self, transform: torch.nn.Module, p: float = 0.5):
        super().__init__()
        self.transform = transform
        self.p = p

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.dim() != 4:
            raise ValueError(f"Expected input of shape [B, C, H, W], but got {x.shape}")
        
        # Apply transform with probability p
        if torch.rand(1).item() < self.p:
            out = self.transform(x)
            if out.dim() != 4:
                raise ValueError(f"Transform output must be 4D [B, C, H, W], got {out.shape}")
            return out
        return x

AUGMENTATIONS_TRAIN = torch.nn.Sequential(
    # Random Gamma
    RandomApplyWrapper(
        K.RandomGamma(gamma=(0.6, 1.2), p=1.0),
        p=0.33
    ),
    
    # Random Brightness + Contrast
    RandomApplyWrapper(
        torch.nn.Sequential(
            K.RandomBrightness(0.2, p=1.0),
            K.RandomContrast(0.2, p=1.0)
        ),
        p=0.33
    ),
    
    # Gaussian Blur
    RandomApplyWrapper(
        K.RandomGaussianBlur((3, 5), (0.1, 2.0), p=1.0),
        p=0.25
    ),
    
    # Motion Blur
    RandomApplyWrapper(
        K.RandomMotionBlur(kernel_size=3, angle=20.0, direction=0.5, p=1.0),
        p=0.25
    ),
    
    # Horizontal Flip
    K.RandomHorizontalFlip(p=0.5),

    # Affine Transform (rotate/scale/translate)
    K.RandomAffine(
        degrees=20,
        translate=(0.2, 0.2),
        scale=(0.8, 1.2),
        p=1.0,
        padding_mode="zeros"
    ),

    # Normalize
    K.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225])
    )
)

train_transform = transforms.Compose([
    transforms.Resize(resize),
    transforms.ToTensor(),
    lambda x: AUGMENTATIONS_TRAIN(x.unsqueeze(0)).squeeze(0)  # [C,H,W] → [1,C,H,W] → apply → [C,H,W]
])

val_test_transform = transforms.Compose([
    transforms.Resize(resize),  # Resize for EfficientNet-B4
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225))
])