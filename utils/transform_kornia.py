import torch
import torchvision.transforms as transforms
import kornia.augmentation as K
from torchvision.transforms import InterpolationMode

AUGMENTATIONS_TRAIN = torch.nn.Sequential(
    # Random choice between Gamma, Brightness/Contrast, or CLAHE (approximated)
    K.RandomApply([
        torch.nn.Sequential(
            K.RandomGamma(gamma=(0.6, 1.2), p=0.9),  # gamma ~ inverse of Albumentations scale
        )
    ], p=0.33),
    K.RandomApply([
        torch.nn.Sequential(
            K.RandomBrightness(0.2, p=1.0),
            K.RandomContrast(0.2, p=1.0),
        )
    ], p=0.33),
    # CLAHE has no direct PyTorch or Kornia equivalent — could be skipped or added manually

    # Blur or Motion Blur
    K.RandomApply([
        K.RandomGaussianBlur((3, 5), (0.1, 2.0)),  # Approximating Blur
    ], p=0.25),
    K.RandomApply([
        K.RandomMotionBlur(kernel_size=3, angle=20.0, direction=0.5),
    ], p=0.25),

    # Horizontal Flip
    K.RandomHorizontalFlip(p=0.5),

    # Shift, Scale, Rotate
    K.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), p=1.0, padding_mode="zeros"),

    # Normalize
    K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]))
)

train_transform = transforms.Compose([
    transforms.Resize((380, 380)),  # Resize for EfficientNet-B4
    transforms.ToTensor(),                 # Convert PIL image to Tensor
    lambda x: AUGMENTATIONS_TRAIN(x),  # Apply Kornia transforms
])

val_test_transform = transforms.Compose([
    transforms.Resize((380, 380)),  # Resize for EfficientNet-B4
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225))
])