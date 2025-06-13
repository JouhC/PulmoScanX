import numpy as np
import cv2
import torch

def reverseVector(vector):
    RLUNG = 44
    LLUNG = 50
    HEART = 26
    RCLAV = 23  # not used here
    # LCLAV = 23

    p1 = RLUNG * 2
    p2 = p1 + LLUNG * 2
    p3 = p2 + HEART * 2
    p4 = p3 + RCLAV * 2

    rl = vector[:p1].reshape(-1, 2)
    ll = vector[p1:p2].reshape(-1, 2)
    h  = vector[p2:p3].reshape(-1, 2)
    rc = vector[p3:p4].reshape(-1, 2)
    lc = vector[p4:].reshape(-1, 2)

    return rl, ll, h, rc, lc

def points_to_mask(points, image_size=224):
    """
    Converts polygon points into a binary mask.
    If points are normalized (0-1), ensure to multiply outside this function.
    """
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    points = np.array(points, dtype=np.int32)

    if points.shape[0] >= 3:
        cv2.fillPoly(mask, [points], 1)

    return torch.tensor(mask, dtype=torch.float32)

def process_landmarks_to_mask(landmarks_tensor, image_size=224, normalized=True):
    """
    Converts a (1, N, 2) landmark tensor into (3, H, W) binary masks for:
    Right Lung, Left Lung, Heart
    """
    landmarks = landmarks_tensor.squeeze(0).cpu().numpy()  # → (N, 2)
    if normalized:
        landmarks *= image_size

    rl, ll, h, _, _ = reverseVector(landmarks.flatten())

    right_lung_mask = points_to_mask(rl, image_size)
    left_lung_mask  = points_to_mask(ll, image_size)
    heart_mask      = points_to_mask(h, image_size)

    # Combine to shape: (3, H, W)
    return torch.stack([right_lung_mask, left_lung_mask, heart_mask], dim=0)
