import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm
import numpy as np

def train(model, train_loader, val_loader, device, epochs=10, lr=1e-4, save_path=None,
          file_name="PulmoScanX_v2", pos_weight=None, multi_gpu=False, device_ids=None):
    
    # Set visible CUDA devices
    if device_ids is not None:
        torch.cuda.set_device(device_ids[0])
        device = torch.device(f"cuda:{device_ids[0]}")

    model.to(device)

    # Multi-GPU support
    if multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {len(device_ids)} GPUs: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device)) if pos_weight is not None else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Training Loss: {avg_loss:.4f}")

        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f} | Accuracy: {val_metrics['acc']:.4f}\n")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            if save_path:
                torch.save(model.module.state_dict() if multi_gpu else model.state_dict(),
                           f"{save_path}/{file_name}_checkpoint.pth")


def validate(model, val_loader, criterion, device, threshold=0.5):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > threshold).astype(int)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Metrics (macro average for multi-label)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_preds, average='macro')
    except ValueError:
        auc = 0.0
    acc = accuracy_score(all_labels, all_preds)

    return avg_val_loss, {'f1': f1, 'auc': auc, 'acc': acc}


def compute_pos_weight(dataloader, num_classes):
    total_counts = torch.zeros(num_classes)
    total_samples = 0

    for _, labels in dataloader:
        total_counts += labels.sum(dim=0)
        total_samples += labels.size(0)

    num_pos = total_counts
    num_neg = total_samples - num_pos

    pos_weight = num_neg / (num_pos + 1e-6)  # avoid division by zero
    return pos_weight