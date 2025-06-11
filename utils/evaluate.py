import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import pandas as pd         

def evaluate(model, dataloader, device, threshold=0.5, pos_weight=None):
    model.eval()

    # Use pos_weight if passed
    if pos_weight is not None:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    all_probs = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # raw logits
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)  # convert logits to probabilities
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    # Stack all batches
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    # Binarize with threshold
    pred_labels = (all_probs > threshold).int()

    # Convert to numpy
    y_pred = pred_labels.numpy()
    y_true = all_labels.numpy()

    # Compute evaluation metrics
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)  # still optional in multilabel

    avg_loss = total_loss / len(dataloader.dataset)

    print(f"\nEvaluation Results:")
    print(f"Avg Loss   : {avg_loss:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"Accuracy   : {accuracy:.4f}")

    return {
        "loss": avg_loss,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy
    }

def evaluate_per_class(model, dataloader, device, class_names, threshold=0.5):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()

    all_probs = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    pred_labels = (all_probs > threshold).int()
    y_pred = pred_labels.numpy()
    y_true = all_labels.numpy()

    # Compute per-class F1
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Create a dataframe
    df = pd.DataFrame({
        'Pathology': class_names,
        'F1 Score': f1s
    }).sort_values(by='F1 Score', ascending=False)

    print("\nPer-Class F1 Scores:")
    print(df.to_string(index=False))

    avg_loss = total_loss / len(dataloader.dataset)

    return {
        "loss": avg_loss,
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "per_class_f1": df
    }