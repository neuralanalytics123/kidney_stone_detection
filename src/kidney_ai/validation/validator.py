"""
Clinical Validation Module:
Evaluates model performance on validation/test sets computing Accuracy,
Sensitivity (Recall), Specificity, and F1-Score for clinical reliability.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np


def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(data_loader.dataset), 1)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)

    # Compute specificity from confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy, 4),
        "sensitivity_recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
    }