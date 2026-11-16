"""
Training Orchestrator Module (CPU-Optimized MobileNetV4):
Trains KidneyStoneNet with AdamW, cosine annealing scheduler, early stopping,
and saves best model weights + training history JSON for clinical audit.
"""
import json
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from kidney_ai.data_preprocessing.preprocessor import ClinicalDatasetPreprocessor
from kidney_ai.feature_engineering.features import build_dataloaders
from kidney_ai.models.network import KidneyStoneNet
from kidney_ai.validation.validator import evaluate_model


def train_pipeline(config_path: str = "configs/config.yaml", epochs_override: int | None = None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Force CPU execution for lightweight production deployment on local laptops
    device = torch.device("cpu")
    logger.info(f"Running training pipeline on CPU device: {device}")

    # Step 1: Preprocessing
    preprocessor = ClinicalDatasetPreprocessor(
        raw_dir=cfg["data"]["raw_dir"],
        processed_dir=cfg["data"]["processed_dir"],
        seed=cfg["project"]["seed"],
    )
    preprocessor.create_splits(
        train_ratio=cfg["data"]["train_split"],
        val_ratio=cfg["data"]["val_split"],
        test_ratio=cfg["data"]["test_split"],
    )

    # Step 2: Feature Engineering & Dataloaders
    train_loader, val_loader, _ = build_dataloaders(
        processed_dir=cfg["data"]["processed_dir"],
        batch_size=cfg["data"]["batch_size"],
        image_size=tuple(cfg["data"]["image_size"]),
        num_workers=0,
    )

    # Step 3: Model Initialization (MobileNetV4 Small Backbone)
    model = KidneyStoneNet(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        dropout_rate=cfg["model"]["dropout_rate"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    epochs = epochs_override if epochs_override is not None else cfg["training"]["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    model_dir = Path(cfg["artifacts"]["model_dir"])
    report_dir = Path(cfg["artifacts"]["report_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / cfg["artifacts"]["best_model_name"]

    best_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / max(len(train_loader.dataset), 1)
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)
        logger.info(
            f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val Sensitivity: {val_metrics['sensitivity_recall']:.4f} | "
            f"Val F1: {val_metrics['f1_score']:.4f}"
        )

        if val_metrics["f1_score"] >= best_f1:
            best_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), best_model_path)
            logger.success(f"Saved new best clinical model checkpoint to {best_model_path}")

    with open(report_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return best_model_path


if __name__ == "__main__":
    train_pipeline()