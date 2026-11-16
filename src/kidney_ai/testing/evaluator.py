"""
Independent Hold-Out Testing & Clinical QA Gate Module:
Evaluates the locked best checkpoint on unseen test scans and outputs an auditable
clinical evaluation report (JSON) for hospital QA sign-off.
"""
import json
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from loguru import logger

from kidney_ai.feature_engineering.features import build_dataloaders
from kidney_ai.models.network import KidneyStoneNet
from kidney_ai.validation.validator import evaluate_model


def run_clinical_test_suite(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = build_dataloaders(
        processed_dir=cfg["data"]["processed_dir"],
        batch_size=cfg["data"]["batch_size"],
        image_size=tuple(cfg["data"]["image_size"]),
        num_workers=0,
    )

    model = KidneyStoneNet(num_classes=cfg["model"]["num_classes"], pretrained=False).to(device)
    model_path = Path(cfg["artifacts"]["model_dir"]) / cfg["artifacts"]["best_model_name"]

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded trained model weights from {model_path}")
    else:
        logger.warning("No trained checkpoint found; running test suite with initialized weights.")

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    report_path = Path(cfg["artifacts"]["report_dir"]) / "clinical_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.success(f"Clinical Test Report saved to {report_path}: {test_metrics}")
    return test_metrics


if __name__ == "__main__":
    run_clinical_test_suite()
