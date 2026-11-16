"""
Production Clinical Monitoring & Data Drift Detection Module:
Logs every inference event (timestamp, prediction, confidence, radiomic intensity features)
and computes Population Stability Index (PSI) & low-confidence alerts for healthcare QA teams.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd


class ClinicalModelMonitor:
    def __init__(self, log_path: str = "artifacts/reports/inference_audit_log.jsonl", confidence_threshold: float = 0.75):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold

    def log_prediction(
        self,
        scan_id: str,
        predicted_label: str,
        stone_probability: float,
        confidence: float,
        radiomics: Dict[str, float],
        latency_ms: float,
    ) -> Dict[str, Any]:
        needs_human_review = confidence < self.confidence_threshold
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scan_id": scan_id,
            "predicted_label": predicted_label,
            "stone_probability": round(stone_probability, 4),
            "confidence": round(confidence, 4),
            "needs_senior_radiologist_review": needs_human_review,
            "latency_ms": round(latency_ms, 2),
            **radiomics,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def load_audit_logs(self) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame()
        records = []
        with open(self.log_path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    @staticmethod
    def calculate_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculates Population Stability Index (PSI) to detect scanner/image intensity drift."""
        if len(baseline) < 5 or len(current) < 5:
            return 0.0
        quantiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(baseline, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        base_counts = np.histogram(baseline, bins=bin_edges)[0] / len(baseline)
        curr_counts = np.histogram(current, bins=bin_edges)[0] / len(current)

        base_counts = np.clip(base_counts, 1e-4, 1.0)
        curr_counts = np.clip(curr_counts, 1e-4, 1.0)

        psi = np.sum((curr_counts - base_counts) * np.log(curr_counts / base_counts))
        return round(float(psi), 4)
