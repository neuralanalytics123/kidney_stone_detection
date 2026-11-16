"""
Data Preprocessing Module:
Handles directory ingestion (case-insensitive 'non-stone' and 'stone'/'Stone' folders),
corrupt image filtering, stratified splitting (train/val/test), and metadata indexing.
"""
import os
import shutil
from pathlib import Path
from typing import Tuple, Dict, List
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from loguru import logger


class ClinicalDatasetPreprocessor:
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed", seed: int = 42):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.seed = seed
        self.valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def _discover_class_folders(self) -> Dict[str, Path]:
        """Case-insensitive discovery of 'non-stone' and 'stone' folders."""
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir.resolve()}")

        mapping = {}
        for item in self.raw_dir.iterdir():
            if item.is_dir():
                name_lower = item.name.lower().strip()
                if name_lower in ["non-stone", "non_stone", "nonstone"]:
                    mapping["non-stone"] = item
                elif name_lower in ["stone", "stones"]:
                    mapping["stone"] = item

        if len(mapping) < 2:
            raise ValueError(
                f"Expected both 'non-stone' and 'stone' folders in {self.raw_dir}. "
                f"Found: {[d.name for d in self.raw_dir.iterdir() if d.is_dir()]}"
            )
        return mapping

    def validate_and_index(self) -> pd.DataFrame:
        """Verifies image integrity and builds a structured manifest DataFrame."""
        class_folders = self._discover_class_folders()
        records = []
        corrupt_count = 0

        for label_name, folder_path in class_folders.items():
            label_id = 1 if label_name == "stone" else 0
            for img_path in folder_path.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in self.valid_extensions:
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        with Image.open(img_path) as img:
                            width, height = img.size
                            mode = img.mode
                        records.append({
                            "filepath": str(img_path.resolve()),
                            "filename": img_path.name,
                            "label_name": label_name,
                            "label": label_id,
                            "width": width,
                            "height": height,
                            "mode": mode,
                        })
                    except Exception as e:
                        corrupt_count += 1
                        logger.warning(f"Skipping corrupted image {img_path}: {e}")

        df = pd.DataFrame(records)
        logger.info(f"Indexed {len(df)} valid medical scans (skipped {corrupt_count} corrupt files).")
        return df

    def create_splits(
        self, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Creates stratified Train, Validation, and Test manifests and saves CSVs."""
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Splits must sum to 1.0"
        df = self.validate_and_index()

        if len(df) < 4:
            logger.warning("Very small sample dataset detected; duplicating rows minimally for pipeline smoke-testing.")
            df = pd.concat([df] * 4, ignore_index=True)

        train_df, temp_df = train_test_split(
            df, test_size=(1.0 - train_ratio), stratify=df["label"], random_state=self.seed
        )
        relative_val = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1.0 - relative_val), stratify=temp_df["label"], random_state=self.seed
        )

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(self.processed_dir / "train_manifest.csv", index=False)
        val_df.to_csv(self.processed_dir / "val_manifest.csv", index=False)
        test_df.to_csv(self.processed_dir / "test_manifest.csv", index=False)

        logger.success(
            f"Saved manifests to {self.processed_dir}: "
            f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )
        return train_df, val_df, test_df


if __name__ == "__main__":
    preprocessor = ClinicalDatasetPreprocessor()
    preprocessor.create_splits()
