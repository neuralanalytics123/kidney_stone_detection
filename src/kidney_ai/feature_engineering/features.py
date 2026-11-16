"""
Feature Engineering & Medical Image Augmentation Module:
Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance ultrasound/CT
calcification contrast, domain-specific augmentations, and PyTorch Dataset construction.
"""
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MedicalCLAHETransform:
    """Enhances local contrast in medical ultrasound/CT scans to highlight calculi (stones)."""
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img.convert("RGB"))
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l_enhanced = clahe.apply(l_channel)
        merged = cv2.merge((l_enhanced, a_channel, b_channel))
        rgb_enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_enhanced)


def get_transforms(image_size: Tuple[int, int] = (224, 224)):
    """Returns training and inference/validation transformation pipelines."""
    train_transform = transforms.Compose([
        MedicalCLAHETransform(clip_limit=2.0),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        MedicalCLAHETransform(clip_limit=2.0),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def extract_radiomic_summary(img: Image.Image) -> dict:
    """Extracts basic intensity & texture radiomic descriptors for clinician explainability."""
    gray = np.array(img.convert("L"), dtype=np.float32)
    return {
        "mean_intensity": round(float(np.mean(gray)), 2),
        "std_intensity": round(float(np.std(gray)), 2),
        "p95_hyper_echoic_intensity": round(float(np.percentile(gray, 95)), 2),
        "contrast_ratio": round(float((np.max(gray) - np.min(gray)) / (np.mean(gray) + 1e-6)), 3),
    }


class KidneyScanDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, transform=None):
        self.df = manifest_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def build_dataloaders(
    processed_dir: str = "data/processed",
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 0,
):
    p_dir = Path(processed_dir)
    train_df = pd.read_csv(p_dir / "train_manifest.csv")
    val_df = pd.read_csv(p_dir / "val_manifest.csv")
    test_df = pd.read_csv(p_dir / "test_manifest.csv")

    train_tf, eval_tf = get_transforms(image_size)

    train_loader = DataLoader(KidneyScanDataset(train_df, train_tf), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(KidneyScanDataset(val_df, eval_tf), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(KidneyScanDataset(test_df, eval_tf), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
