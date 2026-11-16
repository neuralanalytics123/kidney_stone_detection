"""
Automated Unit & Integration Tests for CI/CD Pipeline:
Validates data ingestion, CLAHE feature engineering, model forward pass,
Grad-CAM explainability, and FastAPI clinical endpoint.
"""
import io
from PIL import Image
import torch
from fastapi.testclient import TestClient

from kidney_ai.data_preprocessing.preprocessor import ClinicalDatasetPreprocessor
from kidney_ai.feature_engineering.features import MedicalCLAHETransform, extract_radiomic_summary
from kidney_ai.models.network import KidneyStoneNet, GradCAMExplainer
from kidney_ai.deploying.api import app


def test_data_discovery_and_splits():
    prep = ClinicalDatasetPreprocessor(raw_dir="data/raw", processed_dir="data/processed")
    train_df, val_df, test_df = prep.create_splits()
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0


def test_clahe_and_radiomics():
    dummy_img = Image.new("RGB", (256, 256), color=(120, 120, 120))
    clahe = MedicalCLAHETransform()
    out = clahe(dummy_img)
    assert out.size == (256, 256)
    stats = extract_radiomic_summary(dummy_img)
    assert "mean_intensity" in stats
    assert "p95_hyper_echoic_intensity" in stats


def test_model_and_gradcam():
    model = KidneyStoneNet(num_classes=2, pretrained=False)
    explainer = GradCAMExplainer(model)
    x = torch.randn(1, 3, 224, 224)
    heatmap = explainer.generate_heatmap(x, target_class=1)
    assert heatmap.shape == (224, 224)


def test_fastapi_predict_endpoint():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200

    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(80, 80, 80)).save(buf, format="JPEG")
    buf.seek(0)

    resp = client.post("/predict", files={"file": ("test_scan.jpg", buf, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert "diagnosis" in data
    assert "gradcam_overlay_base64" in data
