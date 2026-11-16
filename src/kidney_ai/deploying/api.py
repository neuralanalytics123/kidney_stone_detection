"""
Production FastAPI Deployment Service:
Exposes REST endpoints (/health, /predict, /metrics) for hospital PACS/EHR integration
and returns clinical probabilities, Grad-CAM base64 heatmaps, and radiomic summaries.
"""
import base64
import io
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from kidney_ai.feature_engineering.features import get_transforms, extract_radiomic_summary
from kidney_ai.models.network import KidneyStoneNet, GradCAMExplainer
from kidney_ai.monitoring.monitor import ClinicalModelMonitor

app = FastAPI(
    title="Kidney Stone Detection Clinical Inference API",
    version="1.0.0",
    description="Enterprise AI diagnostic assistant with Grad-CAM explainability and drift monitoring.",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("artifacts/models/kidney_stone_resnet50_best.pt")

model = KidneyStoneNet(num_classes=2, pretrained=False).to(DEVICE)
if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

_, EVAL_TRANSFORM = get_transforms((224, 224))
explainer = GradCAMExplainer(model)
monitor = ClinicalModelMonitor()


class PredictionResponse(BaseModel):
    scan_id: str
    diagnosis: str
    stone_probability: float
    non_stone_probability: float
    confidence: float
    needs_senior_radiologist_review: bool
    radiomics: dict
    gradcam_overlay_base64: str
    latency_ms: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL_PATH.exists(),
        "device": str(DEVICE),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_scan(file: UploadFile = File(...)):
    start_time = time.perf_counter()
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid medical image file: {e}")

    radiomics = extract_radiomic_summary(pil_img)
    tensor_img = EVAL_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    non_stone_prob = float(probs[0])
    stone_prob = float(probs[1])
    pred_idx = int(np.argmax(probs))
    diagnosis = "Kidney Stone Detected" if pred_idx == 1 else "Normal (Non-Stone)"
    confidence = max(non_stone_prob, stone_prob)

    # Generate Grad-CAM visual explanation
    heatmap = explainer.generate_heatmap(tensor_img, target_class=1)
    overlay_rgb = explainer.overlay_heatmap(np.array(pil_img), heatmap)
    buf = io.BytesIO()
    Image.fromarray(overlay_rgb).save(buf, format="PNG")
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    audit_record = monitor.log_prediction(
        scan_id=file.filename or "uploaded_scan",
        predicted_label=diagnosis,
        stone_probability=stone_prob,
        confidence=confidence,
        radiomics=radiomics,
        latency_ms=latency_ms,
    )

    return PredictionResponse(
        scan_id=audit_record["scan_id"],
        diagnosis=diagnosis,
        stone_probability=round(stone_prob, 4),
        non_stone_probability=round(non_stone_prob, 4),
        confidence=round(confidence, 4),
        needs_senior_radiologist_review=audit_record["needs_senior_radiologist_review"],
        radiomics=radiomics,
        gradcam_overlay_base64=gradcam_b64,
        latency_ms=round(latency_ms, 2),
    )
