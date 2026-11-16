import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from kidney_ai.feature_engineering.features import MedicalCLAHETransform, get_transforms, extract_radiomic_summary
from kidney_ai.models.network import KidneyStoneNet, GradCAMExplainer
from kidney_ai.monitoring.monitor import ClinicalModelMonitor

st.set_page_config(
    page_title="NephroAI | Clinical Kidney Stone Co-Pilot",
    page_icon="🩺",
    layout="wide",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("artifacts/models/kidney_stone_resnet50_best.pt")
TEST_REPORT_PATH = Path("artifacts/reports/clinical_test_report.json")
TRAIN_HISTORY_PATH = Path("artifacts/reports/training_history.json")


@st.cache_resource
def load_clinical_model():
    model = KidneyStoneNet(num_classes=2, pretrained=False).to(DEVICE)
    is_trained = False
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        is_trained = True
    model.eval()
    explainer = GradCAMExplainer(model)
    return model, explainer, is_trained


model, explainer, is_trained = load_clinical_model()
_, eval_transform = get_transforms((224, 224))
clahe_viz = MedicalCLAHETransform(clip_limit=2.0)
monitor = ClinicalModelMonitor()

st.title("🩺 NephroAI — Clinical Kidney Stone Diagnostic & Model Monitoring Suite")
st.caption("Enterprise Explainable AI System (ResNet50 + CLAHE Radiomics + Grad-CAM + Drift Monitoring)")

if not is_trained:
    st.warning(
        "⚠️ Trained model checkpoint not found yet at `artifacts/models/kidney_stone_resnet50_best.pt`. "
        "Run `uv run python -m kidney_ai.pipeline --epochs 5` first to train on your dataset."
    )

tab1, tab2, tab3 = st.tabs([
    "🔬 1. Clinician Scan Diagnosis & Explainability (Grad-CAM)",
    "📊 2. Clinical Validation & Model Performance",
    "📡 3. Production Drift & Audit Monitoring",
])

# --- TAB 1: DOCTOR INFERENCE & EXPLAINABILITY ---
with tab1:
    st.subheader("Upload Patient Ultrasound / CT Scan for AI Second-Opinion")
    uploaded_file = st.file_uploader("Select Kidney Scan (JPG, PNG, BMP)", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        start_t = time.perf_counter()
        pil_img = Image.open(uploaded_file).convert("RGB")
        clahe_img = clahe_viz(pil_img)
        radiomics = extract_radiomic_summary(pil_img)

        tensor_img = eval_transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor_img)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        stone_prob = float(probs[1])
        non_stone_prob = float(probs[0])
        pred_label = "Kidney Stone Detected" if stone_prob >= 0.5 else "Normal (Non-Stone)"
        confidence = max(stone_prob, non_stone_prob)

        heatmap = explainer.generate_heatmap(tensor_img, target_class=1)
        gradcam_overlay = explainer.overlay_heatmap(np.array(pil_img), heatmap)
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        record = monitor.log_prediction(
            scan_id=uploaded_file.name,
            predicted_label=pred_label,
            stone_probability=stone_prob,
            confidence=confidence,
            radiomics=radiomics,
            latency_ms=latency_ms,
        )

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("AI Diagnostic Finding", pred_label)
        col_m2.metric("Calculus (Stone) Probability", f"{stone_prob * 100:.1f}%")
        col_m3.metric("Model Confidence", f"{confidence * 100:.1f}%")
        col_m4.metric("Inference Latency", f"{latency_ms:.1f} ms")

        if record["needs_senior_radiologist_review"]:
            st.error("🚨 Low Confidence Flag: Automated triage recommends senior radiologist manual review.")
        else:
            st.success("✅ High-Confidence Prediction logged to hospital audit trail.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(pil_img, caption="Original Patient Scan", use_container_width=True)
        with c2:
            st.image(clahe_img, caption="CLAHE Contrast-Enhanced Feature View", use_container_width=True)
        with c3:
            st.image(gradcam_overlay, caption="Grad-CAM Attention Heatmap (Red = Stone Focus)", use_container_width=True)

        st.markdown("#### Extracted Radiomic Intensity Biomarkers")
        st.json(radiomics)

# --- TAB 2: CLINICAL VALIDATION METRICS ---
with tab2:
    st.subheader("Hold-Out Test Set & Training Convergence Metrics")
    if TEST_REPORT_PATH.exists():
        with open(TEST_REPORT_PATH, "r") as f:
            report = json.load(f)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Sensitivity (Recall)", f"{report.get('sensitivity_recall', report.get('recall', 0.0)) * 100:.2f}%")
        m2.metric("Specificity", f"{report.get('specificity', 0.0) * 100:.2f}%")
        m3.metric("Precision (PPV)", f"{report.get('precision', report.get('ppv', 0.0)) * 100:.2f}%")
        m4.metric("ROC-AUC Score", f"{report.get('roc_auc', 0.0):.4f}")
        m5.metric("False Negative Rate", f"{report.get('false_negative_rate', 0.0) * 100:.2f}%")

        cm = report.get("confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig_cm = px.imshow(
            z,
            text_auto=True,
            x=["Predicted: Normal", "Predicted: Stone"],
            y=["Actual: Normal", "Actual: Stone"],
            color_continuous_scale="Blues",
            title="Clinical Hold-Out Confusion Matrix",
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Run the training & testing pipeline (`uv run python -m kidney_ai.pipeline`) to populate clinical metrics.")

    if TRAIN_HISTORY_PATH.exists():
        with open(TRAIN_HISTORY_PATH, "r") as f:
            hist = json.load(f)
        df_hist = pd.DataFrame([
            {
                "Epoch": h["epoch"],
                "Train Loss": h["train_loss"],
                "Val Loss": h["val_metrics"]["loss"],
                "Val Accuracy": h["val_metrics"]["accuracy"],
                "Val F1": h["val_metrics"]["f1_score"],
            }
            for h in hist
        ])
        fig_loss = px.line(df_hist, x="Epoch", y=["Train Loss", "Val Loss"], markers=True, title="Loss Convergence Curve")
        st.plotly_chart(fig_loss, use_container_width=True)

# --- TAB 3: PRODUCTION MONITORING & DRIFT ---
with tab3:
    st.subheader("Real-Time Inference Audit Log & Scanner Drift Telemetry")
    logs_df = monitor.load_audit_logs()
    if not logs_df.empty:
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Scans Audited", len(logs_df))
        review_rate = logs_df["needs_senior_radiologist_review"].mean() * 100
        k2.metric("Human-in-the-Loop Escalation Rate", f"{review_rate:.1f}%")

        # Compute PSI on mean scan intensity comparing first half vs second half of logs
        mid = len(logs_df) // 2
        psi_val = monitor.calculate_psi(
            logs_df["mean_intensity"].iloc[:max(mid, 1)].values,
            logs_df["mean_intensity"].iloc[max(mid, 1):].values,
        )
        k3.metric("Scanner Intensity Drift (PSI)", f"{psi_val:.4f}", delta="Alert > 0.20" if psi_val > 0.2 else "Stable")

        fig_conf = px.histogram(
            logs_df,
            x="confidence",
            color="predicted_label",
            nbins=20,
            title="Production Prediction Confidence Distribution",
        )
        st.plotly_chart(fig_conf, use_container_width=True)
        st.dataframe(logs_df.tail(50), use_container_width=True)
    else:
        st.info("No live inference logs recorded yet. Upload a scan in Tab 1 to generate live telemetry.")
