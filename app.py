"""
Advanced Clinical Kidney Stone Detection & Explainability Dashboard (Streamlit UI):
Uses clean, native Streamlit components for maximum readability and a standard web app look.
"""
import io
import json
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from kidney_ai.models.network import KidneyStoneNet, GradCAMExplainer

# Page Configuration
st.set_page_config(
    page_title="RenalGuard AI | Clinical Diagnostic Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_diagnostic_model():
    device = torch.device("cpu")
    model = KidneyStoneNet(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load("artifacts/models/kidney_stone_resnet50_best.pt", map_location=device))
    model.to(device)
    model.eval()
    explainer = GradCAMExplainer(model)
    return model, explainer, device

with st.spinner("Initializing EfficientNet-B0 Diagnostic Core & Grad-CAM Engine..."):
    model, explainer, device = load_diagnostic_model()

# Header
st.title("🔬 Clinical Scan Analysis & Explainability Suite")
st.markdown("Upload a patient ultrasound or CT scan to execute real-time neural inference paired with **Grad-CAM** visual attribution mapping.")

# Sidebar Scan Controls (Using native Streamlit elements)
st.sidebar.image("https://img.icons8.com/color/96/kidney.png", width=70)
st.sidebar.title("RenalGuard AI")
st.sidebar.caption("Enterprise Clinical Decision Support System")
st.sidebar.markdown("---")

st.sidebar.header("📁 Patient Scan Input")
uploaded_file = st.sidebar.file_uploader("Upload Medical Scan (JPG, PNG)", type=["jpg", "jpeg", "png"])

analysis_mode = st.sidebar.radio(
    "Visualization Perspective",
    ["Clinical View (Heatmap Overlay)", "Raw Scan Analysis", "Side-by-Side Comparison"]
)

alpha_slider = st.sidebar.slider(
    "Heatmap Blend Opacity (Grad-CAM)",
    min_value=0.1, max_value=0.9, value=0.45, step=0.05,
    help="Adjust transparency of the attention overlay for detailed clinician inspection."
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    orig_np = np.array(image)

    # Preprocess tensor
    img_resized = image.resize((224, 224))
    img_arr = np.array(img_resized) / 255.0
    tensor = torch.tensor(img_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        stone_prob = probs[0][1].item() * 100
        normal_prob = probs[0][0].item() * 100
        confidence = probs[0][pred_class].item() * 100

    # Generate Grad-CAM heatmap
    heatmap = explainer.generate_heatmap(tensor, target_class=pred_class)
    blended_overlay = GradCAMExplainer.overlay_heatmap(orig_np, heatmap, alpha=alpha_slider)

    # Top Metric Row (Native Streamlit metrics)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Primary Classification", "Stone Detected" if pred_class == 1 else "Normal / Clear")
    m2.metric("Confidence Score", f"{confidence:.2f}%")
    m3.metric("Calculi Probability", f"{stone_prob:.1f}%")
    m4.metric("Baseline Probability", f"{normal_prob:.1f}%")

    st.markdown("---")

    # Main Visualization Layout
    if analysis_mode == "Side-by-Side Comparison":
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Original Scan")
            st.image(image, width=420)
        with col2:
            st.subheader("🔥 Grad-CAM Attention Overlay")
            st.image(blended_overlay, width=420)
    elif analysis_mode == "Clinical View (Heatmap Overlay)":
        st.subheader("🔥 Grad-CAM Clinical Attention Overlay")
        st.image(blended_overlay, width=500, caption="Highlighted regions indicate primary visual features driving the prediction.")
    else:
        st.subheader("🖼️ Uploaded Scan")
        st.image(image, width=500)

    # Clinical Recommendation Banner
    st.markdown("### 📋 Diagnostic Decision Support & Next Steps")
    if pred_class == 1:
        st.error(
            "⚠️ High Probability of Renal Calculi (Kidney Stone) Detected\n\n"
            "Clinical Recommendations:\n"
            "1. Verify bright focal echoes or acoustic shadows highlighted by the Grad-CAM attribution map.\n"
            "2. Correlate findings with patient history (hematuria, flank pain, renal colic).\n"
            "3. Order supplementary non-contrast CT or urinalysis for stone composition analysis."
        )
    else:
        st.success(
            "✅ Scan Appears Clear of Major Calculi Signatures\n\n"
            "Clinical Recommendations:\n"
            "1. No significant hyper-reflective stone signatures detected in primary renal calyces.\n"
            "2. If clinical symptoms persist, consider repeating scan or testing for non-radiopaque matrix stones."
        )

    # Export Audit Report Option
    st.markdown("---")
    report_data = {
        "prediction": "Kidney Stone Detected" if pred_class == 1 else "Normal / No Stone",
        "confidence_percentage": round(confidence, 2),
        "stone_probability_percent": round(stone_prob, 2),
        "normal_probability_percent": round(normal_prob, 2),
        "model_architecture": "EfficientNet-B0 (Clinical Fine-Tuned)",
        "explainability": "Grad-CAM visual attention mapping generated successfully"
    }
    
    st.download_button(
        label="📥 Download Clinical Audit JSON Report",
        data=json.dumps(report_data, indent=4),
        file_name="clinical_audit_report.json",
        mime="application/json"
    )

else:
    st.info("👈 Please upload a medical scan image using the sidebar to begin analysis.")