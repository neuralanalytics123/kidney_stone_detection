# NephroAI: Enterprise Clinical Kidney Stone Diagnostic & Model Monitoring Suite

## 1. Executive Summary & Clinical Problem Solved

**NephroAI** is an advanced, enterprise-grade AI diagnostic tool built specifically for hospitals, emergency rooms, and radiology departments to assist doctors in detecting kidney stones.

### The Problem It Solved

Kidney stone disease (nephrolithiasis) causes severe pain and affects millions of people globally. Catching these stones quickly and accurately on ultrasound or CT scans is crucial to prevent dangerous complications like severe blockages, kidney swelling, infections, and permanent kidney damage.

However, manual image review can be slow and challenging. Doctors often face heavy workloads, eye fatigue, and delays in busy emergency rooms, which can lead to missed details.

NephroAI acts as a reliable second-opinion co-pilot. It instantly reviews patient scans, highlights potential kidney stones using visual heatmaps so doctors can see *why* the AI made its decision, extracts key image measurements, and automatically flags uncertain cases for review by a senior radiologist.

### Clinical Accuracy & Performance Metrics

Trained and validated on a balanced renal imaging corpus comprising **1,787 Non-Stone** and **1,577 Stone** clinical images, NephroAI achieves high diagnostic rigor on hold-out test evaluations:

* **Sensitivity / Recall:** $\ge 95.5\%$ (minimizing dangerous false negatives and missed calculi)
* **Specificity:** $\ge 94.0\%$ (ensuring reliable filtering of normal renal parenchyma)
* **Precision (PPV):** $\ge 94.8\%$
* **ROC-AUC Score:** $0.9780+$
* **Inference Latency:** $< 45\text{ ms}$ per scan on standard CPU/GPU hardware

---

## 2. Project Structure

```text
kidney_stone_detection/
├── .github/
│   └── workflows/
│       └── ci_cd.yml
├── artifacts/
│   ├── models/
│   │   └── kidney_stone_resnet50_best.pt  # Trained PyTorch model weights
│   └── reports/
│       ├── clinical_test_report.json      # Hold-out test metrics (Sensitivity, Specificity, AUC)
│       ├── training_history.json          # Epoch-by-epoch loss & accuracy convergence
│       └── inference_audit_log.jsonl      # Persistent audit trail and telemetry log
├── configs/
│   └── config.yaml                        # Hyperparameters, augmentation rules, and paths
├── dashboard/
│   └── app.py                             # Unified 3-tab Streamlit clinical UI
├── data/
│   ├── processed/                         # test_manifest.csv, train_manifest.csv, val_manifest.csv
│   └── raw/                               # Non-Stone (1,787 images) and Stone (1,577 images)
├── src/
│   └── kidney_ai/
│       ├── data_preprocessing/            # Image normalization and dataset splitting
│       ├── feature_engineering/           # Medical CLAHE transform & radiomics extraction
│       ├── training/                      # Model training loops and checkpointing
│       ├── testing/                       # Hold-out evaluation and confusion matrix generation
│       ├── validation/                    # Cross-validation and validation loss metrics
│       ├── models/                        # Network architectures (ResNet50 / EfficientNet) & Grad-CAM
│       ├── tests/                         # Unit tests and regression testing suites
│       ├── monitor/                       # ClinicalModelMonitor & audit logger (PSI drift)
│       └── deploying/                     # Deployment helper utilities
├── Dockerfile                             # Single-port Python 3.11-slim container spec
├── docker-compose.yml                     # Docker compose orchestration with volume bind-mounts
├── pyproject.toml                         # Project metadata, dependencies, and hatchling build setup
├── uv.lock                                # Locked dependency manifest managed by Astral uv
└── README.md                              # Enterprise system documentation
```

---

## 3. Tech Stack Used

* **Core Programming Language:** Python 3.11 (`python:3.11-slim`)
* **Deep Learning & Neural Networks:** PyTorch 2.2+, Torchvision (`ResNet50` & `EfficientNet` backbones)
* **Medical Computer Vision & Radiomics:** OpenCV (`opencv-python-headless`), Pillow (`PIL`), NumPy
* **Explainable AI (XAI):** Custom Gradient-weighted Class Activation Mapping (`Grad-CAM`)
* **Clinical Web Interface & Visualization:** Streamlit 1.32+, Plotly Express & Graph Objects, Pandas
* **MLOps, Monitoring & Drift Detection:** Custom `ClinicalModelMonitor` (JSONL audit logging & Population Stability Index `PSI`)
* **Containerization & Environment Management:** Docker, Docker Compose, Astral `uv` Package Manager, GitHub Actions CI/CD

### Technical Tools & Models Explained Simply

For readers, clinicians, or stakeholders who may not be familiar with machine learning terminology, here is what each core tool inside this project does in plain English:

* **ResNet50 (Deep Learning Brain):** A powerful image-recognition neural network trained to spot tiny bright spots and shadows inside kidney scans that indicate stones, while ignoring normal healthy tissue.
* **EfficientNet (Lightweight AI Model):** A fast, compact version of an image-recognition AI designed to run smoothly on smaller hospital computers or portable bedside ultrasound tablets without needing expensive graphics cards.
* **Explainable AI / Grad-CAM (Visual Heatmap):** Instead of just saying "Yes, there is a stone," this tool draws a glowing red/yellow heat map directly over the scan to show doctors *where* the AI is looking—building clinical trust and preventing "black-box" guesswork.
* **Medical CLAHE (Smart Image Enhancer):** Ultrasound and CT scans are often dark, grainy, or blurry. CLAHE acts like an intelligent medical brightness and contrast filter that sharpens the edges of hidden kidney stones without making the background fuzzy.
* **Radiomics (Digital Scan Measurements):** Software that automatically measures mathematical traits of the scan—such as average brightness, contrast, and texture—to give doctors objective numbers alongside the visual image.
* **PSI Drift Monitoring (Quality Watchdog):** A background safety monitor that checks if new hospital scanner machines are producing darker or different-looking images over time, automatically warning engineers if the AI needs recalibration.
* **Docker & `uv` (Portable Packaging):** Docker wraps the entire software, AI models, and Python libraries into a single self-contained box that runs identically on any hospital server or laptop with one command.

---

## 4. Technical Architecture & How It Works

NephroAI is built on a streamlined, single-port enterprise architecture that runs deep learning inference, image enhancement, explainability, and drift telemetry in-memory without microservice latency.

1. **Scan Upload & Ingestion:** A doctor or technician uploads a patient kidney ultrasound or CT scan (`.jpg`, `.png`, `.bmp`) into Tab 1 of the web interface.
2. **Contrast Enhancement & Radiomics (`MedicalCLAHETransform`):** OpenCV converts the scan into LAB color space and applies Contrast Limited Adaptive Histogram Equalization ($clip\_limit = 2.0$, $tile\_grid\_size = 8 \times 8$) strictly on the luminance channel, while `extract_radiomic_summary` logs pixel intensity statistics.
3. **AI Diagnosis (`ResNet50` / `EfficientNet`):** PyTorch passes the normalized image tensor through the neural network to calculate exact probability scores for `Normal Parenchyma` (Class `0`) vs. `Kidney Stone Detected` (Class `1`).
4. **Visual Attention Mapping (`GradCAMExplainer`):** Gradients from the final convolutional layer (`layer4`) are weighted and overlaid onto the original scan to highlight the exact calcification location.
5. **Audit Logging & Triage (`ClinicalModelMonitor`):** Every prediction is saved to `artifacts/reports/inference_audit_log.jsonl`. Low-confidence scans are automatically flagged for senior radiologist review, and Population Stability Index ($\text{PSI} > 0.20$) checks monitor for scanner drift.

---

## 5. Local Installation & Execution

### Prerequisites

* Linux workstation (Dell Linux / Ubuntu recommended) or Docker Desktop.
* Docker and Docker Compose installed.

### Step-by-Step Local Run

1. Navigate to your project directory:
   ```bash
   cd ~/Documents/kidney_stone_detection
   ```
2. Build and launch the containerized single-port Streamlit application:
   ```bash
   docker compose up --build -d
   ```
3. Access the live clinical dashboard in your browser:
   ```text
   http://localhost:8501
   ```

---

## 6. Clinical Deployment & Hospital Production Guide

Deploying NephroAI for live hospital operations requires strict adherence to data security, high availability, and network encryption. Follow these production steps:

1. **Server Provisioning:** Deploy to a dedicated on-premise hospital Linux server or private Cloud instance (AWS EC2 / GCP Compute Engine) equipped with minimum 8 GB RAM and persistent storage.
2. **Transfer Project Files:** Copy the production codebase and trained model weights (`artifacts/models/kidney_stone_resnet50_best.pt`) to the server. *(Note: Raw training image folders under `data/` are omitted from production servers to conserve disk space).*
3. **Background Daemonization:** Run the application container in detached production mode:
   ```bash
   docker compose up -d --build
   ```
4. **Secure HTTPS Reverse Proxy:** Configure an **Nginx** reverse proxy in front of port `8501` paired with a **Let's Encrypt SSL certificate (`certbot`)** to ensure all clinician-patient web traffic is encrypted via TLS (`https://`).
5. **Data Privacy Guarantee:** Patient scans uploaded to the UI are processed entirely in-memory (`RAM`) and are never written to permanent disk storage within the container, ensuring full alignment with medical privacy standards (HIPAA and GDPR).

---

## 7. Troubleshooting & Engineering Fixes

During the end-to-end development and containerization of NephroAI, several critical technical challenges were successfully resolved:

* **Issue 1: Port Conflicts on Port 8501 (`Address already in use`)**
  * *Cause:* An orphaned container or lingering local Streamlit process was occupying port `8501`.
  * *Fix:* Terminated the conflicting host process and cleaned up Docker networks:
    ```bash
    sudo kill -9 23054
    sudo fuser -k 8501/tcp
    docker compose down
    ```

* **Issue 2: Multi-Port Complexity & Redundant FastAPI Backend**
  * *Cause:* Initial microservice splitting across FastAPI (port `8000`) and Streamlit (port `8501`) introduced unnecessary orchestration overhead and latency.
  * *Fix:* Completely deprecated the FastAPI backend, consolidating the architecture into a **single, high-performance Streamlit application running exclusively on port `8501`**.

* **Issue 3: Missing OpenCV Graphical Shared Libraries (`libGL.so.1`)**
  * *Cause:* Minimal Python Docker base images (`python:3.11-slim`) lack native C++ rendering libraries required by OpenCV (`cv2`).
  * *Fix:* Added system dependencies directly to the `Dockerfile`:
    ```dockerfile
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*
    ```

* **Issue 4: Obsolete `version` Attribute in Docker Compose**
  * *Cause:* Modern Docker Compose runtimes deprecate top-level `version: "3.9"` declarations, triggering warning logs.
  * *Fix:* Cleaned up `docker-compose.yml` by removing the obsolete version key.

* **Issue 5: Missing JSON Report Dictionary Keys**
  * *Cause:* Discrepancies between metric naming conventions (`precision` vs `ppv`, `sensitivity_recall` vs `recall`) caused runtime `KeyError` exceptions on Tab 2.
  * *Fix:* Implemented robust safe dictionary lookups (`report.get('precision', report.get('ppv', 0.0))`) across all metric rendering components.

---

## 8. Future Advancements

* **Multi-Modal Fusion:** Integrating structured patient clinical metadata (age, hematuria status, serum creatinine, urinalysis) alongside imaging tensors into a hybrid fusion head.
* **Native DICOM Support:** Direct PACS integration and dynamic Hounsfield Unit (HU) windowing for raw hospital `.dcm` CT volumes.
* **TensorRT / ONNX Edge Acceleration:** Quantizing model weights to FP16/INT8 for sub-15 ms inference on portable bedside ultrasound devices.

---

## 9. Collaboration & Contact

For enterprise licensing, clinical validation inquiries, research collaborations, or technical contributions, please contact the lead maintainer:

* **Email:** [neuralrush82@gmail.com](mailto:neuralrush82@gmail.com)