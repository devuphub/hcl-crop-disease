# 🌾 Crop Disease Detection Using Convolutional Neural Networks (CNN)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 📌 Executive Summary

**Crop Disease Detection Using Convolutional Neural Networks** is an end-to-end, production-ready AI/ML agricultural diagnosis and pathology classification platform. Built using **PyTorch**, **Streamlit**, and **FastAPI**, the system diagnoses plant diseases from leaf imagery in under **35 milliseconds**, localizes disease lesions using **Explainable AI (Grad-CAM)**, and automatically generates an **Agronomic Health & Treatment Advisory Report (PDF)** containing targeted organic and chemical treatments.

---

## 🌟 Key Features

- **⚡ End-to-End Deep Learning Pipeline**: Multi-block Convolutional Neural Network (CNN) trained with data augmentation, learning rate scheduling, and regularization.
- **🔬 Explainable AI (Grad-CAM)**: Visualizes gradient-weighted class activation heatmaps so users and agronomists can verify the exact foliar lesions triggering the diagnosis.
- **🌱 Comprehensive Multi-Crop Support**: Detects 18 distinct pathological and healthy classes across 5 staple crops: **Apple, Corn (Maize), Potato, Rice, and Tomato**.
- **📋 Automated Agronomic Advisory**: Provides detailed causal pathogens, environmental disease triggers, organic bio-control steps, chemical fungicide schedules, and long-term prevention protocols.
- **📄 Instant PDF Advisory Export**: One-click generation of downloadable health certificates formatted for farmers, agronomists, and field scouts.
- **🚀 Dual Deployment Interfaces**:
  - **Interactive Streamlit Web Dashboard**: Drag & drop upload, live webcam capture, and curated sample gallery.
  - **FastAPI Production REST Service**: Asynchronous endpoints (`/predict`, `/classes`, `/health`) with auto-generated Swagger documentation.
- **🐳 Production Ready**: Complete Dockerfile, Docker Compose, and cloud PaaS compatibility (Render, Streamlit Cloud, Hugging Face Spaces).

---

## 🌾 Supported Crops & Pathology Classes

The system is trained and configured for **18 agricultural classes**:

| Crop | Disease / Class Name | Pathogen Type | Typical Symptoms |
| :--- | :--- | :--- | :--- |
| **Apple** | `Apple Scab` | Fungus (*Venturia inaequalis*) | Olive-green velvety lesions on leaves & fruit |
| **Apple** | `Black Rot (Frogeye)` | Fungus (*Botryosphaeria obtusa*) | Circular lesions with purple borders & tan center |
| **Apple** | `Cedar Apple Rust` | Fungus (*Gymnosporangium*) | Bright yellow-orange pustules on leaf surface |
| **Apple** | `Healthy` | None | Vibrant green foliage, intact cellular structure |
| **Corn** | `Common Rust` | Fungus (*Puccinia sorghi*) | Powdery cinnamon-brown pustules on both leaf faces |
| **Corn** | `Northern Leaf Blight` | Fungus (*Exserohilum turcicum*) | Long, elliptical cigar-shaped grayish-tan lesions |
| **Corn** | `Healthy` | None | Vigorous parallel venation, uniform chlorophyll |
| **Potato** | `Early Blight` | Fungus (*Alternaria solani*) | Dark brown spots with concentric target rings |
| **Potato** | `Late Blight` | Oomycete (*Phytophthora infestans*) | Water-soaked rotting lesions with white sporulation |
| **Potato** | `Healthy` | None | Clean compound foliage, sturdy canopy |
| **Rice** | `Brown Spot` | Fungus (*Bipolaris oryzae*) | Oval sesame-seed shaped brown spots with yellow halo |
| **Rice** | `Leaf Blast` | Fungus (*Magnaporthe oryzae*) | Spindle/diamond shaped lesions with gray center |
| **Rice** | `Healthy` | None | Upright green blades, sturdy tillering |
| **Tomato** | `Bacterial Spot` | Bacteria (*Xanthomonas*) | Angular water-soaked lesions with yellow halos |
| **Tomato** | `Early Blight` | Fungus (*Alternaria solani*) | Bullseye concentric rings on lower leaves, defoliation |
| **Tomato** | `Late Blight` | Oomycete (*Phytophthora*) | Rapid foliar browning, dark water-soaked rot |
| **Tomato** | `Yellow Leaf Curl Virus` | Begomovirus (Whitefly vector) | Severe chlorosis, upward leaf curling, stunting |
| **Tomato** | `Healthy` | None | Deep green, crisp leaf margins, normal flowers |

---

## 🧠 CNN Model Architecture

```
Input Image (3 x 224 x 224)
   │
   ▼
[ConvBlock 1]: Conv2d(3->32, 3x3) -> BatchNorm2d -> ReLU -> Conv2d(32->32) -> MaxPool(2x2) -> Dropout(0.15)
   │
   ▼
[ConvBlock 2]: Conv2d(32->64, 3x3) -> BatchNorm2d -> ReLU -> Conv2d(64->64) -> MaxPool(2x2) -> Dropout(0.20)
   │
   ▼
[ConvBlock 3]: Conv2d(64->128, 3x3) -> BatchNorm2d -> ReLU -> Conv2d(128->128) -> MaxPool(2x2) -> Dropout(0.25)
   │
   ▼
[ConvBlock 4]: Conv2d(128->256, 3x3) -> BatchNorm2d -> ReLU -> Conv2d(256->256) -> MaxPool(2x2) -> Dropout(0.30)
   │
   ▼
[Global Average Pooling (GAP)]: AdaptiveAvgPool2d(1, 1) -> (Batch, 256)
   │
   ▼
[Dense Classifier Head]:
   ├── Linear(256 -> 128)
   ├── BatchNorm1d(128)
   ├── ReLU(inplace=True)
   ├── Dropout(p=0.35)
   └── Linear(128 -> 18)
   │
   ▼
Softmax Probability Distribution (18 Classes)
```

### Architectural Highlights:
1. **Batch Normalization**: Applied after convolutions and fully-connected layers to accelerate convergence and prevent internal covariate shift.
2. **Global Average Pooling (GAP)**: Replaces bulky flatten layers, drastically reducing parameters (down to ~1.2M parameters) and mitigating spatial overfitting.
3. **Regularization**: Progressive spatial dropout (0.15 to 0.30) and classifier dropout (0.35) ensure robust generalization across varying field lighting and angles.

---

## 📁 Project Directory Structure

```
hcl-crop-disease/
│
├── data/
│   ├── dataset/                      # 18 organized crop pathology class folders
│   ├── sample_test_images/           # Curated test images for 1-click evaluation
│   └── disease_info.json             # Comprehensive agronomic treatment knowledge base
│
├── models/
│   ├── cnn_model.py                  # Custom CropDiseaseCNN & MobileNetV2 architectures
│   ├── gradcam.py                    # Explainable AI (XAI) Grad-CAM implementation
│   ├── predictor.py                  # Production inference pipeline with treatment lookup
│   └── saved_models/
│       ├── crop_disease_model.pth    # Trained PyTorch model checkpoint
│       ├── class_names.json          # Label index mapping
│       ├── metrics.json              # Training convergence metadata
│       ├── training_curves.png       # Loss and accuracy plots
│       ├── confusion_matrix.png      # 18-class confusion matrix
│       └── evaluation_report.json    # Precision, recall, and F1 metrics
│
├── src/
│   ├── dataset_builder.py            # Generates balanced dataset & test samples
│   ├── train.py                      # Training script with AdamW & LR scheduling
│   ├── evaluate.py                   # Evaluation script generating confusion matrix
│   └── report_generator.py           # ReportLab PDF advisory certificate generator
│
├── web/
│   ├── app.py                        # Streamlit web application
│   └── style.css                     # Custom Agri-Tech CSS styling
│
├── api/
│   └── main.py                       # FastAPI REST API service
│
├── tests/
│   └── test_pipeline.py              # Unit & integration test suite (7/7 passing)
│
├── .streamlit/
│   └── config.toml                   # Streamlit server and theme configurations
│
├── Dockerfile                        # Multi-stage production container configuration
├── docker-compose.yml                # Microservices orchestration (Web + API)
├── requirements.txt                  # Python dependencies
├── Procfile                          # Cloud PaaS deployment file (Render/Heroku)
├── run_app.bat                       # 1-Click Windows launcher for Web App
├── run_api.bat                       # 1-Click Windows launcher for REST API
├── PROJECT_REPORT.md                 # Full academic project documentation
└── README.md                         # Comprehensive documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10 or 3.11** installed on your system.
- Git (optional).

### 2. Setup Virtual Environment
```bash
# Clone or navigate to the project directory
cd hcl-crop-disease

# Create a virtual environment
py -3.11 -m venv venv

# Activate the virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the Web Application (Streamlit)
**Method A: 1-Click Launcher (Windows)**
Double click on `run_app.bat`.

**Method B: Command Line**
```bash
streamlit run web/app.py
```
Open your browser at `http://localhost:8501`.

---

### 4. Launch the REST API (FastAPI)
**Method A: 1-Click Launcher (Windows)**
Double click on `run_api.bat`.

**Method B: Command Line**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Base: `http://localhost:8000`
- Interactive Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 📡 REST API Reference

### 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```
**Response:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "num_classes": 18,
  "model_loaded": true
}
```

### 2. Predict Disease from Leaf Image
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/sample_test_images/Tomato___Early_blight_sample.jpg" \
     -F "top_k=3"
```
**Response (Sample):**
```json
{
  "status": "success",
  "filename": "Tomato___Early_blight_sample.jpg",
  "primary_prediction": {
    "crop": "Tomato",
    "disease": "Early Blight",
    "confidence": 96.42,
    "is_healthy": false
  },
  "severity": "Moderate to High",
  "status_badge": "High Alert",
  "advisory": {
    "pathogen": "Alternaria solani (Fungus)",
    "cause": "Warm temperatures (24-29°C) and alternating wet and dry weather.",
    "symptoms": "Brown to black lesions with concentric bullseye rings on oldest bottom leaves.",
    "organic_treatment": [
      "Apply organic bio-fungicide Bacillus subtilis or Trichoderma viride.",
      "Mulch soil around plants with straw to prevent soil splash onto leaves."
    ],
    "chemical_treatment": [
      "Foliar spray with Chlorothalonil (2 g/L) or Mancozeb (2.5 g/L)."
    ],
    "prevention": [
      "Stake and prune tomato vines to keep foliage well above the ground.",
      "Use drip irrigation rather than overhead sprinklers."
    ]
  }
}
```

---

## 🧪 Testing & Model Evaluation

### Run Test Suite
The project comes with a comprehensive test suite covering the CNN forward pass, MobileNet backbone, Grad-CAM heatmap, PDF generator, and REST API:
```bash
python -m unittest tests/test_pipeline.py
```
*Result: 7/7 tests pass cleanly.*

### Re-Train or Fine-Tune Model
To train the CNN model on new or expanded dataset:
```bash
python src/train.py --epochs 15 --batch_size 32 --lr 0.001 --model cnn
```

### Generate Confusion Matrix & Metrics
To compute precision, recall, F1-scores, and generate the confusion matrix:
```bash
python src/evaluate.py
```
Visual artifacts will be saved in `models/saved_models/`.

---

## 🐳 Docker Deployment

### Using Docker Compose
Build and run both the Web App and REST API simultaneously:
```bash
docker-compose up --build
```
- Access Web Application: `http://localhost:8501`
- Access REST API Docs: `http://localhost:8000/docs`

### Using Docker Single Container
```bash
docker build -t crop-disease-ai .
docker run -p 8501:8501 crop-disease-ai
```

---

## ☁️ Cloud PaaS Deployment

### Streamlit Community Cloud (Free & Direct)
1. Push this repository to your GitHub account.
2. Log in to [share.streamlit.io](https://share.streamlit.io).
3. Click **"New App"** -> Select this repository.
4. Set Main file path: `web/app.py`.
5. Click **"Deploy"**. The app will automatically install dependencies and launch!

### Render / Hugging Face Spaces
- **Render**: Connect repository, select **Web Service**, environment **Docker** or **Python 3**, start command `streamlit run web/app.py --server.port=$PORT --server.address=0.0.0.0`.
- **Hugging Face Spaces**: Select **Streamlit SDK**, upload repository, and your app will be live globally.

---

## 🎓 Viva & Technical Interview Q&A

### Q1: Why use Convolutional Neural Networks (CNN) instead of traditional Machine Learning (e.g., Random Forest or SVM)?
> **Answer**: Traditional ML requires manual feature engineering (e.g., manually computing color histograms, Haralick texture features, or edge detection using Canny/Sobel filters). CNNs automatically learn hierarchical representations: early convolutional layers learn low-level features (edges, color gradients), intermediate layers learn textures and lesion borders, and deep layers capture complex pathological patterns (concentric rings, pustules, necrotic spots) directly from raw pixel values with translation invariance.

### Q2: What is the role of Batch Normalization in the CNN architecture?
> **Answer**: Batch Normalization normalizes the activation outputs of a convolutional layer across each mini-batch (mean $\approx 0$, variance $\approx 1$). This stabilizes the learning process, prevents vanishing/exploding gradients, mitigates internal covariate shift, acts as a mild regularizer, and allows the model to converge faster at higher learning rates.

### Q3: What is Grad-CAM and why is it crucial for agricultural AI?
> **Answer**: Grad-CAM (Gradient-weighted Class Activation Mapping) is an Explainable AI (XAI) technique. It calculates the gradients of the target class score with respect to the feature activation maps of the final convolutional layer. By performing global average pooling on these gradients, we obtain weights that highlight which spatial regions of the leaf image most strongly influenced the neural network's classification. In agriculture, this builds farmer trust by proving that the AI is identifying actual pathological lesions rather than irrelevant background soil or shadows.

### Q4: Why use Global Average Pooling (GAP) instead of fully flattening the feature maps?
> **Answer**: Flattening high-dimensional feature maps (e.g., $256 \times 14 \times 14 = 50,176$ values) into dense layers requires millions of weights, causing massive memory consumption and a high risk of overfitting. GAP averages each feature channel across its entire spatial dimension, condensing the tensor directly to $256$ values. This makes the network invariant to slight translations, reduces parameter count by over 80%, and creates a direct correspondence between feature maps and categories.

### Q5: How do you address class imbalance and data scarcity in field conditions?
> **Answer**: 
> 1. **Data Augmentation**: We apply random horizontal/vertical flips, affine rotations ($\pm 20^\circ$), color jittering (brightness, contrast, saturation), and random resized cropping.
> 2. **Regularization**: Progressive spatial dropout and weight decay in AdamW optimizer.
> 3. **Transfer Learning**: The architecture supports loading MobileNetV2 pretrained on ImageNet for scenarios with limited target field samples.

---

## 📜 License
This project is open-source and licensed under the [MIT License](LICENSE).
